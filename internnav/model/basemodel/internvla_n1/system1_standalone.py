"""System 1 of InternVLA-N1 without the Qwen2.5-VL System 2.

Lets a lightweight System 2 (e.g. Laya-S2) drive the trained DualVLN System 1 without loading the 7B
checkpoint. Weights are copied straight out of the DualVLN safetensors, and ``generate_traj`` is the
very same function as ``InternVLAN1ForCausalLM.generate_traj``.

Export (reads only the System 1 tensors, no 7B load):
    python -m internnav.model.basemodel.internvla_n1.system1_standalone \
        --src checkpoints/InternVLA-N1-DualVLN --out checkpoints/DualVLN-System1 [--verify]
"""

import argparse
import json
import os
from unittest import mock

import torch
import torch.nn as nn

from . import internvla_n1_arch as arch
from .internvla_n1 import _RESNET_MEAN, _RESNET_STD, InternVLAN1ForCausalLM
from .internvla_n1_arch import InternVLAN1MetaForCausalLM, InternVLAN1MetaModel

S1_PREFIXES = (
    "latent_queries",
    "traj_dit.",
    "action_encoder.",
    "pos_encoding.",
    "action_decoder.",
    "cond_projector.",
    "rgb_model.",
    "memory_encoder.",
    "rgb_resampler.",
    "navdp.",
)
CONFIG_NAME = "system1_config.json"
WEIGHTS_NAME = "system1.pt"


class _ConfigModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


def _depthanything_arch_only(config):
    # architecture only: the weights come from the DualVLN checkpoint, not the original DAv2 file
    from internnav.model.encoder.depth_anything.depth_anything_v2.dpt import (
        DepthAnythingV2,
    )

    return DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384]).pretrained


class DualVLNSystem1Model(InternVLAN1MetaModel, _ConfigModule):
    def __init__(self, config):
        with mock.patch.object(arch, "build_depthanythingv2", _depthanything_arch_only):
            super().__init__(config)


class DualVLNSystem1(nn.Module):
    generate_traj = InternVLAN1ForCausalLM.generate_traj
    get_system1_type = InternVLAN1MetaForCausalLM.get_system1_type

    def __init__(self, config):
        super().__init__()
        self.model = DualVLNSystem1Model(config)
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

    def get_model(self):
        return self.model

    @property
    def device(self):
        return self._resnet_mean.device

    @classmethod
    def from_pretrained(cls, path, device="cpu", dtype=torch.bfloat16):
        with open(os.path.join(path, CONFIG_NAME)) as f:
            config = argparse.Namespace(**json.load(f))
        model = cls(config)
        model.model.load_state_dict(torch.load(os.path.join(path, WEIGHTS_NAME), map_location="cpu"))
        model = model.to(device=device, dtype=dtype).eval()
        # the full model creates these with torch.FloatTensor, so they stay fp32 under from_pretrained(bf16);
        # normalizing in bf16 instead shifts the trajectories (~0.1)
        model._resnet_mean = model._resnet_mean.float()
        model._resnet_std = model._resnet_std.float()
        return model


def export_system1(src: str, out: str):
    from safetensors import safe_open

    with open(os.path.join(src, "config.json")) as f:
        full_cfg = json.load(f)
    cfg = dict(system1=full_cfg["system1"], n_query=full_cfg["n_query"], hidden_size=full_cfg["hidden_size"])
    sd = {}
    for fn in sorted(os.listdir(src)):
        if not fn.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(src, fn), framework="pt") as f:
            for k in f.keys():
                if k.startswith("model.") and k[len("model.") :].startswith(S1_PREFIXES):
                    sd[k[len("model.") :]] = f.get_tensor(k)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, CONFIG_NAME), "w") as f:
        json.dump(cfg, f, indent=2)
    torch.save(sd, os.path.join(out, WEIGHTS_NAME))
    n = sum(v.numel() for v in sd.values())
    print(f"exported {len(sd)} tensors ({n / 1e6:.1f}M params, system1={cfg['system1']}) -> {out}")


def _compare_weights(full, s1, out):
    """System 1 tensors: checkpoint file vs full model vs standalone."""
    file_sd = torch.load(os.path.join(out, WEIGHTS_NAME), map_location="cpu")
    full_sd = full.get_model().state_dict()
    rows = []
    for k, v in s1.model.state_dict().items():
        f = full_sd[k].detach().float().cpu()
        a = v.detach().float().cpu()
        ref = file_sd[k].float()
        rows.append((k, (f - a).abs().max().item(), (f - ref).abs().max().item(), (a - ref).abs().max().item(), full_sd[k].dtype, v.dtype))
    bad = sorted([r for r in rows if r[1] > 0], key=lambda r: -r[1])
    print(f"[weights] {len(rows)} tensors, {len(bad)} differ between full and standalone")
    by_mod = {}
    for r in bad:
        by_mod.setdefault(".".join(r[0].split(".")[:2]), []).append(r)
    for mod, rs in by_mod.items():
        print(f"  {mod}: {len(rs)} tensors, max |full-standalone| {max(r[1] for r in rs):.3e}, "
              f"max |full-file| {max(r[2] for r in rs):.3e}, max |standalone-file| {max(r[3] for r in rs):.3e}")
    for r in bad[:10]:
        print(f"    {r[0]}: full-standalone {r[1]:.3e} full-file {r[2]:.3e} standalone-file {r[3]:.3e} ({r[4]} / {r[5]})")
    for name in ("_resnet_mean", "_resnet_std"):
        f, a = getattr(full, name), getattr(s1, name)
        print(f"[buffer] {name}: full {f.dtype} {f.flatten().tolist()} | standalone {a.dtype} {a.flatten().tolist()}")


@torch.no_grad()
def _compare_stages(full, s1, lat, imgs):
    """Intermediate System 1 tensors on identical inputs (mirrors generate_traj)."""

    def stages(m):
        gm = m.get_model()
        x = imgs.permute(0, 1, 4, 2, 3)
        xn = (x - m._resnet_mean) / m._resnet_std
        gm.rgb_model.to(lat.dtype)
        feat = gm.rgb_model.get_intermediate_layers(xn.flatten(0, 1).to(lat.dtype))[0].unflatten(dim=0, sizes=(1, -1))
        mem = gm.memory_encoder(feat.flatten(1, 2))
        tok = gm.rgb_resampler(torch.cat([feat.flatten(1, 2), mem], dim=-1))
        proj = gm.cond_projector(lat)
        g = torch.Generator(device=lat.device).manual_seed(0)
        noisy = torch.randn(1, 32, 3, generator=g, device=lat.device, dtype=lat.dtype)
        h = gm.action_encoder(noisy) + gm.pos_encoding(torch.arange(32, device=lat.device)[None])
        t = gm.noise_scheduler.timesteps[:1].to(lat.device)
        v = gm.action_decoder(gm.traj_dit(x=h, timestep=t, z_latents=torch.cat([tok, proj], 1)))
        return dict(normalized_input=xn, rgb_features=feat, memory_tokens=tok, cond_projector=proj, dit_velocity=v)

    a, b = stages(full), stages(s1)
    for k in a:
        print(f"[stage] {k:<17} max|full|={a[k].abs().max().item():.3e}  max|full-standalone|={(a[k].float() - b[k].float()).abs().max().item():.3e}  dtypes {a[k].dtype}/{b[k].dtype}")


@torch.no_grad()
def verify(src: str, out: str, device="cuda"):
    """Same seed + same inputs -> identical trajectories from the full model and the standalone one."""
    full = InternVLAN1ForCausalLM.from_pretrained(
        src, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map={"": device}
    ).eval()
    s1 = DualVLNSystem1.from_pretrained(out, device=device)
    n_query, hidden = full.config.n_query, full.config.hidden_size
    g = torch.Generator().manual_seed(0)
    lat = torch.randn(1, n_query, hidden, generator=g).to(device, torch.bfloat16)
    imgs = torch.rand(1, 2, 224, 224, 3, generator=g).to(device, torch.bfloat16)
    deps = torch.rand(1, 2, 224, 224, 1, generator=g).to(device, torch.bfloat16)
    _compare_weights(full, s1, out)
    _compare_stages(full, s1, lat, imgs)

    torch.manual_seed(1)
    a = full.generate_traj(lat, imgs, deps)
    torch.manual_seed(1)
    b = s1.generate_traj(lat, imgs, deps)
    torch.manual_seed(1)
    c = s1.generate_traj(full.get_model().cond_projector(lat), imgs, deps, latents_projected=True)
    torch.manual_seed(1)
    a2 = full.generate_traj(lat, imgs, deps)  # run-to-run noise of the GPU kernels, for reference
    print(f"trajectory scale max|full| = {a.abs().max().item():.3e}")
    print(f"max |full - full (rerun)|  = {(a - a2).abs().max().item():.3e}")
    print(f"max |full - standalone|    = {(a - b).abs().max().item():.3e}")
    print(f"max |full - standalone, projected latents| = {(a - c).abs().max().item():.3e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="checkpoints/InternVLA-N1-DualVLN")
    ap.add_argument("--out", default="checkpoints/DualVLN-System1")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    export_system1(args.src, args.out)
    if args.verify:
        verify(args.src, args.out)
