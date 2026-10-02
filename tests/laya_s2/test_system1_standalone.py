"""Standalone DualVLN System 1: export from a (fake) checkpoint and reproduce the original trajectories."""

import argparse
import json
import os
import sys
import types

import pytest
import torch

pytest.importorskip("diffusers")
# internnav.model.encoder/__init__ pulls in unrelated heavy deps (gym, LongCLIP submodule, ...); when they are
# missing, expose the package directory only so that the depth_anything subpackage can still be imported
try:
    import internnav.model.encoder  # noqa: F401
except ModuleNotFoundError:
    import internnav.model

    for k in [k for k in sys.modules if k.startswith("internnav.model.encoder")]:
        del sys.modules[k]
    _enc = types.ModuleType("internnav.model.encoder")
    _enc.__path__ = [os.path.join(os.path.dirname(internnav.model.__file__), "encoder")]
    sys.modules["internnav.model.encoder"] = _enc
from safetensors.torch import save_file  # noqa: E402

from internnav.model.basemodel.internvla_n1.system1_standalone import (  # noqa: E402
    DualVLNSystem1,
    export_system1,
)


@torch.no_grad()
def test_export_roundtrip_and_projected_latents(tmp_path):
    torch.manual_seed(0)
    cfg = dict(system1="nextdit_async", n_query=4, hidden_size=3584)
    ref = DualVLNSystem1(argparse.Namespace(**cfg)).eval()

    # fake DualVLN checkpoint: System 1 tensors under "model." plus Qwen tensors that must be skipped
    src = tmp_path / "dualvln"
    src.mkdir()
    sd = {f"model.{k}": v.contiguous() for k, v in ref.model.state_dict().items()}
    sd["model.layers.0.mlp.weight"] = torch.zeros(2, 2)
    sd["visual.blocks.0.weight"] = torch.zeros(2, 2)
    save_file(sd, str(src / "model-00001-of-00001.safetensors"))
    (src / "config.json").write_text(json.dumps(dict(cfg, model_type="internvla_n1")))

    out = tmp_path / "s1"
    export_system1(str(src), str(out))
    s1 = DualVLNSystem1.from_pretrained(str(out), dtype=torch.float32)
    assert not any(k.startswith(("layers.", "visual.")) for k in torch.load(out / "system1.pt"))

    lat = torch.randn(1, 4, 3584)
    imgs = torch.rand(1, 2, 224, 224, 3)
    deps = torch.rand(1, 2, 224, 224, 1)
    torch.manual_seed(1)
    a = ref.generate_traj(lat, imgs, deps, num_sample_trajs=4)
    torch.manual_seed(1)
    b = s1.generate_traj(lat, imgs, deps, num_sample_trajs=4)
    torch.manual_seed(1)
    c = s1.generate_traj(s1.get_model().cond_projector(lat), imgs, deps, num_sample_trajs=4, latents_projected=True)
    assert a.shape == (4, 32, 3)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(a, c)
