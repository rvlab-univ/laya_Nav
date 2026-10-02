"""Extract System 1 conditions from the DualVLN teacher for Laya-S2 distillation.

For every pixel-goal sample, the frozen InternVLA-N1 (DualVLN) System 2 reads the same conversation it
was trained on (history + current frame, "↓", look-down frame, GT goal "x y"), followed by n_query TRAJ
tokens. The last hidden states at the TRAJ positions are passed through ``cond_projector``; this
[n_query, 768] tensor is exactly what System 1 consumes, so the student is trained to reproduce it.

Usage (one node, 8 GPUs):
    torchrun --nproc_per_node=8 scripts/train/laya_s2/extract_teacher_latents.py \
        --teacher_path checkpoints/InternVLA-N1-DualVLN \
        --vln_dataset_use r2r_125cm_0_30,rxr_125cm_0_30 --out_dir data/laya_s2/teacher_latents

Output: <out_dir>/keys.txt and <out_dir>/latents.npy ([N, n_query, 768] fp16), keyed by ``sample_key``.
"""

import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from transformers import AutoProcessor, AutoTokenizer

from internnav.dataset.internvla_n1_lerobot_dataset import (
    TRAJ_TOKEN_INDEX,
    preprocess_qwen_2_visual,
)
from internnav.dataset.laya_s2_dataset import (
    KEY_FIELDS,
    frame_path,
    load_vln_samples,
    sample_key,
)
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM

PROMPT = (
    "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay "
    "on track? Please output the next waypoint's coordinates in the image. Please output STOP when you have "
    "successfully completed the task."
)
CONJ = "you can see "


class TeacherGoalDataset(Dataset):
    def __init__(self, samples, tokenizer, image_processor, num_history, resize):
        self.samples = samples
        self.tok = tokenizer
        self.ip = image_processor
        self.num_history = num_history
        self.resize = v2.Resize(resize)

    def __len__(self):
        return len(self.samples)

    def _proc(self, img):
        out = self.ip.preprocess(img, return_tensors="pt")
        return out["pixel_values"], out["image_grid_thw"][0]

    def __getitem__(self, i):
        s = self.samples[i]
        start = s["start"]
        hist_ids = np.unique(np.linspace(0, start - 1, self.num_history, dtype=np.int32)).tolist() if start else []
        frames = [self.resize(Image.open(frame_path(s, f)).convert("RGB")) for f in hist_ids + [start]]
        frames.append(Image.open(frame_path(s, start, look_down=True)).convert("RGB"))
        pix, thw = zip(*(self._proc(f) for f in frames))

        value = PROMPT.replace("<instruction>", s["instruction"])
        if start:
            value +=" These are your historical observations: " + "<image>\n" * len(hist_ids) + "."
        value += f" {CONJ}<image>."
        conv = [
            {"from": "human", "value": value},
            {"from": "gpt", "value": "↓"},
            {"from": "human", "value": f"{CONJ}<image>."},
            {"from": "gpt", "value": f"{s['goal'][0]} {s['goal'][1]}"},
        ]
        merged = [t.prod() // self.ip.merge_size**2 for t in thw]
        ids = preprocess_qwen_2_visual([conv], self.tok, grid_thw_image=merged)["input_ids"][0]
        return dict(
            input_ids=ids,
            pixel_values=torch.cat(pix, 0),
            image_grid_thw=torch.stack(thw, 0),
            key=sample_key(*(s[k] for k in KEY_FIELDS)),
        )


def make_collate(pad_id, n_query):
    def collate(batch):
        ids = [torch.cat([b["input_ids"], torch.full((n_query,), TRAJ_TOKEN_INDEX)]) for b in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=pad_id)
        attn = torch.arange(input_ids.shape[1])[None] < torch.tensor([len(x) for x in ids])[:, None]
        return dict(
            input_ids=input_ids,
            attention_mask=attn.long(),
            pixel_values=torch.cat([b["pixel_values"] for b in batch], 0),
            image_grid_thw=torch.cat([b["image_grid_thw"] for b in batch], 0),
            keys=[b["key"] for b in batch],
        )

    return collate


@torch.no_grad()
def teacher_latents(model, input_ids, attention_mask, pixel_values, image_grid_thw, n_query):
    """Same embedding construction as InternVLAN1ForCausalLM.forward, without the lm_head."""
    inner = model.get_model()
    emb = inner.embed_tokens(input_ids)
    img = model.visual(pixel_values.type(model.visual.dtype), grid_thw=image_grid_thw)
    img_mask = (input_ids == model.config.image_token_id)[..., None].expand_as(emb)
    emb = emb.masked_scatter(img_mask, img.to(emb.dtype))
    traj = input_ids == TRAJ_TOKEN_INDEX
    emb[traj] = inner.latent_queries.repeat(input_ids.shape[0], 1, 1).flatten(0, 1).to(emb.dtype)
    position_ids, _ = model.get_rope_index(input_ids, image_grid_thw, None, None, attention_mask)
    h = inner(
        input_ids=None,
        inputs_embeds=emb,
        position_ids=position_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    ).last_hidden_state
    start = traj.int().argmax(1)
    hs = torch.stack([h[b, start[b] : start[b] + n_query] for b in range(h.shape[0])])
    return inner.cond_projector(hs).float()


def merge_parts(out_dir, world):
    keys, sizes = [], []
    for r in range(world):
        with open(os.path.join(out_dir, f"part_{r}.keys")) as f:
            k = [x.rstrip("\n") for x in f]
        keys += k
        sizes.append(len(k))
    first = np.load(os.path.join(out_dir, "part_0.npy"), mmap_mode="r")
    out = np.lib.format.open_memmap(
        os.path.join(out_dir, "latents.npy"), mode="w+", dtype=np.float16, shape=(len(keys),) + first.shape[1:]
    )
    o = 0
    for r, n in enumerate(sizes):
        out[o : o + n] = np.load(os.path.join(out_dir, f"part_{r}.npy"), mmap_mode="r")
        o += n
    out.flush()
    with open(os.path.join(out_dir, "keys.txt"), "w") as f:
        f.write("\n".join(keys) + "\n")
    for r in range(world):
        os.remove(os.path.join(out_dir, f"part_{r}.npy"))
        os.remove(os.path.join(out_dir, f"part_{r}.keys"))
    print(f"merged {len(keys)} latents -> {out_dir}/latents.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_path", default="checkpoints/InternVLA-N1-DualVLN")
    ap.add_argument("--vln_dataset_use", required=True, help="same names as the InternVLA-N1 trainer, no %% rates")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--num_history", type=int, default=8)
    ap.add_argument("--resize", type=int, default=384)
    ap.add_argument("--n_query", type=int, default=4)
    ap.add_argument("--max_samples", type=int, default=-1, help="smoke test: only the first N goal samples")
    args = ap.parse_args()

    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    os.makedirs(args.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.teacher_path, use_fast=False)
    ip = AutoProcessor.from_pretrained(args.teacher_path).image_processor
    model = InternVLAN1ForCausalLM.from_pretrained(
        args.teacher_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device)
    model.eval()
    system1 = getattr(model.config, "system1", "")
    assert "nextdit" in system1, f"Laya-S2 distills cond_projector outputs (nextdit System 1 only), got {system1!r}"
    assert getattr(model.config, "n_query", args.n_query) == args.n_query

    samples = load_vln_samples(args.vln_dataset_use, pixel_goal_only=True)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    samples = samples[rank::world]
    ds = TeacherGoalDataset(samples, tok, ip, args.num_history, (args.resize, args.resize))
    dl = DataLoader(
        ds, args.batch_size, num_workers=args.num_workers, collate_fn=make_collate(tok.pad_token_id, args.n_query)
    )

    arr = np.lib.format.open_memmap(
        os.path.join(args.out_dir, f"part_{rank}.npy"), mode="w+", dtype=np.float16, shape=(len(ds), args.n_query, 768)
    )
    keys, o = [], 0
    for step, batch in enumerate(dl):
        lat = teacher_latents(
            model,
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["pixel_values"].to(device),
            batch["image_grid_thw"].to(device),
            args.n_query,
        )
        arr[o : o + lat.shape[0]] = lat.cpu().numpy().astype(np.float16)
        o += lat.shape[0]
        keys += batch["keys"]
        if rank == 0 and step % 50 == 0:
            print(f"[rank0] {o}/{len(ds)}", flush=True)
    arr.flush()
    with open(os.path.join(args.out_dir, f"part_{rank}.keys"), "w") as f:
        f.write("\n".join(keys) + ("\n" if keys else ""))

    if distributed:
        dist.barrier()
    if rank == 0:
        merge_parts(args.out_dir, world)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
