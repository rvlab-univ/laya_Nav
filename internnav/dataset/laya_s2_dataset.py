"""Dataset for the Laya-style lightweight System 2 (see internnav/model/basemodel/laya_s2).

Samples are enumerated exactly like ``NavPixelGoalDataset`` (pixel-goal / turn / stop rounds every
``sample_step`` frames), so that teacher latents extracted by
``scripts/train/laya_s2/extract_teacher_latents.py`` can be joined by ``sample_key``.
"""

import hashlib
import os
import random
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

GOAL, TURN, STOP = 0, 1, 2


def sample_key(video, ep_id, height, pitch_1, pitch_2, instruction, start_frame_id) -> str:
    ins = hashlib.md5(instruction.encode("utf-8")).hexdigest()[:12]
    return f"{video}|{ep_id}|{height}_{pitch_1}_{pitch_2}|{start_frame_id}|{ins}"


def enumerate_samples(annotations, height, pitch_1, pitch_2, sample_step: int, num_future_steps: int):
    """Mirror of the sample enumeration in ``NavPixelGoalDataset.__init__``."""
    goals, turns, stops = [], [], []
    for item in annotations["episodes"]:
        ep_id, instruction, video = item["id"], item["instructions"], item["video"]
        actions = item["actions"][1:] + [0]
        pixel_goals = item["pixel_goals"]
        n = len(actions)
        if n < 4:
            continue
        base = dict(video=video, ep_id=ep_id, height=height, pitch_1=pitch_1, pitch_2=pitch_2, instruction=instruction)
        for r in range(n // sample_step + 1):
            start = r * sample_step
            if start == n or start == n - 1:
                continue
            goal_len, goal = pixel_goals[start]
            if goal_len == -1:
                if actions[start] == 1:
                    continue
                turn = []
                for i in range(start, min(n, start + num_future_steps)):
                    if actions[i] == 1:
                        break
                    turn.append(actions[i])
                turns.append(dict(base, start=start, kind=TURN, action=turn[0]))
            elif goal_len >= 3:
                # poses (shared per episode, not copied) feed the trajectory targets of LayaNav
                poses = item.get(f"poses_{height}cm_{pitch_2}deg")
                goals.append(dict(base, start=start, kind=GOAL, goal=list(goal), goal_len=goal_len, poses=poses))
        stops.append(dict(base, start=n - 1, kind=STOP, action=0))
    return goals, turns, stops


def load_vln_samples(dataset_use: str, sample_step=4, num_future_steps=4, pixel_goal_only=False, stop_repeat=5):
    from .internvla_n1_lerobot_dataset import data_list, get_annotations_from_lerobot_data

    samples = []
    for data in data_list(dataset_use.split(",")):
        height, pitch_1, pitch_2 = data.get("height"), data.get("pitch_1"), data.get("pitch_2")
        ann = get_annotations_from_lerobot_data(data["data_path"], f"{height}cm_{pitch_2}deg")
        goals, turns, stops = enumerate_samples(ann, height, pitch_1, pitch_2, sample_step, num_future_steps)
        cur = goals if pixel_goal_only else goals + turns + stops * stop_repeat
        rate = data.get("sampling_rate", 1.0)
        if rate < 1.0:
            cur = random.sample(cur, int(len(cur) * rate))
        print(f"[laya_s2] {data['data_path']} {height}cm {pitch_1}/{pitch_2}deg: "
              f"goal={len(goals)} turn={len(turns)} stop={len(stops)} -> {len(cur)}")
        samples.extend(cur)
    return samples


def frame_path(s: Dict, frame_id: int, look_down: bool = False) -> str:
    pitch = s["pitch_2"] if look_down else s["pitch_1"]
    return os.path.join(
        s["video"], f"observation.images.rgb.{s['height']}cm_{pitch}deg", f"episode_{s['ep_id']:06d}_{frame_id}.jpg"
    )


def trajectory_frame_ids(goal_len: int, max_len: int = 12) -> np.ndarray:
    """Frames along the way to the pixel goal used as System 1 starting points (as in NavPixelGoalDataset)."""
    ids = np.arange(0, goal_len, 2)
    if len(ids) > max_len:
        ids = np.arange(0, goal_len, int(np.ceil(goal_len / max_len)))
    return ids


def trajectory_target(s: Dict, cid: int, steps: int) -> np.ndarray:
    """Remaining path from frame start + cid to the goal, in the DualVLN System 1 format [steps, 3]."""
    from .internvla_n1_lerobot_dataset import (
        clip_or_pad,
        get_trajectory_relative_to_frame,
        interpolate_and_resample_trajectory,
    )

    pose = np.asarray(s["poses"][s["start"] : s["start"] + s["goal_len"] + 1], dtype=np.float64)
    pose = pose.reshape(len(pose), 4, 4)
    rel = get_trajectory_relative_to_frame(pose[cid:], camera_deg=s["pitch_2"])
    _, deltas = interpolate_and_resample_trajectory(rel, steps)
    return clip_or_pad(deltas, steps).astype(np.float32)


def make_image_transform(cfg, augment: bool = False):
    s = cfg.image_size
    ops = [v2.ToImage(), v2.Resize((s, s), antialias=True)]
    if augment:
        ops.append(v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    ops += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(cfg.image_mean, cfg.image_std)]
    return v2.Compose(ops)


class TeacherLatentStore:
    """keys.txt + latents.npy ([N, n_query, latent_dim] fp16), memory-mapped."""

    def __init__(self, root: str):
        with open(os.path.join(root, "keys.txt")) as f:
            self.index = {k.rstrip("\n"): i for i, k in enumerate(f)}
        self.root = root
        self._arr = None

    def get(self, key: str) -> Optional[np.ndarray]:
        i = self.index.get(key)
        if i is None:
            return None
        if self._arr is None:  # open lazily so each dataloader worker gets its own mmap
            self._arr = np.load(os.path.join(self.root, "latents.npy"), mmap_mode="r")
        return np.asarray(self._arr[i], dtype=np.float32)


class LayaS2Dataset(Dataset):
    def __init__(
        self,
        samples: List[Dict],
        tokenizer,
        cfg,  # LayaS2Config
        teacher_latents: Optional[str] = None,
        augment: bool = False,
        goal_xy_order: str = "xy",
        with_traj: bool = False,  # LayaNav: add a (current look-down frame, remaining path) pair to goal samples
        traj_steps: int = 32,
    ):
        self.actions = list(cfg.actions)
        self.samples = [s for s in samples if s["kind"] == GOAL or s["action"] in self.actions]
        if len(self.samples) != len(samples):
            print(f"[laya_s2] dropped {len(samples) - len(self.samples)} samples with actions outside {self.actions}")
        self.tok = tokenizer
        self.cfg = cfg
        self.store = TeacherLatentStore(teacher_latents) if teacher_latents else None
        self.goal_xy_order = goal_xy_order
        self.transform = make_image_transform(cfg, augment)
        self.augment = augment
        self.with_traj = with_traj
        self.traj_steps = traj_steps

    def __len__(self):
        return len(self.samples)

    def _img(self, path):
        return self.transform(Image.open(path).convert("RGB"))

    def __getitem__(self, i):
        s = self.samples[i]
        start = s["start"]
        hist_ids = np.unique(np.linspace(0, start - 1, self.cfg.num_history, dtype=np.int32)).tolist() if start else []

        S = self.cfg.image_size
        hist = torch.zeros(self.cfg.num_history, 3, S, S)
        hist_mask = torch.zeros(self.cfg.num_history, dtype=torch.bool)
        for j, fid in enumerate(hist_ids):
            hist[j] = self._img(frame_path(s, fid))
            hist_mask[j] = True

        down_raw = Image.open(frame_path(s, start, look_down=True)).convert("RGB")
        item = dict(
            input_ids=torch.tensor(
                self.tok(s["instruction"], truncation=True, max_length=self.cfg.max_text_len)["input_ids"]
            ),
            hist_pixels=hist,
            hist_mask=hist_mask,
            cur_pixels=self._img(frame_path(s, start)),
            down_pixels=self.transform(down_raw),
            is_goal=s["kind"] == GOAL,
            action_idx=-1,
            goal_xy=torch.zeros(2),
            latent=torch.zeros(self.cfg.n_query, self.cfg.latent_dim),
            latent_mask=False,
        )
        if s["kind"] == GOAL:
            a, b = s["goal"]
            x, y = (a, b) if self.goal_xy_order == "xy" else (b, a)
            W, H = down_raw.size
            item["goal_xy"] = torch.tensor([x / W, y / H], dtype=torch.float32).clamp(0, 1)
            if self.store is not None:
                lat = self.store.get(sample_key(*(s[k] for k in KEY_FIELDS)))
                if lat is not None:
                    item["latent"] = torch.from_numpy(lat)
                    item["latent_mask"] = True
        else:
            item["action_idx"] = self.actions.index(s["action"])
        if self.with_traj:
            item.update(traj_pixels=torch.zeros(3, S, S), traj=torch.zeros(self.traj_steps, 3), traj_mask=False)
            ids = self._traj_start_ids(s)
            if len(ids):
                cid = int(np.random.choice(ids)) if self.augment else int(ids[len(ids) // 2])
                item["traj_pixels"] = self._img(frame_path(s, start + cid, look_down=True))
                item["traj"] = torch.from_numpy(trajectory_target(s, cid, self.traj_steps))
                item["traj_mask"] = True
        return item

    @staticmethod
    def _traj_start_ids(s: Dict) -> np.ndarray:
        if s["kind"] != GOAL or s.get("poses") is None:
            return np.zeros(0, dtype=int)
        ids = trajectory_frame_ids(s["goal_len"])
        return ids[s["start"] + ids < len(s["poses"])]  # starting frames that exist in the episode


KEY_FIELDS = ("video", "ep_id", "height", "pitch_1", "pitch_2", "instruction", "start")


def collate_laya_s2(batch: Sequence[Dict], pad_id: int) -> Dict[str, torch.Tensor]:
    ids = [b["input_ids"] for b in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=pad_id)
    text_mask = torch.arange(input_ids.shape[1])[None] < torch.tensor([len(x) for x in ids])[:, None]
    out = dict(input_ids=input_ids, text_mask=text_mask)
    for k in ("hist_pixels", "hist_mask", "cur_pixels", "down_pixels", "goal_xy", "latent"):
        out[k] = torch.stack([b[k] for b in batch])
    for k in ("is_goal", "latent_mask", "action_idx"):
        out[k] = torch.tensor([b[k] for b in batch])
    if "traj" in batch[0]:
        out["traj_pixels"] = torch.stack([b["traj_pixels"] for b in batch])
        out["traj"] = torch.stack([b["traj"] for b in batch])
        out["traj_mask"] = torch.tensor([b["traj_mask"] for b in batch])
    return out
