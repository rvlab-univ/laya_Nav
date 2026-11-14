"""Go2 in the KIMM room (Isaac Sim 5.1 + Isaac Lab 2.3), driven by the student with NaVILA's decision loop.

- Low level: Isaac Lab's published Go2 walking policy (rough-terrain checkpoint on a flat floor) under velocity
  commands; it stands in for Go2's built-in controller.
- Executor: lightnav.executor (25 cm / 15 degree steps closed on the robot's pose).
- High level: the student over HTTP (lightnav.server), the same request as scripts/eval_vlnce.py: one frame per
  low-level step, NaVILA's 8-frame choice, the answer's steps queued.
- Two cameras on the body, both 512 x 512 with a 90 degree field of view like NaVILA's VLN-CE camera:
  "high" at 1.25 m (NaVILA's training height) and "low" at Go2's head camera (about 0.36 m). One of them feeds the
  student (--camera); both are saved.

    # student server first (student env):  env/bin/python -m lightnav.server --ckpt runs/N2/last --port 8765
    unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL; source /mnt/storage/light_nav/env_isaaclab/bin/activate
    cd /mnt/storage/IsaacLab && ./isaaclab.sh -p /mnt/storage/light_nav/scripts/isaac/go2_kimm_drive.py --headless \
        --enable_cameras --instruction "Walk to the table and stop." --out /mnt/storage/light_nav/runs/isaac/drive/t1
    # --look: no student; turn 360 degrees in 15 degree steps and save the views (to choose instructions)
    # --furniture: the TV stand, chairs, sofas and lamp are fixed in place by default (Go2 knocks loose ones over,
    #              and some chairs tip over on their own); remove takes them out, physics leaves them loose
    # --video: drive.mp4 at 10 frames per simulated second: a chase camera behind the robot, the student's view,
    #          a top-down map of the path, and the instruction and current answer underneath
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.request

from isaaclab.app import AppLauncher

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KIMM = ("/mnt/storage/Isaac Sim/RVLab_isaacsim_aiworker/isaacsim_ai_worker_kimm/usd_env/env2_kimm_room/"
        "env2_kimm_room_ver1.usd")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", required=True)
parser.add_argument("--instruction", default=None)
parser.add_argument("--look", action="store_true", help="turn 360 degrees and save the views; no student")
parser.add_argument("--look_steps", type=int, default=24, help="15 degree turns for --look (0: the first view only)")
parser.add_argument("--url", default="http://127.0.0.1:8765", help="student server")
parser.add_argument("--camera", choices=("high", "low"), default="high", help="which camera the student sees")
parser.add_argument("--start", type=float, nargs=3, default=(-2.2, 0.66, 0.0), metavar=("X", "Y", "YAW_DEG"))
parser.add_argument("--max_steps", type=int, default=200, help="low-level steps (25 cm / 15 degrees) per episode")
parser.add_argument("--checkpoint", default=os.path.join(ROOT, "assets", "isaac", "go2", "go2_rough.pt"))
parser.add_argument("--lights", type=float, default=30000.0, help="intensity of the sphere lights under the ceiling")
parser.add_argument("--always", default=None, metavar="KEY",
                    help="no student: answer KEY every time (lightnav.questions.KEYS, e.g. \"forward 75\"), "
                         "to test the executor and the room")
parser.add_argument("--avoid", choices=("depth", "none"), default="depth",
                    help="depth: forward steps side-step round what the low camera's depth shows close ahead "
                         "(lightnav.executor), a stand-in for Go2's own avoidance; none: walk into it")
parser.add_argument("--furniture", choices=("fixed", "remove", "physics"), default="fixed",
                    help="the room's free bodies (TV stand, chairs, sofas, lamp): fixed in place, removed, or loose")
parser.add_argument("--video", action="store_true", help="record drive.mp4 (adds a chase camera)")
parser.add_argument("--video_fps", type=int, default=10, help="frames per simulated second")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

sys.path.insert(0, ROOT)
from lightnav.data import frame_slots  # noqa: E402
from lightnav.executor import FORWARD, LEFT, STOP, MotionConfig, Segment  # noqa: E402
from lightnav.questions import ANSWERS, KEYS  # noqa: E402
from lightnav.vlnce import decode_response, encode_request, navila_actions  # noqa: E402

TASK = "Isaac-Velocity-Rough-Unitree-Go2-Play-v0"
CAM_HIGH_Z, CAM_LOW = 1.25 - 0.31, (0.30, 0.0, 0.05)     # offsets from the body (stands at about 0.31 m)


# Free rigid bodies of the KIMM room (no joint holds them; drawers and doors are jointed and stay as they are): a
# TV on a stand in the middle of the room, 7 chairs, 2 sofas, a floor lamp and a small kitchen appliance. Go2 knocks
# them over, so by default they are fixed in place (kinematic), like heavy furniture in a real room.
FREE_BODIES = ["OBJ_app_0009", "OBJ_cha_0001", "OBJ_cha_0002", "OBJ_cha_0003", "OBJ_cha_0004", "OBJ_cha_0005",
               "OBJ_cha_0006", "OBJ_cha_0007", "OBJ_sof_0001", "OBJ_sof_0002", "OBJ_lit_0001", "OBJ_app_0006"]


def room_layer(path, furniture="fixed"):
    """A layer whose default prim is the KIMM room only (the stage's default prim is the AI Worker robot). The whole
    /Environment is referenced so the room keeps the original stage's coordinates. `furniture`: "fixed" makes the
    free bodies kinematic, "remove" deactivates them, "physics" leaves them as authored."""
    body = ""
    if furniture != "physics":
        one = ('            over "%s"\n            {\n                bool physics:kinematicEnabled = 1\n            }\n'
               if furniture == "fixed" else '            over "%s" (\n                active = false\n            )\n'
                                            '            {\n            }\n')
        body = ('    over "LAB_3_scene"\n    {\n        over "LAB_3_scene"\n        {\n%s        }\n    }\n'
                % "".join(one % name for name in FREE_BODIES))
    with open(path, "w") as f:
        f.write('#usda 1.0\n(\n    defaultPrim = "kimm"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n\n'
                'def Xform "kimm" (\n    prepend references = @%s@</Environment>\n)\n{\n%s}\n' % (KIMM, body))
    return path


def camera(prim, pos, height=512, width=512, pitch_deg=0.0, aperture=20.0, depth=False, rgb=True, near=0.05):
    """A camera on the body looking forward (+x), tilted down by `pitch_deg`; aperture 20 with focal length 10 is a
    90 degree horizontal field of view."""
    h = math.radians(pitch_deg) / 2
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/" + prim, update_period=0.0, height=height, width=width,
        data_types=(["rgb"] if rgb else []) + (["distance_to_image_plane"] if depth else []),
        spawn=sim_utils.PinholeCameraCfg(focal_length=10.0, horizontal_aperture=aperture, clipping_range=(near, 50.0)),
        offset=CameraCfg.OffsetCfg(pos=pos, rot=(math.cos(h), 0.0, math.sin(h), 0.0), convention="world"))


ROOM = (-4.64, 4.11, -5.54, 4.89)
# Obstacle check on the low camera's depth (--avoid depth): points 6 to 60 cm above the floor (Go2 walks under
# anything higher; lower is floor and rugs), in a corridor as wide as Go2 plus a margin.
OBSTACLE_Z = (0.06, 0.60)
HALF_WIDTH = 0.23
DEPTH_CAM, DEPTH_PITCH, DEPTH_FOV = (0.25, 0.0, 0.10), 30.0, 120.0   # offset from the body (m), tilt down, view (deg)
OWN_BODY_M = 0.12            # depth points nearer than this ahead of the sensor are Go2's own head
FALLEN_Z = 0.15          # base height (m) below which Go2 has fallen over; it stands at about 0.31 and the walking
                         # policy cannot get up again, so the episode ends there        # x min, x max, y min, y max of the KIMM room (m)


class Recorder:
    """drive.mp4: chase view | student view | map, with a text bar."""

    def __init__(self, path, fps, instruction):
        import imageio
        self.writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=16)
        self.instruction = instruction or "(look around)"
        self.text = ""
        self.note = ""          # what the executor is doing on its own (side-stepping)
        self.path = []
        self.frames = 0

    def map(self, pose, h=480):
        import cv2
        x0, x1, y0, y1 = ROOM
        s = (h - 40) / (y1 - y0)
        w = int((x1 - x0) * s) + 40
        img = np.full((h, w, 3), 245, np.uint8)
        to = lambda x, y: (int(20 + (x - x0) * s), int(h - 20 - (y - y0) * s))     # noqa: E731
        cv2.rectangle(img, to(x0, y1), to(x1, y0), (120, 120, 120), 2)
        pts = [to(x, y) for x, y, _ in self.path]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, (200, 90, 40), 2)
        if pts:
            cv2.circle(img, pts[0], 6, (60, 160, 60), -1)
        x, y, yaw = pose
        c = to(x, y)
        cv2.circle(img, c, 7, (40, 40, 200), -1)
        cv2.arrowedLine(img, c, (int(c[0] + 22 * math.cos(yaw)), int(c[1] - 22 * math.sin(yaw))), (40, 40, 200), 2,
                        tipLength=0.4)
        cv2.putText(img, "top view", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
        return img

    def add(self, chase, view, pose, t):
        import cv2
        self.path.append(pose)
        chase = cv2.resize(chase, (640, 480))
        view = cv2.resize(view, (480, 480))
        cv2.putText(chase, "chase camera", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(view, "student input (1.25 m)" if args.camera == "high" else "student input (head)", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        top = np.concatenate([chase, view, self.map(pose)], axis=1)
        bar = np.full((80, top.shape[1], 3), 30, np.uint8)
        cv2.putText(bar, "Instruction: " + self.instruction, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255),
                    1, cv2.LINE_AA)
        cv2.putText(bar, "t = %5.1f s   %s%s" % (t, self.text, "   " + self.note if self.note else ""), (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 120),
                    1, cv2.LINE_AA)
        frame = np.concatenate([top, bar], axis=0)
        H, W = frame.shape[:2]
        frame = np.pad(frame, ((0, -H % 16), (0, -W % 16), (0, 0)))
        self.writer.append_data(frame)
        self.frames += 1

    def close(self):
        self.writer.close()


def make_env(out):
    cfg = parse_env_cfg(TASK, device=args.device or "cuda:0", num_envs=1)
    # flat floor at z = 0 under the room (the policy's height scan reads it; the room floor is also at z = 0)
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.curriculum.terrain_levels = None
    room = room_layer(os.path.join(out, "kimm_room.usda"), args.furniture)
    cfg.scene.kimm = AssetBaseCfg(prim_path="/World/kimm", spawn=sim_utils.UsdFileCfg(usd_path=room))
    # The room's ceiling mesh does not render (the sky shows through); NaVILA only saw indoor scenes, so a plain
    # white slab (visual only) closes the room at the ceiling height, 2.75 m.
    cfg.scene.ceiling = AssetBaseCfg(
        prim_path="/World/ceiling",
        spawn=sim_utils.CuboidCfg(size=(9.0, 10.6, 0.1),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.93, 0.93, 0.93))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.265, -0.325, 2.80)))
    for i, (x, y) in enumerate([(-2.5, -2.5), (-2.5, 2.5), (2.0, -2.5), (2.0, 2.5), (0.0, 0.0)]):
        setattr(cfg.scene, "lamp%d" % i, AssetBaseCfg(
            prim_path="/World/lamp%d" % i, spawn=sim_utils.SphereLightCfg(intensity=args.lights, radius=0.15),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, 2.5))))
    cfg.scene.cam_high = camera("cam_high", (0.0, 0.0, CAM_HIGH_Z))
    cfg.scene.cam_low = camera("cam_low", CAM_LOW)
    if args.avoid == "depth":
        # obstacle sensor (stands in for Go2's LiDAR): depth only, above the head and tilted down so that the floor
        # right in front of the nose is in view; the 0.36 m RGB head camera loses what is closer than 5 cm
        cfg.scene.cam_depth = camera("cam_depth", DEPTH_CAM, height=128, width=128, pitch_deg=DEPTH_PITCH,
                                     aperture=2 * 10.0 * math.tan(math.radians(DEPTH_FOV / 2)), depth=True, rgb=False,
                                     near=0.01)
    if args.video:
        cfg.scene.cam_chase = camera("cam_chase", (-1.3, 0.0, 0.75), height=480, width=640, pitch_deg=22.0,
                                     aperture=24.0)
    # velocity commands come from the executor, every tick
    cmd = cfg.commands.base_velocity
    cmd.heading_command, cmd.rel_standing_envs, cmd.debug_vis = False, 0.0, False
    cmd.resampling_time_range = (1.0e9, 1.0e9)
    cfg.episode_length_s = 1.0e5
    cfg.terminations.base_contact = None          # bumping furniture should not reset the episode
    x, y, yaw = args.start
    cfg.events.reset_base.params = {"pose_range": {"x": (x, x), "y": (y, y), "yaw": (math.radians(yaw),) * 2},
                                    "velocity_range": {}}
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    return env, runner.get_inference_policy(device=env.unwrapped.device)


class Sim:
    def __init__(self, env, policy, out):
        self.env, self.policy, self.out = env, policy, out
        self.base = env.unwrapped
        self.robot = self.base.scene["robot"]
        self.term = self.base.command_manager.get_term("base_velocity")
        self.dt = self.base.step_dt
        self.obs = env.get_observations()
        self.t, self.n_frames, self.ticks = 0.0, 0, 0
        self.motion, self._grid, self.last = MotionConfig(), None, None
        self.fell_at = None         # simulated time at which the base dropped below FALLEN_Z (Go2 on its side)
        self.rec = Recorder(os.path.join(out, "drive.mp4"), args.video_fps, args.instruction) if args.video else None
        self.rec_every = max(1, int(round(1.0 / (args.video_fps * self.dt))))
        self.track = open(os.path.join(out, "trajectory.csv"), "w")
        self.track.write("t,x,y,z,yaw_deg,vx_cmd,wz_cmd\n")
        for c in ("high", "low"):
            os.makedirs(os.path.join(out, "frames_" + c), exist_ok=True)

    def pose(self):
        p, q = self.robot.data.root_pos_w[0].tolist(), self.robot.data.root_quat_w[0].tolist()
        w, x, y, z = q
        return p[0], p[1], math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def tick(self, cmd=(0.0, 0.0, 0.0)):
        self.term.vel_command_b[:] = torch.tensor(cmd, device=self.base.device)
        with torch.inference_mode():
            self.obs, _, _, _ = self.env.step(self.policy(self.obs))
        self.t += self.dt
        x, y, yaw = self.pose()
        z = float(self.robot.data.root_pos_w[0, 2])
        if z < FALLEN_Z and self.fell_at is None:
            self.fell_at = self.t
        self.track.write("%.2f,%.3f,%.3f,%.3f,%.1f,%.2f,%.2f\n" % (self.t, x, y, z, math.degrees(yaw), cmd[0], cmd[2]))
        self.ticks += 1
        if self.rec is not None and self.ticks % self.rec_every == 0:
            grab = lambda c: self.base.scene[c].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)  # noqa
            self.rec.add(grab("cam_chase"), grab("cam_" + args.camera), (x, y, yaw), self.t)

    def frames(self):
        """Both cameras now, saved; returns the student's camera as H x W x 3 uint8."""
        out = {}
        for c in ("high", "low"):
            rgb = self.base.scene["cam_" + c].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
            Image.fromarray(rgb).save(os.path.join(self.out, "frames_" + c, "%04d.jpg" % self.n_frames), quality=90)
            out[c] = rgb
        self.n_frames += 1
        return out[args.camera]

    def gap(self):
        """Free space ahead from the obstacle sensor's depth (lightnav.executor.Gap).

        Ahead: the nearest obstacle point in Go2's corridor (from the sensor). Side: toward the image column nearest
        the centre whose nearest obstacle is beyond the clear distance, i.e. the nearest opening in view; with none,
        toward the side where the obstacle at the image edge is farther; None (no way round) when both edges are
        close too."""
        cam = self.base.scene["cam_depth"]
        depth = cam.data.output["distance_to_image_plane"][0, ..., 0].float()
        if self._grid is None:
            H, W = depth.shape
            f = (W / 2) / math.tan(math.radians(DEPTH_FOV / 2))
            v, u = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device),
                                  indexing="ij")
            self._grid = (-(u - (W - 1) / 2) / f, -(v - (H - 1) / 2) / f)
        lateral, up = self._grid
        # each pixel in the world frame with the sensor's full pose (Go2 pitches a few degrees as it walks), then
        # ahead / left of the robot's heading and height above the floor
        w, qx, qy, qz = cam.data.quat_w_world[0].tolist()
        R = torch.tensor([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - w * qz), 2 * (qx * qz + w * qy)],
                          [2 * (qx * qy + w * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - w * qx)],
                          [2 * (qx * qz - w * qy), 2 * (qy * qz + w * qx), 1 - 2 * (qx * qx + qy * qy)]],
                         device=depth.device)
        world = (torch.stack([torch.ones_like(lateral), lateral, up], -1) * depth[..., None]) @ R.T
        _, _, yaw = self.pose()
        c, s_ = math.cos(yaw), math.sin(yaw)
        d = c * world[..., 0] + s_ * world[..., 1]                  # ahead of the sensor (m)
        y = -s_ * world[..., 0] + c * world[..., 1]                 # left of the sensor (m)
        z = float(cam.data.pos_w[0, 2]) + world[..., 2]             # above the floor (m)
        ok = torch.isfinite(depth) & (z > OBSTACLE_Z[0]) & (z < OBSTACLE_Z[1]) & (d > OWN_BODY_M) & (d < 3.0)
        m = ok & (y.abs() < HALF_WIDTH)
        ahead = float(d[m].min()) if bool(m.any()) else 3.0
        nearest = torch.where(ok, d, torch.full_like(d, 3.0)).min(dim=0).values      # per image column
        W = nearest.numel()
        open_ = (nearest > self.motion.clear_dist + 0.2).tolist()
        left = next((i for i, col in enumerate(reversed(range(W // 2))) if open_[col]), None)
        right = next((i for i, col in enumerate(range(W // 2, W)) if open_[col]), None)
        if left is not None or right is not None:
            go_left = right is None or (left is not None and left <= right)
        else:
            if max(float(nearest[0]), float(nearest[-1])) < self.motion.clear_dist:
                return ahead, None
            go_left = float(nearest[0]) >= float(nearest[-1])
        return ahead, (0.1 if go_left else -0.1)

    def run(self, action, k, on_boundary):
        """k low-level steps of `action`; `on_boundary()` at each step boundary. True if the robot was blocked."""
        seg = Segment(action, k, self.pose(), self.motion)
        self.last = seg
        while True:
            gap = self.gap() if args.avoid == "depth" and action == FORWARD else None
            cmd, new = seg.update(self.pose(), self.dt, gap)
            if self.rec is not None:
                self.rec.note = ("side-stepping %s round an obstacle" % ("left" if seg.avoiding > 0 else "right")
                                 if seg.avoiding else "")
            for _ in range(new):
                on_boundary()
            if cmd is None:
                break
            self.tick(cmd)
        if self.rec is not None:
            self.rec.note = ""
        for _ in range(5):          # a short stand before the next decision
            self.tick()
        return seg.blocked


def ask(instruction, frames, slots):
    req = urllib.request.Request(args.url + "/decide", data=encode_request(instruction, frames, slots, "kimm"),
                                 headers={"Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return decode_response(r.read())


def main():
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    env, policy = make_env(out)
    sim = Sim(env, policy, out)
    for _ in range(int(1.5 / sim.dt)):     # settle, and let the renderer warm up
        sim.tick()
    log = open(os.path.join(out, "decisions.jsonl"), "w")
    summary = {"instruction": args.instruction, "camera": args.camera, "start": list(args.start),
               "furniture": args.furniture, "avoid": args.avoid,
               "checkpoint": args.checkpoint}
    if args.look:
        sim.frames()
        if args.look_steps:
            sim.run(LEFT, args.look_steps, sim.frames)
        summary["frames"] = sim.n_frames
    else:
        if args.always:
            summary["student"] = {"always": args.always}
        else:
            with urllib.request.urlopen(args.url + "/health", timeout=60) as r:
                summary["student"] = json.loads(r.read().decode())
        past, current, steps, blocked, stopped, side_steps = [], sim.frames(), 0, 0, False, 0
        while steps < args.max_steps:
            frames = past + [current]
            slots = frame_slots(len(frames), 8)
            real = [(s, frames[f]) for s, f in enumerate(slots) if f >= 0]
            t0 = time.time()
            if args.always:
                i = KEYS.index(args.always)
                answer, probs, ms = ANSWERS[i], [float(j == i) for j in range(len(ANSWERS))], 0.0
            else:
                answer, probs, ms = ask(args.instruction, [f for _, f in real], [s for s, _ in real])
            actions = navila_actions(answer) or [STOP]
            if sim.rec is not None:
                sim.rec.text = "step %d   answer: %s (p = %.2f)   decision %.0f ms" % (
                    steps, answer.replace("The next action is ", "").rstrip("."), max(probs), ms)
            x, y, yaw = sim.pose()
            rec = {"step": steps, "frames": len(frames), "answer": KEYS[ANSWERS.index(answer)] if answer in ANSWERS
                   else answer, "p": [round(p, 3) for p in probs], "ms": round(ms, 1),
                   "round_trip_ms": round(1000 * (time.time() - t0), 1),
                   "pose": [round(x, 3), round(y, 3), round(math.degrees(yaw), 1)]}
            print(json.dumps(rec), flush=True)
            if actions[0] == STOP:
                stopped = True
                log.write(json.dumps(rec) + "\n")
                for _ in range(int(2.0 / sim.dt)):      # hold the last view for the video
                    sim.tick()
                break
            k = min(len(actions), args.max_steps - steps)

            def boundary():
                nonlocal current
                past.append(current)
                current = sim.frames()

            rec["blocked"] = sim.run(actions[0], k, boundary)
            blocked += rec["blocked"]
            if sim.last.avoids:
                rec["side_steps"], rec["side_step_s"] = sim.last.avoids, round(sim.last.avoid_time, 2)
                rec["stalls"] = sim.last.stalls
                side_steps += sim.last.avoids
            steps += k
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if sim.fell_at is not None:
                print("Go2 fell over at t = %.1f s; episode ended" % sim.fell_at, flush=True)
                break
        x, y, yaw = sim.pose()
        summary.update({"steps": steps, "stopped": stopped, "fell_at_s": sim.fell_at, "blocked_segments": blocked, "side_steps": side_steps,
                        "end_pose": [round(x, 3), round(y, 3), round(math.degrees(yaw), 1)],
                        "sim_seconds": round(sim.t, 1), "frames": sim.n_frames})
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY " + json.dumps(summary), flush=True)
    sim.track.close()
    log.close()
    if sim.rec is not None:
        sim.rec.close()
        print("video %s (%d frames)" % (os.path.join(out, "drive.mp4"), sim.rec.frames), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
