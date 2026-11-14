"""Isaac Lab's pretrained Go2 walking policy under fixed velocity commands: does it track them without falling?

This is the low level that stands in for Go2's built-in controller in simulation (the real Go2 takes the same
velocity commands through Unitree SDK2). Headless; the verdict comes from numbers, not from watching.

    unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL; source /mnt/storage/light_nav/env_isaaclab/bin/activate
    cd /mnt/storage/IsaacLab && ./isaaclab.sh -p /mnt/storage/light_nav/scripts/isaac/go2_walk_check.py --headless \
        --checkpoint /mnt/storage/light_nav/assets/isaac/go2/go2_flat.pt
"""
import argparse
import json
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Isaac-Velocity-Flat-Unitree-Go2-Play-v0")
parser.add_argument("--checkpoint", required=True, help="rsl_rl checkpoint (Isaac Lab's published go2 flat / rough)")
parser.add_argument("--out", default=None, help="JSON with the per-segment numbers")
parser.add_argument("--plan", choices=("basic", "turns"), default="basic", help="which command sequence")
parser.add_argument("--flat_ground", action="store_true", help="plane terrain (for a rough-terrain task)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401  (registers the tasks)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

# (name, vx m/s, vy m/s, wz rad/s, seconds)
QUARTER = math.pi / 2 / 0.5
SEGMENTS = [
    ("stand", 0.0, 0.0, 0.0, 2.0),
    ("forward 0.5 m/s", 0.5, 0.0, 0.0, 4.0),
    ("turn left 90 deg", 0.0, 0.0, 0.5, QUARTER),
    ("forward 0.3 m/s", 0.3, 0.0, 0.0, 3.0),
    ("turn right 90 deg", 0.0, 0.0, -0.5, QUARTER),
    ("stop", 0.0, 0.0, 0.0, 2.0),
]
TURNS = [
    ("stand", 0.0, 0.0, 0.0, 2.0),
    ("turn in place 0.5 rad/s", 0.0, 0.0, 0.5, 4.0),
    ("stand", 0.0, 0.0, 0.0, 1.0),
    ("turn in place 1.0 rad/s", 0.0, 0.0, 1.0, 3.0),
    ("stand", 0.0, 0.0, 0.0, 1.0),
    ("turn in place -1.0 rad/s", 0.0, 0.0, -1.0, 3.0),
    ("stand", 0.0, 0.0, 0.0, 1.0),
    ("arc 0.2 m/s + 0.5 rad/s", 0.2, 0.0, 0.5, 3.0),
    ("arc 0.1 m/s + 1.0 rad/s", 0.1, 0.0, 1.0, 3.0),
    ("stop", 0.0, 0.0, 0.0, 2.0),
]


def yaw_of(q):
    w, x, y, z = q.tolist()
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=1)
    cmd = env_cfg.commands.base_velocity
    cmd.heading_command = False          # the command is set directly, every step
    cmd.rel_standing_envs = 0.0
    cmd.resampling_time_range = (1.0e9, 1.0e9)
    cmd.debug_vis = False
    env_cfg.episode_length_s = 3600.0
    if args.flat_ground:
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
        if getattr(env_cfg.curriculum, "terrain_levels", None) is not None:
            env_cfg.curriculum.terrain_levels = None
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")

    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base = env.unwrapped
    robot = base.scene["robot"]
    term = base.command_manager.get_term("base_velocity")
    dt = base.step_dt
    print("control dt %.3f s (%.0f Hz), physics dt %.4f s" % (dt, 1 / dt, base.physics_dt), flush=True)

    obs = env.get_observations()
    results, fell_total = [], 0
    plan = SEGMENTS if args.plan == "basic" else TURNS
    for name, vx, vy, wz, seconds in plan:
        steps = int(round(seconds / dt))
        p0 = robot.data.root_pos_w[0].clone()
        y0 = yaw_of(robot.data.root_quat_w[0])
        heights, vxs, wzs, fell, yaw_acc, prev_yaw = [], [], [], 0, 0.0, y0
        for i in range(steps):
            term.vel_command_b[:] = torch.tensor([vx, vy, wz], device=base.device)
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
            if bool(dones[0]):
                fell += 1
            yaw = yaw_of(robot.data.root_quat_w[0])
            yaw_acc += wrap(yaw - prev_yaw)
            prev_yaw = yaw
            heights.append(float(robot.data.root_pos_w[0, 2]))
            if i >= steps // 2:                  # steady state: second half of the segment
                vxs.append(float(robot.data.root_lin_vel_b[0, 0]))
                wzs.append(float(robot.data.root_ang_vel_b[0, 2]))
        p1 = robot.data.root_pos_w[0]
        rec = {
            "segment": name, "cmd": [vx, vy, wz], "seconds": round(steps * dt, 2),
            "moved_m": round(float(torch.linalg.norm((p1 - p0)[:2])), 3),
            "turned_deg": round(math.degrees(yaw_acc), 1),
            "vx_measured": round(sum(vxs) / max(1, len(vxs)), 3),
            "wz_measured": round(sum(wzs) / max(1, len(wzs)), 3),
            "min_height_m": round(min(heights), 3), "resets": fell,
        }
        fell_total += fell
        results.append(rec)
        print(json.dumps(rec), flush=True)
    verdict = {"fell": fell_total > 0}
    for r in results:
        if r["cmd"][2]:
            verdict[r["segment"]] = {"turned_deg": r["turned_deg"],
                                     "expected_deg": round(math.degrees(r["cmd"][2] * r["seconds"]), 1),
                                     "wz_tracking": round(r["wz_measured"] / r["cmd"][2], 2)}
        elif r["cmd"][0]:
            verdict[r["segment"]] = {"vx_tracking": round(r["vx_measured"] / r["cmd"][0], 2)}
    print("VERDICT " + json.dumps(verdict), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"task": args.task, "checkpoint": args.checkpoint, "segments": results, "verdict": verdict}, f,
                      indent=2)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
