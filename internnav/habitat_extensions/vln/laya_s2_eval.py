"""Habitat VLN evaluation with Laya-S2 as System 2 and the DualVLN System 1 (mode='laya_s2').

The loop mirrors ``HabitatVLNEvaluator._run_eval_dual_system`` step for step (same look-down capture,
System 1 re-planning every MAX_LOCAL_STEPS, System 2 re-query after MAX_STEPS or a local STOP, same
metrics and progress.json format), so the only difference to the baseline is the System 2 model.
"""

import json
import os

import numpy as np
import torch
import tqdm
from depth_camera_filtering import filter_depth
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
from PIL import Image

from internnav.habitat_extensions.vln.habitat_vln_evaluator import (
    MAX_LOCAL_STEPS,
    MAX_STEPS,
    action_code,
)
from internnav.habitat_extensions.vln.timing import EpisodeTimers
from internnav.habitat_extensions.vln.utils import preprocess_depth_image_v2
from internnav.model.basemodel.laya_s2.agent import LayaS2Agent
from internnav.model.utils.vln_utils import traj_to_actions


def _look_down_depth(ev, depth):
    depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
    depth = (depth * (ev._max_depth - ev._min_depth) + ev._min_depth) * 1000
    d, _ = preprocess_depth_image_v2(
        Image.fromarray(depth.astype(np.uint16), mode='I;16'),
        do_depth_scale=True,
        depth_scale=1000,
        target_height=224,
        target_width=224,
    )
    d = torch.as_tensor(np.ascontiguousarray(d)).float()
    d[d > 5.0] = 5.0
    return d


def _plan_s1(agent, timers, latent, pix_goal_image, pix_goal_depth, look_down_image, look_down_depth):
    image_dp = torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255
    images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(agent.device)
    depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)
    depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(agent.device)
    with torch.no_grad(), timers.s1():
        dp_actions = agent.s1.generate_traj(latent, images_dp, depths_dp, latents_projected=True)
    action_list = traj_to_actions(dp_actions)
    if len(action_list) < MAX_STEPS:
        action_list += [0] * (MAX_STEPS - len(action_list))
    return action_list[:MAX_LOCAL_STEPS]


def run_eval_laya_s2(ev):  # noqa: C901
    agent: LayaS2Agent = ev.model
    agent.eval()
    sucs, spls, oss, nes, ndtw = ev.resume_from_output_path()
    process_bar = tqdm.tqdm(total=len(ev.env.episodes), desc=f"Eval Epoch {ev.epoch} Rank {ev.rank}")

    while ev.env.is_running:
        observations = ev.env.reset()
        if not ev.env.is_running or observations is None:
            break
        episode = ev.env.get_current_episode()
        scene_id = episode.scene_id.split('/')[-2]
        episode_id = int(episode.episode_id)
        instruction = episode.instruction.instruction_text
        if ev.save_video:
            os.makedirs(os.path.join(ev.output_path, f'vis_{ev.epoch}', f'{scene_id}'), exist_ok=True)

        timers = EpisodeTimers()
        vis_frames, rgb_list, action_seq, local_actions, esc_probs = [], [], [], [], []
        step_id, forward_action = 0, 0
        done = False
        pixel_goal = latent = None

        while (not done) and (step_id <= ev.max_steps_per_episode):
            image = Image.fromarray(observations["rgb"]).convert('RGB')
            rgb_list.append(image.resize((ev.model_args.resize_w, ev.model_args.resize_h)))

            # look-down view, captured exactly like the dual-system baseline
            ev.env.step(action_code.LOOKDOWN)
            down_obs, _, _, _ = ev.env.step(action_code.LOOKDOWN)
            look_down_image = Image.fromarray(down_obs["rgb"]).convert('RGB')
            look_down_depth = _look_down_depth(ev, down_obs["depth"])
            ev.env.step(action_code.LOOKUP)
            ev.env.step(action_code.LOOKUP)

            if len(action_seq) == 0 and pixel_goal is None:
                history_id = (
                    sorted(np.unique(np.linspace(0, step_id - 1, ev.num_history, dtype=np.int32)).tolist())
                    if step_id
                    else []
                )
                with timers.s2():
                    d = agent.decide(instruction, [rgb_list[i] for i in history_id], rgb_list[-1], look_down_image)
                esc_probs.append(d["escalate_prob"])
                if d["kind"] == "goal":
                    W, H = look_down_image.size
                    pixel_goal = [int(d["goal_xy"][0] * W), int(d["goal_xy"][1] * H)]  # (x, y), for vis only
                    latent = d["latent"][None].to(torch.bfloat16)
                    forward_action = 0
                    pix_goal_image = torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255
                    pix_goal_depth = look_down_depth.unsqueeze(-1).to(torch.bfloat16)
                    local_actions = _plan_s1(
                        agent, timers, latent, pix_goal_image, pix_goal_depth, look_down_image, look_down_depth
                    )
                    if local_actions[0] == action_code.STOP:
                        # same fallback as the baseline when System 1 stops right away
                        pixel_goal = None
                        observations, _, done, _ = ev.env.step(action_code.LEFT)
                        step_id += 1
                        continue
                else:
                    action_seq = [d["action"]]

            if len(action_seq) != 0:
                action = action_seq.pop(0)
            elif pixel_goal is not None:
                if len(local_actions) == 0:
                    local_actions = _plan_s1(
                        agent, timers, latent, pix_goal_image, pix_goal_depth, look_down_image, look_down_depth
                    )
                action = local_actions.pop(0)
                forward_action += 1
                if forward_action > MAX_STEPS or action == action_code.STOP:
                    pixel_goal = None
                    step_id += 1
                    forward_action = 0
                    local_actions = []
                    continue
            else:
                action = 0

            info = ev.env.get_metrics()
            if info['top_down_map'] is not None and ev.save_video:
                frame = observations_to_image({'rgb': np.asarray(image)}, info)
                vis_frames.append(frame)

            observations, _, done, _ = ev.env.step(action)
            step_id += 1

        process_bar.update(1)
        metrics = ev.env.get_metrics()
        sucs.append(metrics['success'])
        spls.append(metrics['spl'])
        oss.append(metrics['oracle_success'])
        nes.append(metrics["distance_to_goal"])
        if 'ndtw' in metrics:
            ndtw.append(metrics["ndtw"])
        print(
            f"scene_episode {scene_id}_{episode_id:04d} success: {metrics['success']}, "
            f"spl: {metrics['spl']}, os: {metrics['oracle_success']}, ne: {metrics['distance_to_goal']}"
        )
        result = {
            "scene_id": scene_id,
            "episode_id": episode_id,
            "success": metrics["success"],
            "spl": metrics["spl"],
            "os": metrics['oracle_success'],
            "ne": metrics["distance_to_goal"],
            "steps": step_id,
            "episode_instruction": instruction,
            **timers.summary(),
            "escalate_prob_mean": float(np.mean(esc_probs)) if esc_probs else 0.0,
        }
        if 'ndtw' in metrics:
            result['ndtw'] = metrics['ndtw']
        os.makedirs(ev.output_path, exist_ok=True)
        with open(os.path.join(ev.output_path, 'progress.json'), 'a') as f:
            f.write(json.dumps(result) + "\n")
        if ev.save_video and metrics['success'] == 1.0:
            images_to_video(
                vis_frames,
                os.path.join(ev.output_path, f'vis_{ev.epoch}', f'{scene_id}'),
                f'{episode_id:04d}',
                fps=6,
                quality=9,
            )
        vis_frames.clear()

    ev.env.close()
    t = lambda x: torch.tensor(x).to(ev.device)  # noqa: E731
    return t(sucs), t(spls), t(oss), t(nes), (t(ndtw) if ndtw else None)
