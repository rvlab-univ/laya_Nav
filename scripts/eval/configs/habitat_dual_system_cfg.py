import os

from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "dual_system",  # inference mode: dual_system or system2
            "model_path": "checkpoints/InternVLA-N1-DualVLN",  # path to model checkpoint
            "num_history": 8,
            "resize_w": 384,  # image resize width
            "resize_h": 384,  # image resize height
            "max_new_tokens": 1024,  # maximum number of tokens for generation
            "vis_debug": False,  # If vis_debug=True, save debug videos per episode
            "vis_debug_path": "./logs/habitat/vis_debug",
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            # habitat sim specifications - agent, sensors, tasks, measures etc. are defined in the habitat config file
            'config_path': 'scripts/eval/configs/vln_r2r.yaml',
            # EVAL_EPISODES=N: fixed random subset of N episodes (same for every model); unset = all
            'episode_subset': int(os.environ.get('EVAL_EPISODES', 0)) or None,
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        # all current parse args
        "output_path": "./logs/habitat/test_dual_system",  # output directory for logs/results
        "save_video": False,  # whether to save videos
        "epoch": 0,  # epoch number for logging
        "max_steps_per_episode": 500,  # maximum steps per episode
        # distributed settings
        "port": "2333",  # communication port
        "dist_url": "env://",  # url for distributed setup
    },
)
