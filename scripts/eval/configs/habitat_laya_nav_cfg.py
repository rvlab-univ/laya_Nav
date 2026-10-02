import os

from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

# LayaNav (single model: Laya-S2 decision + trajectory head instead of the DualVLN System 1). Everything
# else matches habitat_dual_system_cfg.py, so the progress.json files can be compared directly
# (scripts/eval/compare_progress.py).
eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "laya_nav",
            "model_path": os.environ.get("LAYA_NAV", "checkpoints/laya_nav_c2/last"),  # train_laya_nav.py output
            "num_history": 8,
            "resize_w": 384,
            "resize_h": 384,
            "vis_debug": False,
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            'config_path': 'scripts/eval/configs/vln_r2r.yaml',
            # EVAL_EPISODES=N: fixed random subset of N episodes (same for every model); unset = all
            'episode_subset': int(os.environ.get('EVAL_EPISODES', 0)) or None,
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        "output_path": "./logs/habitat/test_laya_nav",
        "save_video": False,
        "epoch": 0,
        "max_steps_per_episode": 500,
        "port": "2333",
        "dist_url": "env://",
    },
)
