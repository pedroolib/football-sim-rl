"""PPO training entry point (M5).

Run from the repository root:

    python -m qb_sim.training.train --timesteps 200000 --seed 0 --run-name ppo_default

Training is headless and independent from the UI; importing ``qb_sim`` never
loads PyTorch or Stable-Baselines3.
"""

import argparse
import json
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from qb_sim.rules.scenarios import SCENARIOS, get_scenario
from qb_sim.training.env import QBPassingEnv
from qb_sim.training.ppo_config import PPOConfig


def resolve_device(device: str) -> str:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_env(scenario_name: str) -> QBPassingEnv:
    """Create a headless environment for the named scenario."""
    return QBPassingEnv(get_scenario(scenario_name))


def build_model(config: PPOConfig, env) -> PPO:
    """Create a PPO model with the experiment hyperparameters."""
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs={
            "net_arch": {"pi": list(config.pi_layers), "vf": list(config.vf_layers)},
            "log_std_init": config.log_std_init,
        },
        seed=config.seed,
        device=resolve_device(config.device),
        verbose=1,
    )


def train(config: PPOConfig, resume: str | None = None) -> tuple[PPO, Path]:
    """Train or resume PPO and persist model, config, checkpoints, and logs."""
    run_dir = Path(config.output_dir) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    env = DummyVecEnv([lambda: Monitor(build_env(config.scenario))])
    eval_env = DummyVecEnv([lambda: Monitor(build_env(config.scenario))])

    if resume is not None:
        model = PPO.load(resume, env=env, device=resolve_device(config.device))
        model.set_random_seed(config.seed)
    else:
        model = build_model(config, env)
    model.set_logger(configure(str(run_dir / "logs"), ["stdout", "csv"]))

    callbacks = [
        CheckpointCallback(
            save_freq=max(config.checkpoint_freq // config.n_envs, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix=config.run_name,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / "best"),
            log_path=str(run_dir / "logs"),
            eval_freq=max(config.eval_freq // config.n_envs, 1),
            n_eval_episodes=config.eval_episodes,
            deterministic=True,
        ),
    ]
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        reset_num_timesteps=resume is None,
        progress_bar=False,
    )

    final_path = run_dir / "final_model.zip"
    model.save(final_path)
    metadata = config.to_dict()
    metadata["device"] = resolve_device(config.device)
    metadata["torch"] = torch.__version__
    metadata["resumed_from"] = resume
    (run_dir / "config.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    env.close()
    eval_env.close()
    return model, final_path


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    defaults = PPOConfig()
    parser = argparse.ArgumentParser(description="Train PPO on QBPassing-v0")
    parser.add_argument("--timesteps", type=int, default=defaults.total_timesteps)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--scenario", default=defaults.scenario, choices=sorted(SCENARIOS))
    parser.add_argument("--run-name", default=defaults.run_name)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--checkpoint-freq", type=int, default=defaults.checkpoint_freq)
    parser.add_argument("--eval-freq", type=int, default=defaults.eval_freq)
    parser.add_argument("--eval-episodes", type=int, default=defaults.eval_episodes)
    parser.add_argument("--n-envs", type=int, default=defaults.n_envs)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args(argv)

    config = PPOConfig(
        run_name=args.run_name,
        scenario=args.scenario,
        total_timesteps=args.timesteps,
        seed=args.seed,
        n_envs=args.n_envs,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        output_dir=args.output_dir,
        device=args.device,
    )
    _, final_path = train(config, resume=args.resume)
    print(f"saved {final_path}")


if __name__ == "__main__":
    main()
