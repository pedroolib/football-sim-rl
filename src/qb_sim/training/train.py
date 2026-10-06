"""PPO training entry point (M5).

Run from the repository root:

    python -m qb_sim.training.train --timesteps 200000 --seed 0 --run-name ppo_default

Training is headless and independent from the UI; importing ``qb_sim`` never
loads PyTorch or Stable-Baselines3.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from qb_sim.agents.rule_controllers import ScramblingQuarterback
from qb_sim.rules.drive import DriveConfig
from qb_sim.rules.routes import ROUTE_OBSERVATION_ORDER
from qb_sim.rules.scenarios import SCENARIOS, get_scenario
from qb_sim.training.drive_env import QBDriveEnv
from qb_sim.training.drive_observations import DRIVE_OBSERVATION_SIZE
from qb_sim.training.env import QBPassingEnv
from qb_sim.training.imitation import (
    collect_demonstrations,
    collect_drive_demonstrations,
    collect_receiver_drive_demonstrations,
    collect_role_drive_demonstrations,
    collect_role_play_demonstrations,
    concatenate_demonstrations,
    warm_start_policy,
)
from qb_sim.training.multiscenario import MultiScenarioEnv
from qb_sim.training.observations import OBSERVATION_SIZE
from qb_sim.training.ppo_config import PPOConfig
from qb_sim.training.randomization import AdaptationRanges
from qb_sim.training.receiver_drive_env import ReceiverDriveEnv
from qb_sim.training.role_context_observations import (
    DEFENDER_CONTEXT_OBSERVATION_SIZE,
    RECEIVER_CONTEXT_OBSERVATION_SIZE,
    RECEIVER_DRIVE_OBSERVATION_SIZE,
)
from qb_sim.training.role_env import RoleDriveEnv, RolePlayEnv

CURRICULUM_DECAY_START_FRACTION = 0.4
CURRICULUM_FULL_FRACTION = 0.8


class AttemptAnnealingCallback(BaseCallback):
    """Linearly decay the attempt shaping to zero over ``decay_steps``."""

    def __init__(self, decay_steps: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._decay_steps = max(int(decay_steps), 1)
        self._supports_attempt_scale: bool | None = None
        self._applied_scale: float | None = None

    def _on_step(self) -> bool:
        scale = max(0.0, 1.0 - self.num_timesteps / self._decay_steps)
        if scale == self._applied_scale:
            return True
        env = self.model.get_env()
        if self._supports_attempt_scale is None:
            self._supports_attempt_scale = env.has_attr("set_attempt_scale")
        if self._supports_attempt_scale:
            env.env_method("set_attempt_scale", scale)
        self._applied_scale = scale
        return True


class AdaptationCurriculumCallback(BaseCallback):
    """Widen the sampled capability ranges as training progresses.

    The phase stays at 0 (narrow) until ``decay_start_fraction`` of the run,
    grows linearly to 1.0 at ``full_fraction``, and stays at 1.0 afterwards.
    """

    def __init__(
        self,
        decay_start_fraction: float = CURRICULUM_DECAY_START_FRACTION,
        full_fraction: float = CURRICULUM_FULL_FRACTION,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        if not 0.0 <= decay_start_fraction < full_fraction <= 1.0:
            raise ValueError("curriculum fractions must satisfy 0 <= start < full <= 1")
        self._decay_start_fraction = float(decay_start_fraction)
        self._full_fraction = float(full_fraction)
        self._total_timesteps = 1
        self._applied_progress: float | None = None
        self._supports_curriculum: bool | None = None

    def _on_training_start(self) -> None:
        self._total_timesteps = max(int(self.model._total_timesteps), 1)

    def _progress(self) -> float:
        fraction = self.num_timesteps / self._total_timesteps
        if fraction <= self._decay_start_fraction:
            return 0.0
        if fraction >= self._full_fraction:
            return 1.0
        span = self._full_fraction - self._decay_start_fraction
        return (fraction - self._decay_start_fraction) / span

    def _on_step(self) -> bool:
        progress = self._progress()
        if progress == self._applied_progress:
            return True
        env = self.model.get_env()
        if self._supports_curriculum is None:
            self._supports_curriculum = env.has_attr("set_curriculum_progress")
        if self._supports_curriculum:
            env.env_method("set_curriculum_progress", progress)
        self._applied_progress = progress
        return True


def resolve_device(device: str) -> str:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def effective_scenarios(config: PPOConfig) -> tuple[str, ...]:
    """Scenarios to train on: the explicit list or the single scenario."""
    return config.scenarios if config.scenarios else (config.scenario,)


def build_opponents(env_kind: str, scenario_name: str, pairs) -> dict:
    """Load frozen opponent policies, mixing ``|``-separated paths per play."""
    from qb_sim.agents.rule_controllers import HeuristicQuarterback
    from qb_sim.training.policy import (
        MixedPolicy,
        load_defender_policy,
        load_drive_policy,
        load_policy,
        load_receiver_policy,
    )

    opponents: dict = {}
    for player_id, path in pairs:
        parts = [part.strip() for part in str(path).split("|")]
        if any(not part for part in parts):
            raise ValueError(f"ruta de oponente vacía para {player_id!r}: {path!r}")
        built: list = []
        for part in parts:
            if part == "rules":
                built.append(
                    HeuristicQuarterback(player_id, get_scenario(scenario_name).config)
                )
            elif player_id == "qb":
                loader = load_policy if env_kind == "passing" else load_drive_policy
                built.append(loader(part, scenario_name))
            elif player_id in ("wr1", "wr2"):
                built.append(load_receiver_policy(part, scenario_name, player_id))
            else:
                built.append(load_defender_policy(part, scenario_name, player_id))
        opponents[player_id] = built[0] if len(built) == 1 else MixedPolicy(tuple(built))
    return opponents


def build_env(
    scenario_names: tuple[str, ...],
    env_kind: str = "passing",
    learner: str = "qb",
    opponents: tuple[tuple[str, str], ...] = (),
    randomization: bool | None = None,
):
    """Create a headless environment for the requested training mode."""
    if env_kind not in ("passing", "drive", "role_play", "role_drive", "receiver_drive"):
        raise ValueError(f"unknown env_kind: {env_kind!r}")
    if (
        env_kind in ("drive", "role_play", "role_drive", "receiver_drive")
        and len(scenario_names) != 1
    ):
        raise ValueError(f"{env_kind} training supports exactly one scenario")
    ranges = AdaptationRanges() if randomization else None
    if env_kind == "drive":
        scenario = get_scenario(scenario_names[0])
        return QBDriveEnv(
            scenario,
            drive_config=DriveConfig(),
            opponents=build_opponents(env_kind, scenario_names[0], opponents),
            randomization=ranges,
        )
    if env_kind == "role_play":
        scenario = get_scenario(scenario_names[0])
        return RolePlayEnv(
            scenario,
            learner_id=learner,
            opponents=build_opponents(env_kind, scenario_names[0], opponents),
            randomization=ranges,
        )
    if env_kind == "role_drive":
        scenario = get_scenario(scenario_names[0])
        return RoleDriveEnv(
            scenario,
            learner_id=learner,
            opponents=build_opponents(env_kind, scenario_names[0], opponents),
            randomization=ranges,
        )
    if env_kind == "receiver_drive":
        scenario = get_scenario(scenario_names[0])
        return ReceiverDriveEnv(
            scenario,
            learner_id=learner,
            opponents=build_opponents(env_kind, scenario_names[0], opponents),
            randomization=ranges,
        )
    if len(scenario_names) == 1:
        return QBPassingEnv(get_scenario(scenario_names[0]))
    return MultiScenarioEnv(scenario_names)


class _EnvSpec:
    """Picklable env factory so SubprocVecEnv workers can rebuild the env."""

    def __init__(self, scenarios, env_kind, learner, opponents, randomization=False):
        self.scenarios = tuple(scenarios)
        self.env_kind = env_kind
        self.learner = learner
        self.opponents = tuple(opponents)
        self.randomization = bool(randomization)

    def __call__(self):
        return Monitor(
            build_env(
                self.scenarios,
                self.env_kind,
                self.learner,
                self.opponents,
                self.randomization,
            )
        )


def build_vec_env(config: PPOConfig, scenarios: tuple[str, ...]):
    """Single DummyVecEnv or a forked SubprocVecEnv depending on ``n_envs``."""
    factories = [
        _EnvSpec(
            scenarios,
            config.env_kind,
            config.learner,
            config.opponents,
            config.randomize_config,
        )
        for _ in range(max(config.n_envs, 1))
    ]
    if config.n_envs <= 1:
        return DummyVecEnv(factories)
    return SubprocVecEnv(factories, start_method="fork")


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
        target_kl=config.target_kl,
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
    scenarios = effective_scenarios(config)
    env = build_vec_env(config, scenarios)
    eval_env = DummyVecEnv(
        [
            _EnvSpec(
                scenarios,
                config.env_kind,
                config.learner,
                config.opponents,
                config.randomize_config,
            )
        ]
    )

    if resume is not None:
        model = PPO.load(resume, env=env, device=resolve_device(config.device))
        model.set_random_seed(config.seed)
    else:
        model = build_model(config, env)
    model.set_logger(configure(str(run_dir / "logs"), ["stdout", "csv"]))

    imitation_loss: float | None = None
    imitation_samples = 0
    imitation_requested = config.imitation_episodes > 0 or (
        config.env_kind == "drive" and config.imitation_scrambler_episodes > 0
    )
    if resume is None and imitation_requested:
        if config.env_kind == "drive":
            randomization = AdaptationRanges() if config.randomize_config else None
            datasets = []
            if config.imitation_episodes > 0:
                datasets.extend(
                    collect_drive_demonstrations(
                        get_scenario(name),
                        episodes=config.imitation_episodes,
                        seed_start=config.seed,
                        drive_config=DriveConfig(),
                        randomization=randomization,
                        curriculum_progress=1.0,
                    )
                    for name in scenarios
                )
            if config.imitation_scrambler_episodes > 0:
                datasets.extend(
                    collect_drive_demonstrations(
                        get_scenario(name),
                        episodes=config.imitation_scrambler_episodes,
                        seed_start=config.seed + 10_000,
                        drive_config=DriveConfig(),
                        randomization=randomization,
                        curriculum_progress=1.0,
                        quarterback=ScramblingQuarterback("qb", get_scenario(name).config),
                    )
                    for name in scenarios
                )
        elif config.env_kind == "role_play":
            datasets = [
                collect_role_play_demonstrations(
                    get_scenario(name),
                    learner_id=config.learner,
                    episodes=config.imitation_episodes,
                    seed_start=config.seed,
                    opponents=build_opponents(config.env_kind, name, config.opponents),
                )
                for name in scenarios
            ]
        elif config.env_kind == "role_drive":
            datasets = [
                collect_role_drive_demonstrations(
                    get_scenario(name),
                    learner_id=config.learner,
                    episodes=config.imitation_episodes,
                    seed_start=config.seed,
                    opponents=build_opponents(config.env_kind, name, config.opponents),
                )
                for name in scenarios
            ]
        elif config.env_kind == "receiver_drive":
            datasets = [
                collect_receiver_drive_demonstrations(
                    get_scenario(name),
                    learner_id=config.learner,
                    episodes=config.imitation_episodes,
                    seed_start=config.seed,
                    opponents=build_opponents(config.env_kind, name, config.opponents),
                )
                for name in scenarios
            ]
        else:
            datasets = [
                collect_demonstrations(
                    get_scenario(name),
                    episodes=config.imitation_episodes,
                    seed_start=config.seed,
                )
                for name in scenarios
            ]
        dataset = concatenate_demonstrations(datasets)
        imitation_samples = len(dataset.observations)
        imitation_loss = warm_start_policy(
            model,
            dataset,
            hidden=tuple(config.pi_layers),
            epochs=config.imitation_epochs,
            learning_rate=config.imitation_learning_rate,
            seed=config.seed,
            action_size=dataset.actions.shape[1],
        )
        print(
            f"imitation warm start: {imitation_samples} muestras "
            f"de {', '.join(scenarios)}, loss {imitation_loss:.5f}"
        )

    callbacks = [
        AttemptAnnealingCallback(config.attempt_decay_steps),
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
    if config.randomize_config:
        callbacks.append(AdaptationCurriculumCallback())
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
    metadata["scenarios"] = list(scenarios)
    metadata["imitation_samples"] = imitation_samples
    metadata["imitation_loss"] = imitation_loss
    metadata["env_kind"] = config.env_kind
    metadata["learner"] = config.learner
    metadata["n_envs"] = config.n_envs
    metadata["opponents"] = [list(pair) for pair in config.opponents]
    metadata["randomize_config"] = config.randomize_config
    if config.randomize_config:
        metadata["adaptation_ranges"] = dataclasses.asdict(AdaptationRanges())
        metadata["curriculum"] = {
            "decay_start_fraction": CURRICULUM_DECAY_START_FRACTION,
            "full_fraction": CURRICULUM_FULL_FRACTION,
        }
    if config.env_kind == "drive":
        metadata["observation_size"] = DRIVE_OBSERVATION_SIZE
    elif config.env_kind == "role_play":
        metadata["observation_size"] = RECEIVER_CONTEXT_OBSERVATION_SIZE
    elif config.env_kind == "role_drive":
        metadata["observation_size"] = DEFENDER_CONTEXT_OBSERVATION_SIZE
    elif config.env_kind == "receiver_drive":
        metadata["observation_size"] = RECEIVER_DRIVE_OBSERVATION_SIZE
    else:
        metadata["observation_size"] = OBSERVATION_SIZE
    if config.env_kind in ("role_play", "role_drive", "receiver_drive", "drive"):
        stage_b = get_scenario(scenarios[0]).config
        metadata["fatigue"] = {
            "carrier_fatigue_per_second": stage_b.carrier_fatigue_per_second,
            "carrier_min_speed_factor": stage_b.carrier_min_speed_factor,
        }
    if config.env_kind == "drive":
        metadata["drive"] = dataclasses.asdict(DriveConfig())
        metadata["route_pool"] = [route.value for route in ROUTE_OBSERVATION_ORDER]
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
    parser.add_argument(
        "--env",
        choices=("passing", "drive", "role_play", "role_drive", "receiver_drive"),
        default=defaults.env_kind,
    )
    parser.add_argument("--learner", default=defaults.learner)
    parser.add_argument(
        "--opponent",
        action="append",
        default=[],
        metavar="ID=PATH",
        help=(
            "Frozen opponent policy, e.g. qb=runs/ppo/final_model.zip, or ID=rules "
            "for a rule agent. Mix several per play with ID=rules|runs/... . Repeatable."
        ),
    )
    parser.add_argument(
        "--randomize-config",
        action="store_true",
        default=defaults.randomize_config,
        help="Sample capability ranges per episode with the adaptation curriculum.",
    )
    parser.add_argument(
        "--scenarios",
        default=None,
        help="Comma-separated scenario list to cycle during training.",
    )
    parser.add_argument("--run-name", default=defaults.run_name)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--checkpoint-freq", type=int, default=defaults.checkpoint_freq)
    parser.add_argument("--eval-freq", type=int, default=defaults.eval_freq)
    parser.add_argument("--eval-episodes", type=int, default=defaults.eval_episodes)
    parser.add_argument("--attempt-decay-steps", type=int, default=defaults.attempt_decay_steps)
    parser.add_argument("--imitation-episodes", type=int, default=defaults.imitation_episodes)
    parser.add_argument(
        "--imitation-scrambler-episodes",
        type=int,
        default=defaults.imitation_scrambler_episodes,
        help="Extra scrambling-QB demonstrations mixed into drive BC (0 disables).",
    )
    parser.add_argument("--imitation-epochs", type=int, default=defaults.imitation_epochs)
    parser.add_argument(
        "--imitation-learning-rate",
        type=float,
        default=defaults.imitation_learning_rate,
    )
    parser.add_argument("--n-envs", type=int, default=defaults.n_envs)
    parser.add_argument("--n-steps", type=int, default=defaults.n_steps)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args(argv)
    scenarios = (
        tuple(name.strip() for name in args.scenarios.split(",") if name.strip())
        if args.scenarios
        else ()
    )
    opponents: list[tuple[str, str]] = []
    for entry in args.opponent:
        parts = entry.split("=", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            parser.error(f"--opponent expects ID=PATH, got {entry!r}")
        opponents.append((parts[0], parts[1]))

    config = PPOConfig(
        run_name=args.run_name,
        scenario=args.scenario,
        env_kind=args.env,
        learner=args.learner,
        opponents=tuple(opponents),
        randomize_config=args.randomize_config,
        scenarios=scenarios,
        total_timesteps=args.timesteps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        attempt_decay_steps=args.attempt_decay_steps,
        imitation_episodes=args.imitation_episodes,
        imitation_scrambler_episodes=args.imitation_scrambler_episodes,
        imitation_epochs=args.imitation_epochs,
        imitation_learning_rate=args.imitation_learning_rate,
        output_dir=args.output_dir,
        device=args.device,
    )
    _, final_path = train(config, resume=args.resume)
    print(f"saved {final_path}")


if __name__ == "__main__":
    main()
