"""Train one policy per series on the 03a environment.

    python scripts/train.py imsa
    python scripts/train.py wec --timesteps 800000 --seed 1
    python scripts/train.py            # both series, defaults

Produces, per series, in `outputs/policies/`:

    {series}_maskable_ppo.zip          the artefact of record
    {series}_maskable_ppo.onnx         the export 04 loads
    {series}_maskable_ppo.card.json    what it was trained against
    {series}_monitor/                  episode returns, for the notebook

Nothing here scores anything. The agent's number comes from
`scripts/evaluate.py`, which puts it in `ROSTER` as a sixth member and calls
`harness.compare_roster` on the same banks, field and pace rank as the five
humans. There is deliberately no evaluation code in this file: a second
evaluation path that can differ from the roster's is the failure decision 6
exists to prevent, and it produces plausible numbers rather than errors.

Why MaskablePPO
---------------
The env publishes a mask on every step, and a policy that ignores it spends
capacity on decisions the engine overrides. `sb3-contrib`'s MaskablePPO
applies the mask inside the action distribution, during rollout *and* when
computing the loss. DQN was the alternative and gives Q(s,a) directly, which
04's panel could show with a real unit - but masking it means a custom
policy applying the mask both at the acting argmax and in the target, and
missing the second is silent.

The mask is a training convenience and does not survive into evaluation:
`gym_env.PolicyStrategy` carries none, so at scoring time an agent asking to
stay out on an empty tank is overridden by the engine and scored on that,
exactly as a human strategy would be. `MaskablePPO.predict` called without
`action_masks` is what makes that work, and it is the intended behaviour
rather than an oversight.

Training draws only from the headline bank
------------------------------------------
`EnduranceEnv` refuses held-out seeds, which covers the obvious mistake. It
does not cover a subtler one: SB3 seeds its environments by calling
`env.reset(seed=base + i)` with integers of its own choosing, and those are
not in the bank. `HeadlineSeeded` below maps any seed SB3 supplies onto a
headline race, so the training distribution is the bank rather than whatever
`set_random_seed` happened to produce.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np

def find_project_root(marker: str = "src/endurance") -> Path:
    """Walk up until the project appears, as the notebooks do.

    Not `parents[1]`. A hard-coded depth is correct only while the file sits
    where its author left it, and silently resolves to the wrong folder the
    moment somebody moves it - which puts a non-existent `src` on the path
    and reports a missing package rather than a misplaced script.
    """
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(
        f"could not find {marker!r} at or above {here.parent}.\n\n"
        "This script belongs in `scripts/`, beside `src/` and `notebooks/`:\n"
        "  endurance-strategy-rl/\n"
        "    data/processed/\n"
        "    notebooks/\n"
        "    scripts/          this file\n"
        "    src/endurance/    the package\n")


ROOT = find_project_root()
sys.path.insert(0, str(ROOT / "src"))
# Both, so `freeze_assets` is importable whether this file is run from
# `scripts/` or from wherever an editor decided to put it.
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from endurance.assets import dials_fingerprint                  # noqa: E402
from endurance.gym_env import REWARD, EnduranceEnv              # noqa: E402
from endurance.policy import (                                  # noqa: E402
    PolicyCard, bank_fingerprint, file_sha256,
)
from freeze_assets import SERIES, load_assets                   # noqa: E402


OUT = ROOT / "outputs" / "policies"

# Defaults. 03a measured a six-hour race at about 0.15 s and an episode at
# roughly 190 steps, so 500k steps is a few thousand races and some minutes
# rather than an overnight run - the wrapper is not the bottleneck in any
# training loop and there is no reason to be stingy.
TIMESTEPS = 500_000
N_ENVS = 8
TRAIN_SEED = 0

# Left at the SB3 defaults except for the two the episode shape argues for.
# `n_steps` is per env, so 8 x 512 is about twenty races a rollout, which
# means a rollout spans enough caution timelines to average over them.
# `gamma` is 1.0 because the episode is a fixed-length race scored at the
# flag: discounting would tell the agent that a place lost in hour six
# matters less than one lost in hour one, which is false.
HYPERPARAMS = dict(
    n_steps=512,
    batch_size=512,
    gamma=1.0,
    gae_lambda=0.95,
    learning_rate=3e-4,
    ent_coef=0.01,
    policy_kwargs=dict(net_arch=[64, 64]),
)


class HeadlineSeeded(gym.Wrapper):
    """Map whatever seed the trainer supplies onto a headline race.

    Not a convenience. `EnduranceEnv._pick_seed` refuses the held-out fifty
    but accepts any other integer, so SB3's own seeding would quietly train
    on races drawn from the whole seed space rather than from the two
    hundred the roster was scored on. Training and evaluation would then be
    about different distributions while every artefact still claimed the
    bank.

    `seed=None` is passed through untouched, which is the ordinary path: the
    env draws its own race from the headline bank.
    """

    def __init__(self, env: EnduranceEnv):
        super().__init__(env)
        self._headline = list(env.bank.headline)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            seed = self._headline[seed % len(self._headline)]
        return self.env.reset(seed=seed, options=options)


def make_env(config, field, bank, monitor_dir: Path | None, rank: int):
    """One env factory, deferred - `SubprocVecEnv` pickles this, not the env.

    That matters because `EnduranceEnv` holds a suspended generator once it
    is running, and a generator cannot be pickled.
    """
    from stable_baselines3.common.monitor import Monitor

    def _init():
        env = HeadlineSeeded(EnduranceEnv(config, field, bank))
        if monitor_dir is not None:
            monitor_dir.mkdir(parents=True, exist_ok=True)
            env = Monitor(env, str(monitor_dir / f"{rank:02d}"))
        return env

    return _init


def export_onnx(model, path: Path) -> None:
    """Export the policy's logits, so 04 can run it without torch.

    Logits rather than the chosen action: the app promises visible action
    probabilities and a softmax gives them, whereas an exported argmax would
    have thrown away the thing the explainability panel is for.

    No mask in the graph. The export must agree with the `.zip` under
    `PolicyStrategy`, which is unmasked, and baking a mask in here would
    make the two disagree by construction.
    """
    import torch

    policy = model.policy
    policy.set_training_mode(False)

    class Logits(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, obs):
            features = self.policy.extract_features(obs)
            if isinstance(features, tuple):        # separate extractors
                latent_pi = self.policy.mlp_extractor.forward_actor(features[0])
            else:
                latent_pi, _latent_vf = self.policy.mlp_extractor(features)
            return self.policy.action_net(latent_pi)

    dummy = torch.zeros(1, model.observation_space.shape[0], dtype=torch.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    # `dynamo=False` on purpose. Recent torch defaults to the dynamo exporter,
    # which externalises the initialisers into a companion `.data` file - so a
    # 20 kB policy net leaves a 2 kB `.onnx` holding nothing but a graph and a
    # filename. That is two deliverables where the card records one, and an
    # `.onnx` beside a stale `.data` loads the previous run's weights without
    # raising. The legacy exporter keeps a model this size inline.
    export_kwargs = dict(
        input_names=["observation"], output_names=["logits"],
        dynamic_axes={"observation": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    # A `.data` file left behind by an earlier export (e.g. one that ran
    # before this guard existed) would make the check below fire on stale
    # evidence even when this export is self-contained. Clear it first so
    # the check only ever reflects what this export just did.
    external = Path(str(path) + ".data")
    external.unlink(missing_ok=True)

    try:
        torch.onnx.export(Logits(policy), dummy, str(path),
                          dynamo=False, **export_kwargs)
    except TypeError:
        # Older torch has no `dynamo` argument and no dynamo default either.
        torch.onnx.export(Logits(policy), dummy, str(path), **export_kwargs)

    if external.exists():
        raise RuntimeError(
            f"the export wrote its weights to {external.name} despite "
            f"dynamo=False, so {path.name} is not self-contained. Do not "
            f"ship it: an .onnx beside the wrong .data loads silently.")
    if path.stat().st_size < 8_000:
        raise RuntimeError(
            f"{path.name} is {path.stat().st_size} bytes, too small to hold "
            f"the policy network - the weights have gone somewhere else.")

    # Exported and then checked, because an export that silently disagrees
    # with the checkpoint is the one failure this whole file cannot see.
    # The gate in tests/ does this properly over a whole evaluation pass;
    # this is the cheap version that stops a broken file being written.
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    rng = np.random.default_rng(0)
    probe = rng.random((32, model.observation_space.shape[0])).astype(np.float32)
    exported = np.argmax(session.run(None, {name: probe})[0], axis=1)
    native, _ = model.predict(probe, deterministic=True)
    if not np.array_equal(exported, np.asarray(native).ravel()):
        raise RuntimeError(
            f"the ONNX export at {path.name} disagrees with the checkpoint "
            f"on {int((exported != np.asarray(native).ravel()).sum())} of 32 "
            f"probe observations. Do not use either until this is understood.")


def train(series_code: str, timesteps: int = TIMESTEPS, n_envs: int = N_ENVS,
          seed: int = TRAIN_SEED, out: Path = OUT, onnx: bool = True) -> Path:
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    config, bank, field = load_assets(series_code)
    stem = out / f"{series_code}_maskable_ppo"
    monitor_dir = out / f"{series_code}_monitor"

    print(f"{series_code}: {config.name}, dials {dials_fingerprint(config)}, "
          f"bank {bank_fingerprint(bank)}")
    print(f"    {len(bank.headline)} headline races, "
          f"{len(bank.held_out)} held out and refused")
    print(f"    {timesteps:,} steps over {n_envs} envs, train seed {seed}")

    vec_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    venv = vec_cls([make_env(config, field, bank, monitor_dir, i)
                    for i in range(n_envs)])

    model = MaskablePPO("MlpPolicy", venv, seed=seed, verbose=1,
                        device="cpu", **HYPERPARAMS)
    model.learn(total_timesteps=timesteps, progress_bar=False)

    checkpoint = stem.with_suffix(".zip")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint)
    print(f"    wrote {checkpoint.relative_to(ROOT)}")

    if onnx:
        export_onnx(model, stem.with_suffix(".onnx"))
        print(f"    wrote {stem.with_suffix('.onnx').relative_to(ROOT)}")

    card = PolicyCard(
        series_code=series_code,
        algorithm="MaskablePPO",
        dials_fingerprint=dials_fingerprint(config),
        bank_fingerprint=bank_fingerprint(bank),
        train_seed=seed,
        total_timesteps=timesteps,
        n_envs=n_envs,
        checkpoint=checkpoint.name,
        onnx=stem.with_suffix(".onnx").name if onnx else "",
        checkpoint_sha256=file_sha256(checkpoint),
        onnx_sha256=file_sha256(stem.with_suffix(".onnx")) if onnx else "",
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes={
            "race": config.name,
            "dials_source": "STAND-IN, not calibrated (see freeze_assets.py)",
            "trained_on": "headline bank only; held-out refused by the env",
            "hyperparameters": {k: v for k, v in HYPERPARAMS.items()
                                if k != "policy_kwargs"},
            "net_arch": HYPERPARAMS["policy_kwargs"]["net_arch"],
            "mask": "applied in training; PolicyStrategy carries none",
            # Read from the module, never retyped: a card that claims a
            # reward the code no longer uses is worse than no card.
            "reward": REWARD,
        },
    )
    card.save(checkpoint)
    print(f"    wrote {PolicyCard.path_for(checkpoint).relative_to(ROOT)}")

    venv.close()
    return checkpoint


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("series", nargs="*", default=list(SERIES),
                        help="imsa, wec, or nothing for both")
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--envs", type=int, default=N_ENVS)
    parser.add_argument("--seed", type=int, default=TRAIN_SEED,
                        help="training seed; recorded on the card")
    parser.add_argument("--no-onnx", action="store_true",
                        help="skip the export (the .zip is what is scored)")
    args = parser.parse_args(argv[1:])

    unknown = [s for s in args.series if s not in SERIES]
    if unknown:
        raise SystemExit(f"unknown series {unknown}; choose from {list(SERIES)}")

    for series_code in args.series:
        train(series_code, timesteps=args.timesteps, n_envs=args.envs,
              seed=args.seed, onnx=not args.no_onnx)
        print()

    print("Two policies, two tables, never pooled. Score them with:")
    print("    python scripts/evaluate.py")


if __name__ == "__main__":
    main(sys.argv)
