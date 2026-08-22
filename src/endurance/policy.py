"""A trained policy, on disk and back off it again.

`gym_env.PolicyStrategy` turns a `predict` callable into something the
engine cannot tell from a human strategy. This module supplies the callable
from a saved checkpoint, and it exists as a module rather than as a function
in the training script for one reason: **04 has to load the same checkpoint
03b scored**. A loader living in `scripts/evaluate.py` would be copied into
the app, and a second way of turning a file into a policy is decision 6's
failure arriving by a quieter route.

Nothing here trains, and nothing here decides anything about a race. It
reads a file and hands back a strategy.

Two formats, and which is which
-------------------------------
The `.zip` is the artefact of record: it is what reproduces, what resumes,
and what every number in 03b is measured from. The `.onnx` is an export for
04, so the hosted app can run the policy without torch. They must agree, and
03b's gate is where that is checked rather than assumed - an export that has
quietly diverged from the checkpoint would put the app on numbers nobody
produced.

The export carries **logits**, not the chosen action. 04 promises visible
action probabilities, and a softmax over logits gives them for free; an
export of the argmax alone would have thrown away the thing the panel is
for. `action_probabilities` at the foot of this module is what reads them,
and it lives here rather than in the app for the reason above: a second call
into the session is a second way of turning a checkpoint into an answer, and
one that could disagree with the decision the same policy is taking.

The card
--------
Every checkpoint gets a JSON sidecar recording which dials and which seed
bank it was trained against, and `load_policy` refuses one that does not
match the race it is about to be scored on. This is `assets.py`'s tripwire
applied to the artefact most likely to outlive a config change: a policy
trained on one set of dials and evaluated on another produces a complete,
plausible, meaningless table.

Neither Stable-Baselines3 nor onnxruntime is imported at module scope. The
package must import without them - `endurance` is a race simulator and the
training dependencies belong to the two scripts that train.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .assets import SeedBank, dials_fingerprint
from .gym_env import N_ACTIONS, N_OBS, PolicyStrategy


def file_sha256(path: str | Path) -> str:
    """A digest of the artefact itself, not of what it was trained against.

    The dials and bank fingerprints answer "is this policy about this race".
    They cannot answer "are these the weights that were measured", and at 03b
    that turned out to matter: torch's dynamo export path writes the network
    into a companion `.data` file, so an `.onnx` can travel without its
    weights and pick up whatever `.data` happens to be sitting beside it. No
    error, wrong policy.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def bank_fingerprint(bank: SeedBank) -> str:
    """A short hash of the races, not of the rule that drew them.

    The headline list and the held-out list both go in. Two banks with the
    same draw seed but different sizes are different experiments, and the
    draw seed alone would not say so.
    """
    payload = json.dumps({"headline": bank.headline,
                          "held_out": bank.held_out}, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


# ----------------------------------------------------------------------
# The card
# ----------------------------------------------------------------------
@dataclass
class PolicyCard:
    """What a checkpoint is about, written down beside it.

    Held as a separate file rather than inside the `.zip` so that it can be
    read without Stable-Baselines3 installed, and so the ONNX export - which
    has nowhere to put metadata - is covered by the same record.
    """

    series_code: str
    algorithm: str
    dials_fingerprint: str
    bank_fingerprint: str
    train_seed: int
    total_timesteps: int
    n_envs: int = 1
    checkpoint: str = ""          # file name, not path: the folder may move
    onnx: str = ""
    checkpoint_sha256: str = ""   # the artefact, not what it is about
    onnx_sha256: str = ""
    trained_at: str = ""
    notes: dict = field(default_factory=dict)

    @classmethod
    def path_for(cls, checkpoint: str | Path) -> Path:
        return Path(checkpoint).with_suffix(".card.json")

    def save(self, checkpoint: str | Path) -> Path:
        path = self.path_for(checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, checkpoint: str | Path) -> "PolicyCard":
        path = cls.path_for(checkpoint)
        if not path.exists():
            raise FileNotFoundError(
                f"no card beside {Path(checkpoint).name}. A checkpoint "
                f"without one cannot be checked against the race it is "
                f"being scored on; retrain, or write the card by hand if "
                f"you know what it was trained against.")
        return cls(**json.loads(path.read_text()))

    def check_file(self, path: str | Path) -> None:
        """Refuse an artefact whose bytes have moved since the card was written.

        Silent on a card that records no digest, so cards written before 03b's
        amendment still load. A missing digest is a gap; a wrong one is a lie,
        and only the second is worth refusing over.
        """
        path = Path(path)
        recorded = (self.onnx_sha256 if path.suffix == ".onnx"
                    else self.checkpoint_sha256)
        if not recorded:
            return
        actual = file_sha256(path)
        if actual != recorded:
            raise ValueError(
                f"{path.name} hashes to {actual!r}; the card written beside "
                f"it records {recorded!r}. These are not the weights that "
                f"were scored. If this is an .onnx, check whether a stale "
                f"companion .data file is sitting next to it.")

    def check(self, config=None, bank: SeedBank | None = None) -> None:
        """Refuse a policy that is about a different race from this one."""
        if config is not None:
            actual = dials_fingerprint(config)
            if actual != self.dials_fingerprint:
                raise ValueError(
                    f"{self.checkpoint or 'this policy'} was trained against "
                    f"dials {self.dials_fingerprint!r}; the race it is being "
                    f"scored on is {actual!r}. Same seeds, different races.")
            if config.series_code != self.series_code:
                raise ValueError(
                    f"policy is for {self.series_code!r}, race is for "
                    f"{config.series_code!r}. The two series are never pooled.")
        if bank is not None:
            actual = bank_fingerprint(bank)
            if actual != self.bank_fingerprint:
                raise ValueError(
                    f"{self.checkpoint or 'this policy'} was trained against "
                    f"bank {self.bank_fingerprint!r}, not {actual!r}. The "
                    f"held-out set may no longer be held out.")


# ----------------------------------------------------------------------
# Turning a file into a `predict`
# ----------------------------------------------------------------------
def _sb3_predict(path: Path):
    """The `.zip`, through Stable-Baselines3.

    Loaded with `device="cpu"` and no environment: evaluation runs one
    observation at a time inside a race, so there is nothing for a GPU to
    do and an env would only be a second definition of the race.
    """
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as e:                                  # pragma: no cover
        raise ImportError(
            "loading a .zip checkpoint needs sb3-contrib: "
            "pip install 'sb3-contrib>=2.3'") from e

    model = MaskablePPO.load(path, device="cpu")

    def logits(obs):
        """The same vector through torch, so a `.zip` answers 04 identically.

        Torch is imported inside the closure: this module must still import
        on a machine that has neither Stable-Baselines3 nor torch.
        """
        import torch

        tensor, _ = model.policy.obs_to_tensor(
            np.asarray(obs, dtype=np.float32))
        with torch.no_grad():
            features = model.policy.extract_features(tensor)
            latent = model.policy.mlp_extractor.forward_actor(features)
            return model.policy.action_net(latent).cpu().numpy()[0]

    # Wrapped rather than returned bare: `model.predict` is a bound method
    # and will not take an attribute, and hanging one on the underlying
    # function would set it for every model in the process.
    def predict(obs, deterministic: bool = True, state=None, **kw):
        return model.predict(obs, deterministic=deterministic, state=state,
                             **kw)

    predict.logits = logits
    return predict, model


def _onnx_predict(path: Path):
    """The `.onnx`, through onnxruntime.

    No mask is applied, matching `PolicyStrategy` and decision F: the
    engine's override is what holds the agent to the rules, exactly as it
    holds the five humans. Deterministic is an argmax over the logits;
    sampling uses the softmax, so the two agree on the mode by construction.
    """
    try:
        import onnxruntime as ort
    except ImportError as e:                                  # pragma: no cover
        raise ImportError(
            "loading a .onnx export needs onnxruntime: "
            "pip install onnxruntime") from e

    session = ort.InferenceSession(str(path),
                                   providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    rng = np.random.default_rng(0)

    def predict(obs, deterministic: bool = True, state=None, **_kw):
        batch = np.asarray(obs, dtype=np.float32).reshape(1, N_OBS)
        logits = np.asarray(session.run(None, {input_name: batch})[0])[0]
        if deterministic:
            return int(np.argmax(logits)), None
        shifted = logits - logits.max()
        p = np.exp(shifted) / np.exp(shifted).sum()
        return int(rng.choice(N_ACTIONS, p=p)), None

    def logits(obs):
        """The vector `predict` takes its argmax of, for 04's panel.

        Same session as `predict`, so the ranking on the page and the
        decision in the race cannot come apart.
        """
        batch = np.asarray(obs, dtype=np.float32).reshape(1, N_OBS)
        return np.asarray(session.run(None, {input_name: batch})[0])[0]

    predict.logits = logits
    return predict, session


def load_policy(path: str | Path, config=None, bank: SeedBank | None = None,
                deterministic: bool = True, check: bool = True
                ) -> PolicyStrategy:
    """A checkpoint, wearing the ordinary strategy interface.

    `config` and `bank` are optional only so a policy can be inspected on
    its own. Any call that is going to *score* the policy should pass both,
    because that is what turns the card from a note into a check.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}")

    if check:
        card = PolicyCard.load(path)
        card.check(config, bank)
        card.check_file(path)

    external = Path(str(path) + ".data")
    if path.suffix == ".onnx" and external.exists():
        raise ValueError(
            f"{path.name} keeps its weights in {external.name}, so the "
            f"export is not self-contained and 04 cannot ship one file. "
            f"Re-export with dynamo=False; see export_onnx in train.py.")

    if path.suffix == ".zip":
        predict, _held = _sb3_predict(path)
    elif path.suffix == ".onnx":
        predict, _held = _onnx_predict(path)
    else:
        raise ValueError(f"unknown checkpoint format {path.suffix!r}; "
                         f"expected .zip or .onnx")

    strategy = PolicyStrategy(predict, deterministic=deterministic)
    # Kept alive on the strategy: an onnxruntime session that gets collected
    # under a policy still holding its `predict` closure segfaults rather
    # than raising, which is a bad afternoon to debug.
    strategy._backend = _held
    strategy._source = str(path)
    return strategy


# ----------------------------------------------------------------------
# What the policy thought, before it chose
# ----------------------------------------------------------------------
def action_logits(strategy: PolicyStrategy, obs) -> np.ndarray:
    """The raw vector, from whichever backend loaded this policy.

    Raises rather than returning `None` for a policy that cannot say: a panel
    that silently renders nothing is worse than one that says why, and 04
    catches this and prints the reason.
    """
    fn = getattr(strategy.predict, "logits", None)
    if fn is None:
        raise AttributeError(
            "this policy has no logits handle; load it through `load_policy` "
            "rather than building a `PolicyStrategy` around a bare callable")
    return np.asarray(fn(obs), dtype=np.float64)


def action_probabilities(strategy: PolicyStrategy, obs) -> np.ndarray:
    """The softmax of the above, which is what 04's panel shows.

    **Unmasked**, matching `PolicyStrategy` and the way the agent was scored:
    the engine's override is what holds it to the rules, as it does the five
    humans. So a policy asking to stay out on an empty tank shows a high
    probability on *stay out*, and the page says beside it that the rules
    take that decision anyway. Masking here would make the panel disagree
    with the race going on next to it.

    A ranking, not a magnitude. MaskablePPO has no Q(s,a), and wanting one is
    a retraining job rather than a line of arithmetic.
    """
    z = action_logits(strategy, obs)
    z = z - z.max()                      # shift for overflow, not for taste
    e = np.exp(z)
    return e / e.sum()


# ----------------------------------------------------------------------
# Putting it in the roster
# ----------------------------------------------------------------------
def agent_roster(roster: dict, strategy: PolicyStrategy,
                 name: str = "agent") -> dict:
    """The five humans plus the agent, in the shape `compare_roster` wants.

    `compare_roster` calls each value to get a fresh strategy per race, and
    the natural-looking `lambda: load_policy(...)` would reload the
    checkpoint two hundred times a series. `PolicyStrategy` holds no
    per-race state - it reads the observation it is handed and returns a
    decision - so one instance is shared, which is also what makes the
    agent's row reproducible.
    """
    if name in roster:
        raise KeyError(f"{name!r} is already in the roster; the agent must "
                       f"be a sixth member, not a replacement for a human")
    return {**roster, name: lambda: strategy}
