"""04's one change to the package, offered as a patch rather than applied.

`policy.py`'s own docstring says the export carries **logits** rather than
the chosen action, and gives the reason: "04 promises visible action
probabilities, and a softmax over logits gives them for free; an export of
the argmax alone would have thrown away the thing the panel is for."

The export does carry them. Nothing reads them. `_onnx_predict` runs the
session, takes an argmax and drops the vector on the floor, and the only
handle left is `strategy._backend`, which is a private attribute the module
keeps alive so onnxruntime does not segfault.

So 04 needs a way to ask a loaded policy what it thought, and there are two
places it could live. In the app, where it would be a second call into the
session and therefore a second way of turning a checkpoint into an answer -
decision 6's failure arriving by the quiet route `policy.py` was written to
close. Or here, beside the loader, where the two paths share one session and
cannot disagree. It goes here.

**This is a package change and wants signing off before it lands.** Applied
to `src/endurance/policy.py`; three edits, none of which changes what
`load_policy` returns to anything that already calls it.
"""

# ----------------------------------------------------------------------
# EDIT 1 - in `_onnx_predict`, after `predict` is defined and before the
# return. Same session, so the panel and the decision cannot disagree.
# ----------------------------------------------------------------------
"""
    def logits(obs):
        batch = np.asarray(obs, dtype=np.float32).reshape(1, N_OBS)
        return np.asarray(session.run(None, {input_name: batch})[0])[0]

    predict.logits = logits
    return predict, session
"""

# ----------------------------------------------------------------------
# EDIT 2 - in `_sb3_predict`, the same handle through torch, so a `.zip`
# and a `.onnx` answer the panel identically. Torch is imported inside the
# closure: the package must still import without it.
# ----------------------------------------------------------------------
"""
    def logits(obs):
        import torch
        tensor, _ = model.policy.obs_to_tensor(np.asarray(obs, dtype=np.float32))
        with torch.no_grad():
            latent = model.policy.mlp_extractor.forward_actor(
                model.policy.extract_features(tensor))
            return model.policy.action_net(latent).cpu().numpy()[0]

    model.predict.__func__.logits = logits   # see note below
    return model.predict, model
"""

# ----------------------------------------------------------------------
# EDIT 3 - the public reader, at module level.
# ----------------------------------------------------------------------
"""
def action_logits(strategy: PolicyStrategy, obs) -> np.ndarray:
    \"\"\"What the policy thought, before it chose.

    `None` is not returned for a policy that cannot say: a panel that
    silently renders nothing is worse than one that says the checkpoint
    format does not expose it, so this raises and the page catches it.
    \"\"\"
    fn = getattr(strategy.predict, "logits", None)
    if fn is None:
        raise AttributeError(
            "this policy was not loaded with a logits handle; load it "
            "through `load_policy` rather than constructing "
            "`PolicyStrategy` directly")
    return np.asarray(fn(obs), dtype=np.float64)


def action_probabilities(strategy: PolicyStrategy, obs) -> np.ndarray:
    \"\"\"The softmax of the above, which is what the panel shows.

    Unmasked, matching `PolicyStrategy` and the way the agent was scored:
    the engine's override is what holds it to the rules. So a policy asking
    to stay out on an empty tank shows a high probability on *stay out*, and
    the page says beside it that the rules take that decision anyway.
    \"\"\"
    z = action_logits(strategy, obs)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()
"""

# ----------------------------------------------------------------------
# A note on EDIT 2, which is uglier than EDIT 1 and says something.
# ----------------------------------------------------------------------
# `model.predict` is a bound method and will not take an attribute, so the
# SB3 branch has to hang the handle somewhere else - on the underlying
# function, or by wrapping `predict` in a plain closure. Wrapping is the
# cleaner of the two and is what should land:
"""
    def predict(obs, deterministic: bool = True, state=None, **kw):
        return model.predict(obs, deterministic=deterministic, state=state, **kw)

    predict.logits = logits
    return predict, model
"""
# The `.onnx` is the artefact 04 actually loads, so EDIT 2 is there to keep
# the two formats answering the same question rather than because the app
# needs it. If it is judged not worth the wrapper, drop EDIT 2 and let
# `action_logits` raise on a `.zip` - but then 03b's gate two, which checks
# that the export agrees with the checkpoint, is checking a narrower thing
# than the app relies on.
