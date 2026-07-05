---
layout: post
title: "rl-factory: an RL codebase where a new algorithm is a plug-in, not a rewrite"
date: 2026-06-28 09:00:00
description: "The lean spine that lets DQN, PPO, ICM, RND, and kNN-novelty share one training loop — where adding a model means registering a learner, not branching the control flow."
tags: [reinforcement-learning, software-design, factory-pattern, pytorch]
categories: rl-from-scratch
related_posts: true
---

This is the first post in a series that trains a reinforcement-learning agent to play **Super
Mario Bros** from scratch, one algorithm at a time — DQN, PPO, ICM, RND, and kNN episodic
novelty. Before any single algorithm, this post is about the *thing they all plug into*: a small
framework I keep calling the **rl-factory**, because its whole job is to make new algorithms cheap
to add.

The claim I want to earn: **adding a new RL algorithm to this codebase is registering a learner,
not rewriting a training loop.** Everything in the series after this one is an exercise in cashing
that claim — each model drops into the same spine and the diff is tiny.

If you remember one sentence: **there is exactly one real axis of variation in these algorithms —
on-policy vs. off-policy — so the framework encodes that as a single choice and treats everything
else (DQN vs. PPO vs. curiosity) as a swapped-in `Learner`.**

> **The code.** The engine lives in [`rl-factory`]({{ site.factory_repo }}); the five models that
> plug into it live in [`rl-ablations`]({{ site.ablations_repo }}). This post walks the engine.

## Part 0 — the trap this avoids

The naïve way to grow an RL repo is one script per algorithm: `train_dqn.py`, `train_ppo.py`,
`train_icm.py`. Each re-implements the run loop (step budget, checkpointing, logging, the
env-stepping), and they drift. By the fifth algorithm you have five subtly different training
loops and no way to compare results fairly, because "the setup" isn't one thing.

The opposite trap is a single `train()` with a growing thicket of `if algo == "ppo": ... elif algo
== "icm": ...`. That centralizes the loop but scatters each algorithm's logic across every
function it touches.

rl-factory takes a third path: **one shared loop, and a small typed contract that each algorithm
implements.** The loop never mentions a specific algorithm; an algorithm never re-implements the
loop.

## Part 1 — the three pieces

The spine is three things:

1. **Typed interfaces** (`rl_factory/data_classes/`) — the contracts: `Agent` (something that acts),
   `Learner` (something that turns experience into a weight update), `Environment`, and the
   config objects. Nothing here knows about Mario or about DQN; they're just shapes.

2. **The `Trainer`** (`rl_factory/training/trainer.py`) — owns only the *shared* concerns: the
   env-step budget, periodic checkpointing, and stats logging. It drives a strategy through four
   methods and is otherwise blind to what's being trained:

   ```python
   self._strategy.setup()
   while steps < config.steps:
       done, metrics = self._strategy.step()   # collect + learn one unit
       steps += done
       stats.record(metrics=metrics)
       # ... checkpoint / log on schedule ...
   self._strategy.teardown()
   ```

   That's the entire training loop for *every* algorithm in the series. DQN and PPO and ICM all
   run this exact code.

3. **The `TrainStrategy`** (`rl_factory/training/strategy.py`) — the pluggable "how." A four-method
   contract: `setup / step / weights / teardown`.

   ```python
   class TrainStrategy(ABC):
       @abstractmethod
       def setup(self) -> None: ...
       @abstractmethod
       def step(self) -> tuple[int, dict[str, float]]: ...   # (env-steps done, metrics)
       @abstractmethod
       def weights(self) -> StateDict: ...
       @abstractmethod
       def teardown(self) -> None: ...
   ```

   A strategy owns *how experience is collected and learned from*; the Trainer owns *when to stop,
   save, and log*. Clean seam.

## Part 2 — the one branch that's real: on-policy vs off-policy

Here's the design decision the whole thing pivots on. If you line up DQN, PPO, and the three
curiosity methods, they differ in a hundred small ways — but those differences collapse onto **one**
structural axis:

- **Off-policy** (DQN, the random baseline): collect into a **replay buffer**, learn from sampled
  batches with a TD target against a shared Q-network.
- **On-policy** (PPO, ICM, RND, kNN): collect a fresh **rollout** from a vectorized env, learn from
  it once with GAE, throw it away.

So there are exactly **two** concrete strategies — `OffPolicyStrategy` and `OnPolicyStrategy` — and
which one a model uses is not code, it's **data**: a `family` each model declares when it registers.

```python
class StrategyFamily(StrEnum):
    OFF_POLICY = "off_policy"   # async actors + replay buffer (DQN / random)
    ON_POLICY  = "on_policy"    # vectorized env + on-policy rollout (PPO / ICM / RND / kNN)

# ...and each model just names its family when it registers itself (rl_ablations/register.py):
register_model("dqn", build_agent=dqn.build_agent, build_learner=dqn.build_learner, family=OFF)
register_model("ppo", build_agent=ppo.build_agent, build_learner=ppo.build_learner, family=ON)
register_model("icm", build_agent=ppo.build_agent, build_learner=icm.build_learner, family=ON)
```

Adding ICM did not add an `if icm` anywhere in the training loop. ICM is on-policy, so it inherits
`OnPolicyStrategy` **unchanged**. The only thing that makes it ICM instead of PPO is a different
learner (more on that next). That is the single most important sentence in this whole codebase:

> ICM rides the on-policy strategy unchanged — it's just a different `build_learner`.

## Part 3 — a model, factored

Every model is the same four files, so once you've read one you can read them all:

| file | responsibility |
|------|----------------|
| `config.py` | the model's hyper-parameters (a typed config) |
| `model.py`  | the network(s) — nn.Modules only, no training logic |
| `learner.py`| turns a batch of experience into a loss + weight update |
| `factory.py`| `build_agent(...)` and `build_learner(...)` — the entry points the factory calls |

The `Agent` (acting policy) and the `Learner` (weight update) are deliberately separate. That
separation is why the curiosity models are so cheap: **ICM, RND, and kNN all act with a plain PPO
policy** — they only change *how the update is computed*. You can read that straight off the
registrations — they pass **the same `build_agent`** (PPO's) and differ only in `build_learner`:

```python
# rl_ablations/register.py — icm/rnd/knn share PPO's acting policy; only the learner differs
register_model("ppo", build_agent=ppo.build_agent, build_learner=ppo.build_learner, family=ON)
register_model("icm", build_agent=ppo.build_agent, build_learner=icm.build_learner, family=ON)
register_model("rnd", build_agent=ppo.build_agent, build_learner=rnd.build_learner, family=ON)
register_model("knn", build_agent=ppo.build_agent, build_learner=knn.build_learner, family=ON)
```

Each curiosity `build_learner` wraps PPO's update with an intrinsic-reward term. Same policy, same
rollout, same loop; a different learner. Novelty in, exploration out.

## Part 4 — the factory tie

`rl_factory.algorithm(name)` is the one call that assembles a registered model into everything the
Trainer needs — it looks the model up in the registry and closes over its `family`:

```python
def algorithm(name):
    spec = _REGISTRY[name]                       # build_agent, build_learner, family

    def make_strategy(build_env, config):
        return _build_strategy(spec.family, build_env, spec.build_agent, spec.build_learner, config)

    return Algorithm(build_agent=spec.build_agent, make_strategy=make_strategy)
```

Note what `make_strategy` takes: a `build_env` **passed in from outside**. rl-factory never imports
Mario — the environment is a parameter. That's what keeps it a *framework* and not a Mario script,
and it's the seam this project splits along: [`rl-factory`]({{ site.factory_repo }}) is the engine in
this post, and [`rl-ablations`]({{ site.ablations_repo }}) is the separate repo that brings the Mario
environment and registers the five models on top of it.

## Part 5 — so what does "add a model" actually cost?

Concretely, to add an algorithm:

1. Write its four files (`config / model / learner / factory`).
2. Call `register_model(name, build_agent=…, build_learner=…, family=…)` — one line.

That's the whole cost. No new training loop. No new checkpointing, logging, or env-stepping. If it's on-policy, no new
*strategy* either — you're writing a `Learner` and nothing else structural. That's the payoff, and
it's why the rest of this series can move fast: each post is one model dropped into this frame, and
the interesting part is the *idea*, because the plumbing is already paid for.

## What's next

From here the series walks the models in the order I built them, and each one earns its place by
fixing a limitation of the last:

1. **DQN** — off-policy value learning; beat the random floor.
2. **PPO** — on-policy policy gradients; a more stable learner.
3. **ICM** — add curiosity so Mario explores when the reward is sparse.
4. **RND** — a simpler curiosity signal (and why simpler was better).
5. **kNN episodic novelty** — a count-based signal that *can't* saturate the way the others did.

Every one of them is the same spine from this post, plus a `Learner`. Let's start beating the
floor.
