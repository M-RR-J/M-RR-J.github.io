---
layout: about
title: about
permalink: /
subtitle:

profile:
  align: right
  image: # no hero image for now (Mario clip removed — didn't look good yet)
  image_circular: false
  more_info:

selected_papers: false # includes a list of papers marked as "selected={true}"
social: true # includes social icons at the bottom of the page

announcements:
  enabled: false # includes a list of news items
  scrollable: true # adds a vertical scroll bar if there are more than 3 news items
  limit: 5 # leave blank to include all the news in the `_news` folder

latest_posts:
  enabled: true
  scrollable: true # adds a vertical scroll bar if there are more than 3 new posts items
  limit: 3 # leave blank to include all the blog posts
---

### Start here — the RL-from-scratch series

The series builds a Mario agent from scratch, each model earning its place by fixing a
limitation of the last, all sharing one small framework:

1. [**rl-factory**](/blog/2026/rl-factory/) — the engine: where adding an algorithm is registering a learner, not rewriting a training loop.
2. [**DQN**](/blog/2026/dqn/) — off-policy value learning; beat the random floor.
3. [**PPO**](/blog/2026/ppo/) — on-policy policy gradients; a more stable learner.
4. [**ICM**](/blog/2026/icm/) — curiosity from forward-prediction error, so Mario explores when reward is sparse.
5. [**RND**](/blog/2026/rnd/) — a simpler curiosity signal (and why simpler was better).
6. [**kNN novelty**](/blog/2026/knn/) — a count-based signal that *can't* saturate the way the others did.

Or browse the whole [series](/blog/category/rl-from-scratch/).

### [rl-factory — the engine]({{ site.factory_repo }})

[**rl-factory**]({{ site.factory_repo }}) is a lean, **environment-agnostic** RL training
engine. A single `Trainer` owns only the shared concerns — the env-step budget, checkpointing,
and logging — and drives a pluggable `TrainStrategy` (`setup / step / weights / teardown`).
Models register themselves by name, and whether one is on- or off-policy is **data, not code**,
so there's no per-algorithm branching in the loop. The design goal in one line: **adding a new
RL algorithm is registering a learner, not rewriting a training loop.** It knows nothing about
any specific algorithm or environment — those are supplied by a consumer.

### [rl-ablations — the experiments]({{ site.ablations_repo }})

[**rl-ablations**]({{ site.ablations_repo }}) is the *what* to rl-factory's *how*: five
algorithms — **DQN, PPO, ICM, RND, and kNN episodic novelty** — reproduced on **Super Mario
Bros** (World 1-1) and plugged into the engine. The nice part is how little each one costs:
ICM, RND, and kNN all **act with the PPO policy** and differ only in their learner, so a new
model is a small drop-in rather than a new training loop. The write-ups above walk through them
one at a time, with the Mario 1-1 comparison curves.

### why this exists

Years ago, in a self-supervised-learning seminar at Mila, I met a stack of RL papers I wanted to
*build*, not just read. Back then, one paper was about the ceiling — the limits were real and
ordinary ones: a GPU budget, and code I was still growing into.

What's changed is throughput. With the factory carrying the shared work — and an agent helping me
build it — implementing a paper stops being a one-off project and becomes a drop-in. At Mila I
could take one paper to a working implementation; now I can take ten. This series is me going back
to those ideas and building them for real at that new pace — starting with the curiosity work
(ICM, RND) I most wanted to redo.

### a note on how it was built

I built this with an AI agent as **co-builder**, and myself as the **architect**. That split is
the honest description, and it's the point: an agent can write a learner fast, but it can't
*decide the shape*. The one design decision the whole codebase pivots on — on-policy vs.
off-policy as **data, not branching**, the curiosity models reusing the PPO policy, a new
algorithm arriving as a registered learner rather than a new loop — was mine to make and mine to
defend. The architecture is where the understanding lives, and you can read it straight off the
code.

Which is the thing I didn't expect to come away with. The barrier to building something hard
used to be raw implementation skill. Increasingly it's **architectural taste** — knowing what to
build and whether it's right — and that's a barrier far more people can now cross. Learning, and
building, is democratising. This series is one small proof.
