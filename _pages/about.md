---
layout: about
title: about
permalink: /
subtitle: Reinforcement learning · self-supervised representation learning · transformers.

profile:
  align: right
  image: # no static photo — the Mario clip below stands in for it
  image_circular: false
  more_info: >
    <img src="/assets/img/mario/ppo.gif" alt="A PPO agent playing Super Mario Bros World 1-1"
         style="width:100%; border-radius:8px;" />
    <p style="text-align:center; font-size:0.8em; margin-top:0.4em;">A PPO agent I trained, playing Mario 1-1</p>

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

**I build agentic-AI systems professionally; on my own time I implement reinforcement
learning from scratch to understand *why* it works — then write up what I learn.**

Most of what's here started as a single confused question ("is RND an extension of
ICM?") and turned into a chain of implemented models, matched experiments, and the
occasional honest negative result.

The through-line is **exploration and representation**: teaching an agent to be curious
(ICM, RND, episodic novelty), then asking the deeper question of *what space* novelty
should even be measured in — which leads into self-supervised trajectory models and
transformers that learn the environment's structure on purpose.

The [blog](/blog/) is the main event: tutorial-style write-ups, each aiming to be
readable by someone one course into deep learning, with every term defined the first
time it appears.

### Start here — the RL-from-scratch series

The current series builds a Mario agent from scratch, each model earning its place by
fixing a limitation of the last, all sharing one small framework:

1. [**rl-factory**](/blog/2026/rl-factory/) — the engine: where adding an algorithm is registering a learner, not rewriting a training loop.
2. [**DQN**](/blog/2026/dqn/) — off-policy value learning; beat the random floor.
3. [**PPO**](/blog/2026/ppo/) — on-policy policy gradients; a more stable learner.
4. [**ICM**](/blog/2026/icm/) — curiosity from forward-prediction error, so Mario explores when reward is sparse.
5. [**RND**](/blog/2026/rnd/) — a simpler curiosity signal (and why simpler was better).
6. [**kNN novelty**](/blog/2026/knn/) — a count-based signal that *can't* saturate the way the others did.

Or browse the whole [series](/blog/category/rl-from-scratch/).

The code lives in two repos: [**rl-factory**]({{ site.factory_repo }}) — the
environment-agnostic training engine — and [**rl-ablations**]({{ site.ablations_repo }}) —
the five models (DQN, PPO, ICM, RND, kNN) that plug into it.

