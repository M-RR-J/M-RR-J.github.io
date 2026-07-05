# m-rr-j.github.io

Source for **[m-rr-j.github.io](https://m-rr-j.github.io)** — Maxime Jousset's notes on
reinforcement learning, self-supervised representation learning, and transformers, built from
scratch one experiment at a time.

The blog is the main event: tutorial-style write-ups of an **RL-from-scratch** series that trains a
Super Mario Bros agent one algorithm at a time — **DQN → PPO → ICM → RND → kNN novelty** — each
model earning its place by fixing a limitation of the last.

The code the posts describe lives in two companion repos:

- **[rl-factory](https://github.com/M-RR-J/rl-factory)** — the environment-agnostic training engine
  (adding an algorithm is registering a learner, not rewriting a training loop).
- **[rl-ablations](https://github.com/M-RR-J/rl-ablations)** — the five models, on Mario, that plug
  into it.

## Build locally

```bash
bundle install
bundle exec jekyll serve
```

Built with the [al-folio](https://github.com/alshedivat/al-folio) Jekyll theme (MIT); see `LICENSE`.
