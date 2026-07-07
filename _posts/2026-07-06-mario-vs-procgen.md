---
layout: post
title: "From Mario to Procgen: preprocessing, the emulator tax, and generalization"
date: 2026-07-06 09:00:00
description: "Three concrete reasons I moved the RL work from Super Mario Bros to Procgen — the preprocessing you don't have to write, the environments you can finally parallelize, and the generalization test Mario can't give you."
tags: [procgen, mario, environments, generalization]
categories: rl-from-scratch
related_posts: true
---

After the curiosity posts, I moved the reinforcement-learning work off **Super Mario Bros** and onto
**[Procgen](https://openai.com/index/procgen-benchmark/)**. This tutorial is the *why*, from first
principles — because the **environment** you pick decides both *what you can measure* and *how fast
you can measure it*, often more than the algorithm does. Two goals drove the switch:

1. **Stop being CPU-bound** — a genuine waste in deep RL, and
2. **Be able to test generalization at all.**

If you remember one sentence: **Mario is one hand-built level behind an emulator; Procgen is a factory
of levels running natively in one process — and that difference is preprocessing you don't write,
environments you can actually parallelize, and a generalization test you couldn't run before.**

{% include figure.liquid loading="eager" path="assets/img/blog/envs/what_the_agent_sees.png" class="img-fluid rounded z-depth-1" zoomable=true %}

*What each agent actually sees. Mario's rich 240×256 frame gets reduced to an 84×84 grayscale stack before the network touches it; Procgen's 64×64 frame is fed in as-is.*

Every new term is defined the first time it appears.

---

## Part 0 — What an "environment" gives you

An RL **environment** is the world the agent acts in. Each step it takes an **action** and returns an
**observation** (here: an image of the screen), a **reward**, and whether the episode ended. The agent
never sees the game's internal state — only pixels. So the *shape and cost of those pixels*, and *how
many copies of the world you can run at once*, are first-order design facts. That's what changes
between Mario and Procgen.

---

## Part 1 — Preprocessing: the frames you must fix vs. the frames you don't

**Mario hands you a raw NES frame: `(240, 256, 3)` RGB**, at the console's frame rate. You can't feed
that to a CNN as-is — three problems, three fixes, the standard Atari/DQN pipeline:

- **Too big** → **resize** to `84×84` (less compute per frame).
- **Color is mostly decoration** → **grayscale** (drop 3 channels to 1).
- **A single frame isn't Markov.** From one still image you can't tell if Mario is *rising or falling*
  — velocity is invisible. → **stack the last 4 frames** so the network can infer motion. And **skip
  frames**: repeat each chosen action for 4 game frames (cheaper, and one decision then maps to one
  meaningful change).

> **Markov:** a state is Markov if it contains everything needed to predict the future — no memory of
> the past required. One Mario frame isn't (no velocity); four stacked frames approximately are.

So before the agent learns anything, you write and tune **four wrappers** (resize, grayscale, stack,
skip) that turn `(240,256,3)` into `(4, 84, 84)`.

**Procgen hands you `(64, 64, 3)` RGB — and you feed it straight to the CNN.** No resize (already
small), no grayscale (color is *semantic* here: keys, walls, and enemies are color-coded — graying it
out destroys information), no stacking and no frame-skip (the games are designed to be ~fully
observable from a single frame). The preprocessing you skip isn't laziness — it's the *standard*, so
your results stay comparable to the published Procgen numbers.

| | **Mario** | **Procgen** |
|---|---|---|
| Raw observation | `(240, 256, 3)` RGB | `(64, 64, 3)` RGB |
| Resize | needed → `84×84` | none (already small) |
| Grayscale | yes (color irrelevant) | **no** (color is semantic) |
| Frame stacking | 4 (to see velocity) | **none** (near-Markov single frame) |
| Frame-skip | 4 | none |
| What the net sees | `(4, 84, 84)` | `(3, 64, 64)` |

---

## Part 2 — The emulator tax: why I could run 8 Mario envs but 64 Procgen envs

RL is data-hungry, so you run **many copies of the environment in parallel** (a **vectorized
environment**) to feed the network a big batch each step. *How* you parallelize depends entirely on
what the environment *is*.

**Mario runs on an NES emulator.** Under the hood, `gym-super-mario-bros` drives `nes-py`, a program
that **simulates the console** — a CPU-bound chunk of work, one instance per environment. To run many
in parallel, "parallel" has to mean **N separate operating-system processes** (a *subprocess*
vectorized env): each environment is its own process, and on every step you **pickle** the observation
(a whole image) and push it through an OS **pipe** back to the learner.

> **IPC (inter-process communication):** moving data between separate processes. Pickling an image and
> sending it over a pipe every step is pure overhead — it does no learning, it just *moves bytes*.

That overhead is brutal: on my M1, roughly **two-thirds of the CPU was spent in the kernel** shuffling
observations between processes, not stepping games or training. It also caps how many envs you can
afford. **I ran 8.**

**Procgen is a native C++ environment that batches internally.** `ProcgenGym3Env(num=64)` steps **64
levels inside one process**, writing all 64 observations straight into a single array — no per-env
process, no pickling, no pipes. One call in, a batch out. I jumped straight to **64 environments**
feeding one batched GPU forward pass, and throughput **roughly tripled**.

Why care? Deep RL alternates two phases: **step the environment (CPU)** and **run the network (GPU)**.
If env-stepping is slow, your expensive GPU **sits idle waiting for data** — you're *CPU-bound*, paying
for a GPU that twiddles its thumbs. The whole point of modern RL infra is to keep the GPU fed; an
emulator-one-process-per-env starves it, a native batched env feeds it. **Changing the environment did
more for my throughput than any code optimization could.**

---

## Part 3 — Train and test: the question Mario can't answer

Here's the scientific reason, and it's the big one.

**Mario 1-1 is a single, fixed, hand-authored level.** You train on it and evaluate on it — the *same*
level. So when the agent finally clears it, you genuinely **cannot tell** whether it *learned to play*
or *memorized this exact layout*. There is no held-out level to check against. (Mario has a
"random-stages" mode, but it's a small hand-made pool, not a real protocol.)

**Procgen procedurally generates its levels from a seed.** So you can train on a fixed set — say **200
levels** — and then **test on levels the agent has never seen** (the rest of an effectively infinite
distribution).

{% include figure.liquid loading="eager" path="assets/img/blog/envs/procgen_levels_grid.png" class="img-fluid rounded z-depth-1" zoomable=true %}

*Nine CoinRun levels from nine different seeds — same rules, different layouts. Train on some, test on the rest, and the score difference tells you how much the agent actually learned versus memorized.*

> **generalization gap:** *train score − test score*. How much of the performance is transferable skill
> versus memorization of the specific training levels.

That gap is a single, honest number, and Mario simply can't produce it. Measuring generalization is the
*entire reason* Procgen exists ([Cobbe et al., 2020](https://arxiv.org/abs/1912.01588)). It's what
turns "my agent got good at the game" into "my agent learned something that *transfers*" — which is the
question the rest of this project is actually about.

---

## So: two goals, one environment change

- **Not CPU-bound** — native in-process batching took me from 8 → 64 environments and kept the GPU fed
  instead of idling behind an emulator.
- **Test generalization** — a train/test split over procedurally-generated levels gives a real
  generalization gap, which a single fixed level never could.

There was one catch: none of this mattered until Procgen actually *built* on my Apple-Silicon
laptop — which, out of the box, it didn't. Getting past that was its own small adventure in refusing to
route around a broken build. That's the next post: [the last mile of fixing Procgen on Apple
Silicon]({% post_url 2026-07-07-procgen-last-mile %}).
