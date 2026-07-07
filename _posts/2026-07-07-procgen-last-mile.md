---
layout: post
title: "The last mile was a compiler flag: getting Procgen to build on Apple Silicon"
date: 2026-07-07 09:00:00
description: "Why I moved the RL work from Mario to Procgen — and why the first step was fixing a four-year-old build bug in the benchmark itself, then upstreaming the patch."
tags: [procgen, generalization, open-source, apple-silicon]
categories: rl-from-scratch
related_posts: true
---

The benchmark everyone recommends wouldn't install on my laptop. This is the short story of why
I switched the RL work from Super Mario Bros to [Procgen](https://github.com/openai/procgen), what
stood in the way, and the *last mile* it took to get past it — the part where you either fix the
tool for real and give the fix back, or quietly work around it and move on.

I fixed it. And to be upfront about how I work now: I did it **pairing with an AI coding agent**
(Claude Code) — I drove the diagnosis and the decisions, it accelerated the grind. That collaboration
*is* part of the story.

---

## Why leave Mario at all?

Short version: Mario 1-1 is a **single hand-built level**, so you can't separate *learning* from
*memorizing*, and it runs on an emulator that caps how many environments you can parallelize.
[Procgen](https://arxiv.org/abs/1912.01588) fixes both — procedurally-generated levels with a real
train/test split, batched natively in one process. I unpack all of that in the previous post,
[From Mario to Procgen]({% post_url 2026-07-06-mario-vs-procgen %}). This post is about the wall I hit
*getting there*.

## The wall

There's no prebuilt Procgen wheel for Apple Silicon, so `pip` tries to build from source — and on an
M1 that build simply fails. Three separate root causes, none of them obvious from the error spew:

1. **An x86-only compiler flag.** Procgen's `CMakeLists.txt` compiles packaged builds with
   `-march=ivybridge` — an *Intel* microarchitecture. arm64 Clang rejects it outright. (The fix: on
   `arm64`/`aarch64`, use the base `armv8-a` ISA instead.)
2. **Qt looked for at the wrong address.** The build probes one hardcoded path for Qt —
   `/usr/local/opt/qt5`, the *Intel* Homebrew location. Apple Silicon Homebrew installs under
   `/opt/homebrew`, so CMake never finds it. (The fix: also probe `/opt/homebrew/opt/qt@5/lib/cmake`.)
3. **Warnings that became errors.** Modern AppleClang (21) promotes two "set-but-unused variable"
   warnings in `starpilot.cpp` and `chaser.cpp` to hard errors under `-Werror`. (The fix: mark the two
   locals `[[maybe_unused]]` — zero change to any game logic.)

None of these is deep. All three are the kind of paper cut that makes a person give up: buy a cloud
box, switch benchmarks, or paste a hack into their local copy and never speak of it again.

## The last mile

Here's the part I actually care about. A local hack would have unblocked *me* in ten minutes. Instead:

- I traced each failure to its **root cause** rather than silencing symptoms.
- The fix is **four files, +15/−5 lines, behavior-preserving** — the smallest change that restores the
  source build without touching a single game's determinism.
- I **upstreamed it** as a pull request to `openai/procgen` — and it turned out to close
  [issue #69, "Apple Silicon Support,"](https://github.com/openai/procgen/issues/69) which had been
  open since **2021** with a string of people hitting the same wall. So the fix doesn't just help me;
  it helps everyone who tries Procgen on a Mac next.
- Then I **proved it holds under load** — not a 50-step smoke test, but a full **100M-step** PPO run
  (IMPALA-CNN, 64 environments batched natively, training on Metal/MPS) plus an off-policy world-model
  experiment on top. Stable start to finish.

> **PR:** [openai/procgen#107](https://github.com/openai/procgen/pull/107) — *Fix building from source
> on Apple Silicon (arm64) and recent Clang.* My first open-source contribution.

That's the difference between "I tried Procgen and it didn't work" and "Procgen works on Apple Silicon
now, for everyone, because I fixed it." The distance between those two sentences is the last mile.

## The payoff

With the build working, the question that justified the whole detour became answerable. Trained on 200
Miner levels and evaluated on the unseen full distribution:

| | mean return | solve rate |
|---|---|---|
| **Train** (200 seen levels) | 10.63 | 99% |
| **Test** (unseen levels) | 7.78 | 91% |

The agent genuinely learned to play — 91% solved on levels it had *never seen* — but there's a real
**27% generalization gap** it still leaves on the table. That gap is the entire point: a single,
honest number that Mario could never have shown me, and the starting line for the representation and
transfer experiments this project is actually about.

The compiler flag was never the interesting part. Refusing to route around it was.
