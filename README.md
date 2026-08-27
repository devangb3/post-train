# Post-train

*To open a rendered Markdown preview of this README, press Command-Shift-V.
There are formulas further down that will not display without it.*

This assessment is meant to give you a glimpse of the problems the post-training
team at Mercor works on.

One of the big bottlenecks in training models as large as Nemotron Ultra or
Kimi K3 is that every rollout is expensive. That forces us to be deliberate
about which tasks we spend rollouts on and which we leave alone.

What makes it harder is that, as a researcher, you know very little about the
data before the run starts. You do not know how difficult the tasks are. You do
not know how the current model will do on them. Almost all of that arrives
incrementally, as a consequence of the sampling decisions you already made.

Here you will work with a simulated model with fixed update rules, and work out
the most efficient way to sample and process the data so the model learns as
much as possible.

## Task overview

You have a dataset of **120 tasks** whose difficulty distribution you do not
know, and a budget of **5,760 rollouts** — one rollout being a single attempt at
a single task, coming back `pass` or `fail`. That is roughly six epochs,
running GRPO with a group size of 8 over the dataset, and it is your entire
budget for working out what this data is *and* training on it at the same time.

We tell you how the model updates from the pass rates your rollouts get on a
task. We tell you how a task's pass probability is calculated, from the model's
current skill, the skills that task needs, and how much of them it demands.

Your job is to write an algorithm that samples over the tasks effectively, with
one goal: maximise the model's average pass rate at the end of training.

**The model is five numbers** — one proficiency level per skill, between 0 and
1, and nothing else. Training it means moving those five numbers up under a
budget.

You are writing a **scheduler** — code that decides, 5,760 times over, which
task to spend the next rollout on. Nothing else about the training loop is yours
to change: the update rule is given and fixed, and you only choose what to feed
it. Your score is the model's average pass rate at the end, measured on a
separate **evaluation set** you never train on and never see.

While we share the training tasks with you, that evaluation set is drawn from an
**entirely separate distribution of tasks**.

The rest of this document fills in the details. You will find these useful as
you work:

1. [A primer on GRPO](#a-quick-grpo-primer), if you are not already familiar.
2. [How a task's pass probability is calculated](#how-pass-probability-is-calculated).
3. [How the model actually learns](#how-the-model-learns).
4. [How you are evaluated](#how-we-evaluate).
5. [The notation used throughout](#symbols).
6. [How to submit](#submitting).

## Time limit

**75 minutes.** There is more here than fits, on purpose — nobody finishes it,
and you are not expected to. What we are watching is which ideas you get to and
what you decide to skip.

There is a short walkthrough with us afterwards.

## What you write

**Two classes, both stubbed in `curriculum/__init__.py`.** The assessment has
two parts, and they are graded separately:

| class | assume |
| --- | --- |
| `Scheduler` | **Part 1.** Every task is safe to train on. |
| `PoisonScheduler` | **Part 2.** Some tasks may damage the model, and nothing marks them. |

`PoisonScheduler` is run on banks that contain such tasks **and on banks that do
not**, and it is not told which it is in. Both count toward its score.

Both have the same interface:

```python
choose()                  # -> the task_id to spend the next rollout on
observe(task_id, passed)  # -> called after every rollout
```

Share as much code between them as you like — `PoisonScheduler` starts out
subclassing `Scheduler`. **Start with Part 1**; a strong Part 1 and an honest
note about Part 2 beats two rushed halves.

The loop is always `choose`, one rollout, `observe`, repeat. Your scheduler sees
the task bank and the outcomes it observes — not the simulator's internals, its
seed, or the config.

**Everything else already works.** `bin/train` is wired and writes the artifacts
we read:

```bash
./bin/train --config config/run.json --out runs/mine
```

Add `--baseline` to that command to run the shipped uniform baseline instead of
your scheduler.

**It does not print a score.** The evaluation set is not on your machine, and no
run tells you how good the model got. The only feedback a scheduler has is the
pass or fail of the rollouts it spent — which is the position you are in when you
schedule a real training run. You should not spend time on plumbing; if you
restructure things, keep the command flags and artifact filenames as they are,
because we run them directly.

## Rules

- The standard library is enough. If you do want a package, add it to
  `requirements.txt`.
- Randomness is fine — seed it from the `seed` you are given, not from the
  clock or an unseeded global, so the same run reproduces exactly.
- **Use only what is passed to you**: the task list, and the pass or fail of each
  rollout. Do not reach into the simulator's state — no walking the call stack,
  no reading the running `Simulator`, no inspecting our files. Those hold the
  hidden difficulties and the model's true skill, and a scheduler that reads them
  is not solving the problem. **We read every submission, and doing this fails
  the assessment outright** — a working scheduler that scores badly is worth far
  more to us than one that cheats.
- Spend the budget. Do not exceed it.

## Submitting

After you submit you record a short walkthrough: we ask you about what you
built and the calls you made, and you talk us through it. It counts for more
here than on a longer take-home, because 75 minutes is not enough time for the
score alone to say much about how you think — so answer even the parts you did
not get to, and say what you would have done. Rough is fine throughout; we are
after the reasoning.

**Before you submit, run the self-check:**

```bash
./bin/check
```

It confirms your scheduler imports, runs within budget in both rollouts and
runtime, writes the artifacts we read, and reproduces on a second run. It says
nothing about how good your scheduler is — it is there so nobody loses anything
to plumbing. We run the same commands. Then:

```bash
litmus submit
```

---

*Everything above is what you have to do. Everything below is background: how
the simulator works, and how we read a submission. Jump to whichever part of the
list in the overview you need.*

## A quick GRPO primer

*Skip this if you already work with GRPO — nothing here is specific to the
exercise.*

GRPO (Group Relative Policy Optimization) drops the value network and uses the
**group itself as the baseline**. For one task you sample a group of $G$
rollouts, score each, and set each rollout's advantage to its reward minus the
group's mean:

$$A_i \;=\; r_i - \frac{1}{G}\sum_{j=1}^{G} r_j$$

Rollouts that beat their peers get pushed up, those that fall short get pushed
down. Nothing else is needed to decide which way the gradient points.

**The consequence that drives this whole exercise:** if every rollout in a group
earns the same reward, every advantage is exactly zero and the group contributes
no gradient at all — however many rollouts went into it.

Here $G = 8$ and the reward is 1 for a pass, 0 for a fail. With $k$ passes the
group's reward variance is $k(8-k)/64$, which is the `spread` term you will see
below — it is the group's variance, not a separate invention.

An **epoch** is one pass over the dataset: $120 \text{ tasks} \times 8 = 960$
rollouts. The budget of 5,760 is six of them.

## How pass probability is calculated

**A task is a vector of demands.** For each skill it needs, a task wants a
certain level of that skill — `demand[t][s]`, on the same 0-to-1 scale as the
model's own `prof[s]`. A demand is the level at which that component of the task
passes about 95% of the time.

You see each task's **shape**: its demands divided by the largest of them, so the
biggest reads `1.0` and the rest are relative to it. A task shipped as
`{coding: 1.0, math: 0.4}` needs coding fully and math at 40% of that level.
What you do not see is the **scale** — how much "fully" actually is.

**The task is conjunctive.** Every skill it needs must be met, and the components
multiply:

$$P(\text{pass on } t) \;=\; \prod_{s \,\in\, \text{skills}(t)} \sigma\!\big(S \cdot (\text{prof}[s] - \text{demand}[t][s]) + \text{OK}\big)$$

with $\sigma(z) = 1/(1+e^{-z})$ and $\text{OK} = \log(0.95/0.05) = 2.9444$, which
is what makes a demand mean "the 95% level".

**$S = 6.0$, and the graded runs use the same value.** So does the learning
rate below. You know your own update rule and your own learning rate in a real
run; what you do not know is your data. Everything about the task bank — the
difficulties, how they are spread, and which tasks damage the model — is hidden
and is redrawn for grading.

`OK` is what makes a demand mean "the 95% level": set `prof[s] = demand[s]` and
the exponent is just `OK`, so that component passes 19 times in 20.

Which skills a given task is currently waiting on is not something you can see —
you only see it pass or fail.

## How the model learns

**Learning comes from disagreement.** Nothing is learned from a single rollout.
A group of 8 closes with some number of passes — call it `k` — and that is when
the model changes. This is the whole update rule, in three steps.

**One.** How much the group disagreed with itself is its `spread`:

$$\text{spread} \;=\; \frac{k\,(8-k)}{64}$$

where $k$ is the number of passes in the closed group. It is largest at $k=4$
and exactly $0$ at $k=0$ or $k=8$.

**Two.** That spread becomes `credit`, split across the skills the task needs
in proportion to how much it demands of each. The shares sum to 1, so one closed
group is one unit of training however many skills are involved:

$$\text{share}[s] = \frac{\text{shape}[s]}{\sum_{s'} \text{shape}[s']}
\qquad\qquad
\text{credit}[s] = \text{LR} \cdot \text{spread} \cdot \text{share}[s] \cdot 8$$

The $8$ is the group size. $\text{LR} = 0.030$, given, and the same in the
graded runs.

**Three.** The model's skill at `s` is `prof[s]`, a number from 0 to 1. Credit
lands on it at the next cadence boundary:

$$\text{prof}[s] \;\leftarrow\; \text{prof}[s] + \text{credit}[s] \cdot \big(1 - \text{prof}[s]\big)$$

Run it with `--baseline` to see what uniform sampling does: it sweeps the bank
evenly, three groups on everything.

## What makes it hard

**The rules transfer; the data does not.** `gymsim.py` is complete and honest —
every rule is there and you are meant to read it, though not all of it in 75
minutes. The update rule, the slope, the learning rate, the group size, the
cadence, the budget and the definition of a demand are all fixed, and the graded
runs use exactly these values.

What is redrawn for every graded run is everything about the *data*, and the
model's starting point: the starting proficiency of each skill, the tasks
themselves, and how the bank's difficulty is laid out. That mirrors the real
situation — you know your own training loop, you do not know your dataset.

**You can run the simulator as many times as you like**, and if you go looking,
the difficulties behind this practice bank are recoverable. That is deliberate —
you cannot calibrate an estimator against a target you can never see.

## How we evaluate

Your score is the model's **average pass rate on the evaluation set** at the end
of the run.

We also read the code and the run it produced, not just the number at the end.
**What we are mainly interested in is the reasoning.**

You will not get to everything in 75 minutes. Nobody does. Choose, and tell us
what you chose.

## Symbols

| Symbol | What it is |
| --- | --- |
| `k` | passes in a closed group of 8 |
| `shape[s]` | the task's demands relative to its largest, which reads 1.0 (**you see this**) |
| `demand[t][s]` | the level of skill `s` task `t` wants — its 95% level (hidden) |
| `prof[s]` | the model's skill at `s`, from 0 to 1 (hidden) |
| `share[s]` | `shape[s]` as a fraction of the task's total; sums to 1 |
| `S`, `LR` | slope $6.0$ and learning rate $0.030$. **Given, and the same in the graded runs** |
| `OK` | `log(0.95/0.05) = 2.9444`, fixed: it is what makes a demand mean "the 95% level" |

Every formula above is also in `gymsim.py`. The numbers beside them are one
draw and will not be the numbers you are graded on.
