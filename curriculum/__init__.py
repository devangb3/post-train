"""curriculum: the schedulers under test.

This module must stay importable as `curriculum` from the repository root, with
both classes below, however you organize the rest of your code.

There are two, because the assessment has two parts:

    Scheduler        Part 1. Assume no task damages the model.
    PoisonScheduler  Part 2. Assume some tasks do, and you do not know which.

They are graded separately, each in the world it was written for. Sharing code
between them is expected -- subclass, compose, whatever you like. What we care
about is what the second one does differently, and what that difference costs.
"""


class Scheduler:
    """Part 1: every task is safe to train on."""

    def __init__(self, tasks, budget, cadence, group, seed):
        """tasks: list of {"task_id": str, "skills": {skill_name: float}}.
        budget: total number of rollouts this run may spend.
        cadence: the simulator applies model updates every `cadence` rollouts.
        group: rollouts per group. Read gymsim to see what a group is for.
        seed: an integer seed for any randomness the scheduler itself uses.
        """
        raise NotImplementedError

    def choose(self):
        """Return the task_id (str) to sample the next rollout on."""
        raise NotImplementedError

    def observe(self, task_id, passed):
        """Called after every rollout with its pass/fail outcome (bool)."""
        raise NotImplementedError


class PoisonScheduler(Scheduler):
    """Part 2: some tasks may land their credit with the sign reversed, and
    nothing marks them. Same interface, same budget.

    This one is run BOTH on banks that contain such tasks and on banks that do
    not, and it is not told which. Both count toward its score.

    Inheriting from Scheduler is a convenience, not a requirement -- override
    as much or as little as you want.
    """

    def __init__(self, tasks, budget, cadence, group, seed):
        raise NotImplementedError
