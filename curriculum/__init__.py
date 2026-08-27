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

import math
import random


class Scheduler:
    """Part 1: every task is safe to train on."""

    def __init__(self, tasks, budget, cadence, group, seed):
        """tasks: list of {"task_id": str, "skills": {skill_name: float}}.
        budget: total number of rollouts this run may spend.
        cadence: the simulator applies model updates every `cadence` rollouts.
        group: rollouts per group. Read gymsim to see what a group is for.
        seed: an integer seed for any randomness the scheduler itself uses.
        """
        if not tasks:
            raise ValueError("Scheduler requires at least one task")
        if int(group) <= 0:
            raise ValueError("group must be positive")

        self.group = int(group)
        self.budget = int(budget)
        self.cadence = int(cadence)
        self._rng = random.Random(seed)

        self._task_ids = [task["task_id"] for task in tasks]
        self._shares = {}
        all_skills = set()
        for task in tasks:
            weights = {skill: float(weight)
                       for skill, weight in task["skills"].items()
                       if float(weight) > 0.0}
            total = sum(weights.values()) or 1.0
            self._shares[task["task_id"]] = {
                skill: weight / total for skill, weight in weights.items()
            }
            all_skills.update(weights)

        self._skills = sorted(all_skills)
        self._skill_credit = {skill: 0.0 for skill in self._skills}

        # A shuffled first pass prevents bank order from coupling exploration
        # to cadence boundaries, while remaining exactly reproducible.
        self._explore = list(self._task_ids)
        self._rng.shuffle(self._explore)
        self._explore_at = 0

        # Discounted pseudo-counts estimate each task's *current* pass rate.
        # They intentionally forget old outcomes as the model improves.
        self._stats = {
            task_id: {"a": 1.0, "b": 1.0, "groups": 0, "last": -1}
            for task_id in self._task_ids
        }
        self._groups_seen = 0
        self._active = None
        self._active_n = 0
        self._active_passes = 0

    def choose(self):
        """Return the task_id (str) to sample the next rollout on."""
        if self._active is None:
            if self._explore_at < len(self._explore):
                self._active = self._explore[self._explore_at]
                self._explore_at += 1
            else:
                self._active = max(self._task_ids, key=self._score)
            self._active_n = 0
            self._active_passes = 0
        return self._active

    def observe(self, task_id, passed):
        """Called after every rollout with its pass/fail outcome (bool)."""
        if task_id != self._active:
            raise ValueError("observed a task other than the active task")

        self._active_n += 1
        self._active_passes += int(bool(passed))
        if self._active_n < self.group:
            return

        task_id = self._active
        stat = self._stats[task_id]

        # Roughly three recent groups dominate the estimate. A stationary
        # all-pass task is retired quickly; an old hard task can be rediscovered
        # after shared-skill training moves it into range.
        discount = 0.65
        stat["a"] = discount * stat["a"] + self._active_passes
        stat["b"] = discount * stat["b"] + self.group - self._active_passes
        stat["groups"] += 1
        stat["last"] = self._groups_seen

        spread = (self._active_passes *
                  (self.group - self._active_passes) /
                  float(self.group * self.group))
        for skill, share in self._shares[task_id].items():
            self._skill_credit[skill] += spread * share

        self._groups_seen += 1
        self._active = None

    def _score(self, task_id):
        """Estimate the value of spending the next complete group here."""
        stat = self._stats[task_id]
        a, b = stat["a"], stat["b"]

        # E[p(1-p)] under Beta(a, b), adjusted because observed group spread
        # uses k(n-k)/n^2 rather than population variance.
        expected_spread = ((self.group - 1.0) / self.group) * (
            a * b / ((a + b) * (a + b + 1.0))
        )

        idle = self._groups_seen - stat["last"]
        uncertainty = 0.018 / math.sqrt(stat["groups"] + 1.0)
        stale_bonus = 0.055 * min(idle / 120.0, 1.0)

        # Equalise inferred useful credit across skills. The smoothing keeps
        # this a preference, not a hard quota, so a weak task is never selected
        # solely to make an accounting counter look balanced.
        if self._skills:
            mean_credit = (sum(self._skill_credit.values()) /
                           len(self._skill_credit))
            balance = 0.0
            for skill, share in self._shares[task_id].items():
                balance += share * math.sqrt(
                    (mean_credit + 0.35) /
                    (self._skill_credit[skill] + 0.35)
                )
            balance = min(1.35, max(0.75, balance))
        else:
            balance = 1.0

        return (expected_spread + uncertainty + stale_bonus) * balance


class PoisonScheduler(Scheduler):
    """Part 2: some tasks may land their credit with the sign reversed, and
    nothing marks them. Same interface, same budget.

    This one is run BOTH on banks that contain such tasks and on banks that do
    not, and it is not told which. Both count toward its score.

    Inheriting from Scheduler is a convenience, not a requirement -- override
    as much or as little as you want.
    """

    def __init__(self, tasks, budget, cadence, group, seed):
        super().__init__(tasks, budget, cadence, group, seed)
        self._history = {task_id: [] for task_id in self._task_ids}
        self._risk = {task_id: 0.0 for task_id in self._task_ids}
        self._risk_checked_at = {task_id: 0 for task_id in self._task_ids}
        self._last_choice = None
        self._choice_run = 0

    def choose(self):
        """Choose a frontier task, with exposure and deterioration guards."""
        if self._active is not None:
            return self._active

        if self._explore_at < len(self._explore):
            task_id = self._explore[self._explore_at]
            self._explore_at += 1
        else:
            task_id = max(self._task_ids, key=self._poison_score)

        if task_id == self._last_choice:
            self._choice_run += 1
        else:
            self._last_choice = task_id
            self._choice_run = 1
        self._active = task_id
        self._active_n = 0
        self._active_passes = 0
        return task_id

    def observe(self, task_id, passed):
        closing = task_id == self._active and self._active_n + 1 == self.group
        group_passes = self._active_passes + int(bool(passed))
        super().observe(task_id, passed)
        if closing:
            self._record_poison_evidence(task_id, group_passes)

    def _poison_score(self, task_id):
        score = self._score(task_id)

        # A negative task can look like an excellent frontier task while it
        # drives a skill downward. Diversification bounds the damage before
        # the outcome trend contains enough evidence to identify it.
        if task_id == self._last_choice and self._choice_run >= 3:
            score *= 0.45

        # Risk is deliberately soft: clean banks must not lose a useful task
        # forever because two small groups happened to arrive in a bad order.
        return score * math.exp(-5.0 * self._risk[task_id])

    def _record_poison_evidence(self, task_id, passes):
        history = self._history[task_id]
        history.append(passes)
        if len(history) > 6:
            del history[0]

        groups = self._stats[task_id]["groups"]
        if len(history) < 4 or groups - self._risk_checked_at[task_id] < 2:
            return
        self._risk_checked_at[task_id] = groups

        before = history[-4:-2]
        after = history[-2:]
        before_passes = sum(before)
        after_passes = sum(after)
        n = 2.0 * self.group
        pooled = (before_passes + after_passes + 1.0) / (2.0 * n + 2.0)
        stderr = math.sqrt(max(1e-9, 2.0 * pooled * (1.0 - pooled) / n))
        z = ((before_passes - after_passes) / n) / stderr

        risk = self._risk[task_id]
        if z > 0.8:
            risk += 0.18 * (z - 0.6)
        elif z < -0.5:
            risk -= 0.08 * (-z - 0.3)
        else:
            risk *= 0.96
        self._risk[task_id] = min(1.0, max(0.0, risk))
