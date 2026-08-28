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
        self._rollouts_seen = 0
        self._cadence_credit = {}
        self._landed_credit = []
        self._task_observation = {task_id: None for task_id in self._task_ids}
        self._causal_sum = {task_id: 0.0 for task_id in self._task_ids}
        self._causal_weight = {task_id: 0.0 for task_id in self._task_ids}
        self._bank_up = 0
        self._bank_down = 0
        self._selection_alarm = 0.35

    def choose(self):
        """Choose a frontier task, with exposure and deterioration guards."""
        if self._active is not None:
            return self._active

        if self._explore_at < len(self._explore):
            task_id = self._explore[self._explore_at]
            self._explore_at += 1
        else:
            self._selection_alarm = self._bank_alarm()
            candidates = self._task_ids
            # A poisoned frontier task can continue to outscore every
            # alternative even after a soft penalty. Force one different
            # group after two repeats so exposure is actually bounded while
            # allowing the frontier task to compete again immediately after.
            if self._choice_run >= 2:
                alternatives = [
                    task_id for task_id in self._task_ids
                    if task_id != self._last_choice
                ]
                if alternatives:
                    candidates = alternatives
            task_id = max(candidates, key=self._poison_score)

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
        self._rollouts_seen += 1
        if closing:
            self._record_poison_evidence(task_id, group_passes)
            spread = (group_passes * (self.group - group_passes) /
                      float(self.group * self.group))
            if spread > 0.0:
                self._cadence_credit[task_id] = (
                    self._cadence_credit.get(task_id, 0.0) + spread
                )
            self._record_cadence_observation(task_id, group_passes)
        if self._rollouts_seen % self.cadence == 0:
            epoch = self._rollouts_seen // self.cadence
            self._landed_credit.append((epoch, self._cadence_credit))
            # Only recent credit can be attributed with useful confidence;
            # bounding this list also keeps per-group work constant over time.
            if len(self._landed_credit) > 4:
                del self._landed_credit[0]
            self._cadence_credit = {}

    def _poison_score(self, task_id):
        score = self._score(task_id)

        # Risk is deliberately soft: clean banks must not lose a useful task
        # forever because two small groups happened to arrive in a bad order.
        risk_weight = 4.0 + 4.0 * self._selection_alarm
        return score * math.exp(-risk_weight * self._risk[task_id])

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

    def _record_cadence_observation(self, task_id, passes):
        # Label outcomes by the epoch in which the group started. A group that
        # closes exactly on an update boundary contains pre-update outcomes and
        # must not be mistaken for evidence about the update it just caused.
        epoch = max(0, (self._rollouts_seen - self.group) // self.cadence)
        previous = self._task_observation[task_id]
        self._task_observation[task_id] = (epoch, passes)
        if previous is None or previous[0] >= epoch:
            return

        exposures = {}
        for landed_epoch, sources in self._landed_credit:
            if previous[0] < landed_epoch <= epoch:
                for source, spread in sources.items():
                    overlap = sum(
                        min(weight, self._shares[source].get(skill, 0.0))
                        for skill, weight in self._shares[task_id].items()
                    )
                    if overlap > 0.0:
                        exposures[source] = (exposures.get(source, 0.0) +
                                             spread * overlap)
        total = sum(exposures.values())
        if total <= 0.0:
            return

        delta = (passes - previous[1]) / float(self.group)
        if delta >= 1.0 / self.group:
            self._bank_up += 1
        elif delta <= -1.0 / self.group:
            self._bank_down += 1

        for source, exposure in exposures.items():
            weight = exposure / total
            self._causal_sum[source] += delta * weight
            self._causal_weight[source] += weight

            # Cross-task declines are weaker evidence than a task's own trend,
            # but they are the only way to notice damage to related tasks.
            risk = self._risk[source]
            if delta < -0.125:
                risk += 0.32 * (-delta - 0.05) * weight
            elif delta > 0.125:
                risk -= 0.12 * (delta - 0.05) * weight
            self._risk[source] = min(1.0, max(0.0, risk))

    def _bank_alarm(self):
        """Estimate whether diversification is worth its clean-bank cost."""
        comparisons = self._bank_up + self._bank_down
        if comparisons < 12:
            return 0.35

        decline_rate = (self._bank_down + 2.0) / (comparisons + 4.0)
        global_alarm = min(1.0, max(0.0,
            (decline_rate - 0.42) / 0.25
        ))

        measured = 0
        negative = 0
        for task_id in self._task_ids:
            weight = self._causal_weight[task_id]
            if weight >= 0.75:
                measured += 1
                if self._causal_sum[task_id] / weight < -0.06:
                    negative += 1
        if measured >= 6:
            prevalence = negative / float(measured)
            prevalence_alarm = min(1.0, max(0.0,
                (prevalence - 0.15) / 0.35
            ))
            return 0.65 * global_alarm + 0.35 * prevalence_alarm
        return global_alarm
