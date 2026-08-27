"""gymsim: a seeded, fully offline GRPO training simulator.

The simulator holds a hidden difficulty per task and an evolving hidden
proficiency per skill. It exposes one operation: spend a rollout, see whether
it passed. Model state advances at a fixed update cadence. Every rollout served
is appended to a rollout meter log.

PROFICIENCY IS BOUNDED. `prof[s]` lives in [0, 1]: 0 is no ability at all and 1
is perfect. Gains are damped by `(1 - prof)`, so improvement slows as a skill
approaches mastery and stops at it.

A TASK IS A VECTOR OF DEMANDS. `demand[t][s]` is the level of skill `s` that task
`t` wants, on that same 0-to-1 scale, and it is the level at which that component
of the task passes about 95% of the time. A task is CONJUNCTIVE: every skill it
needs must be met, and the outcome is the product over them,

    P(pass) = product over s of  sigmoid( SLOPE * (prof[s] - demand[t][s]) + OK )

so shortfalls compound -- being a little short on three skills is worse than
being a little short on one -- and no strength anywhere compensates for a skill
you do not have. Keep training and any finite demand is eventually met.

What a candidate sees is the task's SHAPE: the demands divided by the largest of
them, so the biggest reads 1.0 and the rest are relative to it. The overall
scale is hidden. That is the same split as before -- which skills a task needs,
but not how hard it is -- with no normalising to do.

Learning is GROUP-based, and this is the whole point of the environment.
Rollouts on a task accumulate into a pending group; when the group closes at
`group` rollouts, the update is derived from the OUTCOMES THAT ACTUALLY CAME
BACK, not from the task's hidden pass probability. A group that comes back
all-pass or all-fail carries no spread, so it produces no gradient and the
budget spent on it bought nothing. That is the whole reason to sample
adaptively: you cannot tell in advance which groups will be informative, and a
task whose groups keep closing flat is a task to abandon.

A group closes at `group` rollouts. One left open when the budget runs out is
flushed and banked on whatever it holds, rather than silently discarded -- with
the same spread rule applied to its actual size, so a group of one has no spread
by construction and teaches nothing. `group_stats` reports how much of the
budget ended up in groups that never filled.

Everything is deterministic given the config (seed included). There is no
wall-clock and no global randomness anywhere in this module.

Typical use:

    import gymsim
    cfg = gymsim.load_config("config/run.json")
    sim = gymsim.Simulator(cfg, out_dir="runs/demo")
    tasks = sim.tasks                 # public view: task_id + skills only
    while sim.rollouts_used < sim.budget:
        passed = sim.rollout(tasks[0]["task_id"])
    sim.close()                       # flush open groups and banked credit
"""

import hashlib
import json
import math
import os

# Learning dynamics. These are part of the simulator, not tunables.
# The learning rate and the slope ARE given, and the graded runs use these same
# values. In a real post-training run you know your own update rule and your own
# learning rate; what you do not know is your data. So these are disclosed and
# everything about the task bank is not.
_LR = 0.030            # per-rollout learning-rate scale
_SLOPE = 6.0           # how sharply one component turns with its shortfall
_OK = 2.9444           # log(.95/.05): at prof == demand a component passes 95%
# Starting proficiency is drawn per skill from the run salt, so it differs from
# run to run and is not recoverable from this file.
_PROF0_LO, _PROF0_HI = 0.22, 0.38

# How the bank's difficulty is laid out, and where the evaluation probes sit, are
# BOTH drawn per run from the salt. There is no fixed split to read out of this
# file and no fixed exam difficulty: what proportion of a bank is already solved,
# what proportion is out of reach, and how hard the probes are all move from run
# to run. You are told the form, not the numbers.
_N_ZONES = 4
_ZONE_LO, _ZONE_HI = 0.0, 2.6              # the scale range zones are cut from
_BAND_LO, _BAND_HI = 0.45, 1.30            # evaluation probes land in here
_BAND_SPREAD = 0.08
_ZONES = None                              # per-run override; None = derive from salt


def _zones_for(salt):
    """A difficulty layout for this run.

    The four zones keep their meaning -- already solved, live now, comes into
    range later, never reachable -- because otherwise a run is not a scheduling
    problem at all. What moves is where the boundaries fall and how much of the
    bank sits in each, so there is no fixed split to read out of this file and
    nothing memorised from one bank transfers to the next.
    """
    edges = [a + j * (_unit(salt, "zoneedge", i) - 0.5)
             for i, (a, j) in enumerate(((0.55, 0.24), (0.95, 0.24), (1.32, 0.24)))]
    edges = sorted(edges)
    bounds = [_ZONE_LO] + edges + [_ZONE_HI]
    raw = [a + 0.16 * (_unit(salt, "zonefrac", i) - 0.5)
           for i, a in enumerate((0.10, 0.12, 0.40, 0.38))]
    raw = [max(0.04, r) for r in raw]
    tot = sum(raw)
    return tuple(("z%d" % i, raw[i] / tot, (bounds[i], bounds[i + 1]))
                 for i in range(_N_ZONES))


def _bands_for(salt):
    """Where this run's evaluation probes sit. Not disclosed, and not fixed."""
    lo = _BAND_LO + 0.15 * _unit(salt, "bandlo")
    hi = _BAND_HI - 0.15 * _unit(salt, "bandhi")
    mid = lo + (hi - lo) * (0.4 + 0.2 * _unit(salt, "bandmid"))
    return {"easy": lo, "frontier": mid, "hard": hi}

DEFAULT_GROUP = 8      # rollouts per group when the config does not say


class BudgetExceeded(Exception):
    """Raised when a rollout is requested past the hard training budget."""


def _unit(*parts):
    """Deterministic hash -> float in [0, 1)."""
    h = hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


def _sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _pass_prob(prof, demand, slope):
    """Conjunctive: every skill the task needs must be met, and the components
    multiply. A component is met when `prof[s]` reaches `demand[s]`, which is
    defined as the level where that component passes ~95% of the time.

    Because the factors multiply, shortfalls compound: being 0.2 short on two
    skills costs more than being 0.2 short on one, and a skill you simply do not
    have drags the whole product down however strong the others are. Nothing
    substitutes for anything. And because every factor rises toward 1 as its
    skill improves, any finite demand is eventually met -- keep training a task
    and it does become routine.
    """
    q = 1.0
    for s, d in demand.items():
        q *= _sigmoid(slope * (prof[s] - d) + _OK)
    return q


# SOME TASKS TEACH THE WRONG THING. Their groups produce credit exactly as any
# other group does, and then it lands with the sign reversed: training on one of
# these makes the model worse at the skills the task needs. Nothing in the task
# marks it and the outcome of a rollout looks no different.
#
# HOW MANY there are is drawn per run and not disclosed. The range below is wide
# on purpose and includes zero: some runs have no bad tasks at all. So the first
# question is not "which tasks are bad" but "does this bank have a problem worth
# spending budget on" -- and the only way to answer it is to look.
_POISON_LO, _POISON_HI = 0.0, 0.40
_POISON_RATE = None                        # per-run override; None = derive from salt


def _poison_frac(salt):
    if _POISON_RATE is not None:
        return _POISON_RATE
    return _POISON_LO + (_POISON_HI - _POISON_LO) * _unit(salt, "poisonrate")


def _is_poison(salt, tid):
    return _unit(salt, "poison", tid) < _poison_frac(salt)


def _shape(demand):
    """The public view of a task: demands relative to the largest one."""
    top = max(demand.values())
    return {s: round(d / top, 3) for s, d in sorted(demand.items())}


def _share(shape):
    """How a group's credit splits across the skills a task needs. Sums to 1, so
    one closed group is one unit of training however many skills are involved --
    under L2 weights a two-skill task quietly banked up to 29% more."""
    tot = sum(shape.values()) or 1.0
    return {s: w / tot for s, w in shape.items()}


def _group_signal(passes, size):
    """Learning available from a closed group of `size` rollouts.

    The sample variance of the observed outcomes, normalised to [0, 0.25].
    Exactly zero when every rollout in the group agreed, which is the property
    the whole environment turns on: a flat group is a group that taught the
    model nothing, however promising the task looked.
    """
    if size <= 1:
        return 0.0
    return passes * (size - passes) / float(size * size)


def _train_scale(salt, task_ids):
    """Assign each task its hidden scale by zone. Deterministic in the salt, and exact
    in its proportions -- the zone split is sliced, not sampled, so a bank
    always has the intended shape rather than one draw from it."""
    order = sorted(task_ids, key=lambda t: _unit(salt, "zone", t))
    out, i, n = {}, 0, len(order)
    bounds = []
    acc = 0.0
    for name, frac, rng in (_ZONES or _zones_for(salt)):
        acc += frac
        bounds.append((name, int(round(acc * n)), rng))
    for name, upto, (lo, hi) in bounds:
        while i < min(upto, n):
            tid = order[i]
            out[tid] = lo + (hi - lo) * _unit(salt, "difficulty", tid)
            i += 1
    while i < n:                       # rounding tail
        tid = order[i]
        lo, hi = (_ZONES or _zones_for(salt))[-1][2]
        out[tid] = lo + (hi - lo) * _unit(salt, "difficulty", tid)
        i += 1
    return out


def _eval_scale(salt, task):
    bands = _bands_for(salt)
    base = bands.get(task.get("band"), bands["frontier"])
    return base + _BAND_SPREAD * _unit(salt, "evaldiff", task["task_id"])


def load_config(path):
    """Load a run config. Relative paths are resolved against the config file."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    for key in ("task_bank", "eval_bank"):
        if key in cfg and not os.path.isabs(cfg[key]):
            cfg[key] = os.path.join(base, cfg[key])
    return cfg


def load_tasks(path):
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def public_tasks(tasks):
    """The candidate-visible view of a task bank: task_id and skills only."""
    return [{"task_id": t["task_id"], "skills": dict(t["skills"])} for t in tasks]


class Simulator:
    def __init__(self, config, out_dir=None):
        self.config = config
        self.seed = int(config["seed"])
        self.budget = int(config["budget"])
        self.cadence = int(config["cadence"])
        self.group = int(config.get("group", DEFAULT_GROUP))
        self.skills = list(config["skills"])
        self.rollouts_used = 0
        self.round = 0

        # Hidden difficulty derives from `difficulty_salt`, not from the run
        # seed. Both are in the practice config, so a practice run's
        # difficulties ARE recoverable from this file - deliberately, because
        # you cannot calibrate an estimator against a target you cannot see.
        # What does not transfer is the score: every graded run passes a salt
        # of its own that is never shipped, and salts are hashed independently,
        # so a difficulty memorised here tells you nothing there.
        salt = config.get("difficulty_salt", self.seed)

        raw = load_tasks(config["task_bank"])
        self._task_ids = [t["task_id"] for t in raw]
        # the shipped "skills" field IS the shape: demands relative to the
        # largest, so the biggest reads 1.0. Nothing to normalise.
        self._public = {t["task_id"]: dict(t["skills"]) for t in raw}
        scales = _train_scale(salt, self._task_ids)
        # A task's demands are fixed for the whole run. What changes is the
        # model.
        self._scale = {t["task_id"]: (float(t["scale"]) if "scale" in t
                                      else scales[t["task_id"]]) for t in raw}
        self._demand = {tid: {s: self._scale[tid] * w
                              for s, w in self._public[tid].items()}
                        for tid in self._task_ids}

        # The evaluation set is grader-side. A run without it trains normally and
        # simply cannot be scored, which is the candidate's situation: the only
        # feedback a scheduler gets is the pass or fail of rollouts it spent.
        self._eval = []
        for t in load_tasks(config["eval_bank"]) if config.get("eval_bank") else []:
            sc = _eval_scale(salt, t)
            self._eval.append(
                (t["task_id"], {s: sc * w for s, w in t["skills"].items()})
            )

        # Drawn per run, like the difficulties and the starting proficiency.
        self.lr = _LR
        self.slope = _SLOPE
        self._prof = {
            s: _PROF0_LO + (_PROF0_HI - _PROF0_LO) * _unit(salt, "prof0", s)
            for s in self.skills
        }
        self._credit = {s: 0.0 for s in self.skills}
        self._poison = {t: _is_poison(salt, t) for t in self._task_ids}
        # Open groups, per task: [rollouts_so_far, passes_so_far].
        self._pending = {tid: [0, 0] for tid in self._task_ids}
        # Group accounting, readable via `group_stats`.
        self._closed_rollouts = 0   # landed in a group that reached full size
        self._flat_rollouts = 0     # landed in a group that closed with no spread
        self._partial_rollouts = 0  # landed in a group that never reached full size
        self._groups_closed = 0     # groups that reached full size
        self._groups_informative = 0  # ... of those, ones that carried any spread

        self._out_dir = out_dir
        meter_path = os.environ.get("GYMSIM_METER")
        if meter_path is None and out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            meter_path = os.path.join(out_dir, "meter.jsonl")
        # Truncating, not appending. Appending meant a second run into the same
        # directory produced a meter with both runs in it, so a budget check
        # read double the rollouts actually served and a compliant run looked
        # twice over cap.
        self._meter = open(meter_path, "w", encoding="utf-8") if meter_path else None

    @property
    def tasks(self):
        """The public view: each task's shape, exactly as the bank file gives it.
        The largest demand reads 1.0 and the rest are relative to it; the scale
        that turns a shape into actual demands is hidden."""
        return [
            {"task_id": tid, "skills": dict(self._public[tid])}
            for tid in self._task_ids
        ]

    def _pass_prob(self, tid):
        return _pass_prob(self._prof, self._demand[tid], self.slope)

    def _meter_write(self, obj):
        if self._meter is not None:
            self._meter.write(json.dumps(obj, sort_keys=True) + "\n")
            self._meter.flush()

    def rollout(self, task_id):
        """Serve one rollout on task_id. Returns True on pass, False on fail."""
        if task_id not in self._demand:
            raise KeyError("unknown task_id: %r" % (task_id,))
        if self.rollouts_used >= self.budget:
            raise BudgetExceeded("budget of %d rollouts exhausted" % self.budget)
        q = self._pass_prob(task_id)
        passed = _unit(self.seed, "outcome", self.rollouts_used, task_id) < q


        pend = self._pending[task_id]
        pend[0] += 1
        pend[1] += 1 if passed else 0

        self._meter_write(
            {"n": self.rollouts_used, "task_id": task_id, "passed": bool(passed)}
        )
        self.rollouts_used += 1

        if pend[0] >= self.group:
            self._close_group(task_id)
        if self.rollouts_used % self.cadence == 0:
            self._apply_update()
        return bool(passed)


    def _close_group(self, task_id):
        """Bank the learning from one closed group and reset it.

        The signal is the spread of the outcomes THIS group actually returned.
        Nothing here consults the task's hidden pass probability, so a group
        that happened to come back flat is worth nothing even on a task whose
        long-run rate is a coin flip.
        """
        size, passes = self._pending[task_id]
        if size <= 0:
            return
        signal = _group_signal(passes, size)
        if size >= self.group:
            self._closed_rollouts += size
            self._groups_closed += 1
            if signal > 0.0:
                self._groups_informative += 1
        if signal == 0.0:
            self._flat_rollouts += size
        if size < self.group:
            self._partial_rollouts += size
        for s, w in _share(self._public[task_id]).items():
            sign = -1.0 if self._poison.get(task_id) else 1.0
            self._credit[s] += sign * self.lr * signal * w * size
        self._meter_write({
            "event": "group", "on": task_id, "size": size,
            "passes": passes, "signal": round(signal, 6),
        })
        self._pending[task_id] = [0, 0]

    @property
    def group_stats(self):
        """How the budget actually landed.

        `partial` is spend sitting in groups that never reached full size, and
        `flat` is spend inside groups that closed with no spread at all. They
        are independent: a policy can be good at avoiding one and bad at the
        other. `groups_closed` and `groups_informative` count groups rather
        than rollouts. Nothing here is interpreted for you -- read
        `_group_signal` and work out what a group of a given size is worth.
        """
        return {
            "closed": self._closed_rollouts,
            "flat": self._flat_rollouts,
            # Accumulated, not read off live state. Deriving it from `_pending`
            # meant close() - which flushes every open group and empties that
            # dict - reset it to zero, so a candidate inspecting their own run
            # after close() always saw 0 while the grader, reading before it,
            # saw the real number. A metric that reports success to the person
            # being measured and failure to the one measuring is worse than no
            # metric. Open groups are added in because at end-of-budget they can
            # no longer fill.
            "partial": self._partial_rollouts + sum(v[0] for v in self._pending.values()),
            "served": self.rollouts_used,
            # Counts of GROUPS, not rollouts. `informative` is the one that
            # tracks work actually done: full-size groups that carried some
            # spread, and therefore moved the model at all.
            "groups_closed": self._groups_closed,
            "groups_informative": self._groups_informative,
        }

    def _apply_update(self):
        """Land banked credit on proficiency.

        Damped by `(1 - prof)`: the closer a skill is to mastery the less a
        further unit of learning moves it, and it can never pass 1. That is why
        there is no separate saturation term anywhere in this file -- running
        out of headroom IS the diminishing return, and it is a property of the
        model rather than a decay constant bolted on beside it.
        """
        for s in self.skills:
            self._prof[s] = min(1.0, self._prof[s] + self._credit[s] * (1.0 - self._prof[s]))
            self._credit[s] = 0.0
        self.round += 1
        self._meter_write(
            {"event": "update", "n": self.rollouts_used, "round": self.round}
        )

    def eval_report(self):
        """Evaluation evaluation, scored per skill. Deterministic and analytic."""
        rows = [(dem, _pass_prob(self._prof, dem, self.slope))
                for _tid, dem in self._eval]
        return _assemble_report(rows, self.skills)

    def close(self):
        # Flush every open group and any credit banked since the last cadence
        # boundary. A budget that is not a multiple of the cadence used to
        # strand its tail: rollouts were served, their learning was banked, and
        # then thrown away without a word.
        for tid in self._task_ids:
            if self._pending[tid][0] > 0:
                self._close_group(tid)
        if any(self._credit[s] for s in self.skills):
            self._apply_update()
        # A finished run writes its trace and its rollout meter, and nothing
        # else. Scoring happens against a evaluation set that is not shipped.
        if self._meter is not None:
            self._meter.close()
            self._meter = None


def _assemble_report(rows, skills):
    """Score the evaluation probes. `rows` is [(task demands, pass probability)].

    `mean` is the score: a flat average over every probe, each counted once.
    `per_skill` and `min` are reported beside it as diagnostics -- a probe counts
    towards every skill it needs, so a skill the run never trained drags down the
    probes it shares with others too. Nothing is graded on them.
    """
    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    per_skill = {s: [] for s in skills}
    for demands, score in rows:
        for s, weight in demands.items():
            if weight > 0:
                per_skill[s].append(score)

    scores = {s: avg(xs) for s, xs in per_skill.items()}
    return {
        "mean": avg([score for _demands, score in rows]),
        "per_skill": scores,
        "min": min(scores.values()) if scores else 0.0,
    }




class BaselineScheduler:
    """The shipped reference policy: uniform round-robin, fixed group of 8.

    Draws `group` rollouts on each task in bank order, forever, until the budget
    runs out - so every group it opens it also closes, and it never abandons a
    task whose groups keep coming back flat.
    """

    def __init__(self, tasks, budget, cadence, group, seed):
        self.group = int(group)
        self._order = [t["task_id"] for t in tasks]
        self._i = 0
        self._stats = {t["task_id"]: [0, 0] for t in tasks}  # [passes, total]

    def choose(self):
        tid = self._order[(self._i // self.group) % len(self._order)]
        self._i += 1
        return tid

    def observe(self, task_id, passed):
        st = self._stats[task_id]
        st[0] += 1 if passed else 0
        st[1] += 1

