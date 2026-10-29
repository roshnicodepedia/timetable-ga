"""Search algorithms. All of them share one budgeted Evaluator so comparisons are fair
(same number of cost evaluations) and convergence histories are recorded uniformly."""
import math
import random
import time

import numpy as np

from .config import W_HARD, W_GAP, W_SAMEDAY, SLOTS_PER_DAY


class Evaluator:
    """Counts evaluations, tracks the best solution and the convergence history."""

    def __init__(self, problem, budget):
        self.problem = problem
        self.budget = int(budget)
        self.count = 0
        self.best = None
        self.best_cost = float("inf")
        self.first_feasible = None
        self.history = []          # (evaluations, best_cost) at every improvement

    def __call__(self, ind):
        h, s = self.problem.evaluate(ind)
        self.count += 1
        c = W_HARD * h + s
        if c < self.best_cost:
            self.best_cost = c
            self.best = ind.copy()
            self.history.append((self.count, c))
        if h == 0 and self.first_feasible is None:
            self.first_feasible = self.count
        return c

    @property
    def done(self):
        return self.count >= self.budget or self.best_cost == 0


# ---------------------------------------------------------------------------- moves
def apply_move(problem, ind, i, rng):
    """Random in-place move on session i. Returns an undo list [(row_index, old_row), ...]."""
    if rng.random() < 0.2:                                   # swap start slots of two sessions
        grp = problem.groups[(int(problem.sess_len[i]), bool(problem.sess_lab[i]))]
        if len(grp) > 1:
            j = rng.choice(grp)
            si, sj = int(ind[i, 1]), int(ind[j, 1])
            if (j != i and si != sj and sj in problem.start_sets[i]
                    and si in problem.start_sets[j]):
                undo = [(i, ind[i].copy()), (j, ind[j].copy())]
                ind[i, 1], ind[j, 1] = sj, si
                return undo
    undo = [(i, ind[i].copy())]
    if rng.random() < 0.5:
        ind[i, 0] = rng.choice(problem.allowed_rooms[i])
    else:
        ind[i, 1] = rng.choice(problem.allowed_starts[i])
    return undo


def revert(ind, undo):
    for idx, row in undo:
        ind[idx] = row


def mutate(problem, ind, rng, pm):
    for i in range(problem.M):
        if rng.random() < pm:
            apply_move(problem, ind, i, rng)


def crossover(problem, a, b, rng, kind="batch"):
    if kind == "batch":          # whole batches are inherited intact from one parent
        pick = np.array([rng.random() < 0.5 for _ in range(problem.nb)])
        take_a = pick[problem.sess_batch]
    elif kind == "uniform":      # every session independently
        take_a = np.array([rng.random() < 0.5 for _ in range(problem.M)])
    elif kind == "onepoint":
        cut = rng.randrange(1, max(2, problem.M))
        take_a = np.arange(problem.M) < cut
    else:
        raise ValueError(f"unknown crossover '{kind}'")
    return np.where(take_a[:, None], a, b)


def local_search(problem, ev, rng, ind, cost, tries):
    """Hill climbing that focuses on sessions involved in hard conflicts (repair)."""
    for _ in range(tries):
        if ev.done:
            break
        if cost >= W_HARD:
            idx = np.flatnonzero(problem.conflict_mask(ind))
            i = int(idx[rng.randrange(len(idx))]) if len(idx) else rng.randrange(problem.M)
        else:
            i = rng.randrange(problem.M)
        undo = apply_move(problem, ind, i, rng)
        c = ev(ind)
        if c <= cost:
            cost = c
        else:
            revert(ind, undo)
    return cost


# ---------------------------------------------------------------------------- algorithms
def random_search(problem, ev, rng):
    while not ev.done:
        ev(problem.random_individual(rng))


def simulated_annealing(problem, ev, rng, t0=10.0, t_end=0.1):
    cur = problem.random_individual(rng)
    cur_c = ev(cur)
    while not ev.done:
        temp = t0 * (t_end / t0) ** (ev.count / ev.budget)
        i = rng.randrange(problem.M)
        undo = apply_move(problem, cur, i, rng)
        c = ev(cur)
        delta = c - cur_c
        if delta <= 0 or rng.random() < math.exp(-delta / temp):
            cur_c = c
        else:
            revert(cur, undo)


def genetic_algorithm(problem, ev, rng, pop_size=40, pc=0.8, pm=0.005, elite=2, k=3,
                      crossover_kind="batch", memetic=False, ls_tries=10, ls_rate=0.5):
    pop, costs = [], []
    for _ in range(pop_size):
        ind = problem.random_individual(rng)
        pop.append(ind)
        costs.append(ev(ind))
        if ev.done:
            return

    def tournament():
        picks = [rng.randrange(len(pop)) for _ in range(k)]
        return pop[min(picks, key=costs.__getitem__)]

    while not ev.done:
        order = sorted(range(len(pop)), key=costs.__getitem__)
        new_pop = [pop[i] for i in order[:elite]]
        new_costs = [costs[i] for i in order[:elite]]
        while len(new_pop) < pop_size and not ev.done:
            a, b = tournament(), tournament()
            if rng.random() < pc:
                child = crossover(problem, a, b, rng, crossover_kind).copy()
            else:
                child = a.copy()
            mutate(problem, child, rng, pm)
            c = ev(child)
            if memetic and rng.random() < ls_rate and not ev.done:
                c = local_search(problem, ev, rng, child, c, ls_tries)
            new_pop.append(child)
            new_costs.append(c)
        if len(new_pop) == pop_size:
            pop, costs = new_pop, new_costs


def greedy_construct(problem, ev, rng):
    """Constructive heuristic: place the hardest sessions first at the best-scoring slot."""
    P = problem
    order = sorted(range(P.M), key=lambda i: (-int(P.sess_len[i]),
                                              -float(P.fac_load[P.sess_fac[i]]),
                                              len(P.allowed_starts[i]), rng.random()))
    ind = np.zeros((P.M, 2), dtype=np.int64)
    fac_occ = [set() for _ in range(P.nf)]
    bat_occ = [set() for _ in range(P.nb)]
    room_occ = set()
    bat_day = {}
    course_day = set()
    for i in order:
        f, b = int(P.sess_fac[i]), int(P.sess_batch[i])
        L, c = int(P.sess_len[i]), int(P.sess_course[i])
        best = None
        for st in P.allowed_starts[i]:
            slots = range(st, st + L)
            conf = sum(s in fac_occ[f] for s in slots) + sum(s in bat_occ[b] for s in slots)
            free = [r for r in P.allowed_rooms[i] if not any((r, s) in room_occ for s in slots)]
            if free:
                r = rng.choice(free)
            else:
                r = rng.choice(P.allowed_rooms[i])
                conf += L
            d, ins = divmod(st, SLOTS_PER_DAY)
            day_slots = bat_day.get((b, d), set())
            inc = 0
            if day_slots:
                lo, hi = min(day_slots), max(day_slots)
                nlo, nhi = min(lo, ins), max(hi, ins + L - 1)
                inc = ((nhi - nlo + 1) - (len(day_slots) + L)) - ((hi - lo + 1) - len(day_slots))
            score = (W_HARD * conf + W_GAP * inc + (W_SAMEDAY if (c, d) in course_day else 0)
                     + 0.5 * len(day_slots) + rng.random() * 0.1)
            if best is None or score < best[0]:
                best = (score, st, r)
        _, st, r = best
        ind[i, 0], ind[i, 1] = r, st
        d, ins = divmod(st, SLOTS_PER_DAY)
        for s in range(st, st + L):
            fac_occ[f].add(s)
            bat_occ[b].add(s)
            room_occ.add((r, s))
        bat_day.setdefault((b, d), set()).update(range(ins, ins + L))
        course_day.add((c, d))
    ev(ind)


METHODS = {
    "random": random_search,
    "greedy": greedy_construct,
    "sa": simulated_annealing,
    "ga": genetic_algorithm,
    "memetic": lambda p, ev, rng, **kw: genetic_algorithm(p, ev, rng, memetic=True, **kw),
}


def run_algorithm(method, problem, budget, seed, **params):
    """Run one algorithm once. Returns a dict with the best solution and statistics."""
    if method not in METHODS:
        raise ValueError(f"unknown method '{method}', choose from {list(METHODS)}")
    rng = random.Random(seed)
    ev = Evaluator(problem, budget)
    t0 = time.perf_counter()
    METHODS[method](problem, ev, rng, **params)
    elapsed = time.perf_counter() - t0
    hard, soft = problem.evaluate(ev.best)
    return {
        "method": method, "seed": seed, "best": ev.best,
        "hard": int(hard), "soft": int(soft), "cost": int(W_HARD * hard + soft),
        "feasible": hard == 0, "evals": ev.count,
        "first_feasible": ev.first_feasible, "time": elapsed,
        "history": list(ev.history),
    }
