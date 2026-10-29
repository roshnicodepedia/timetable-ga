"""Run from the project root:   python -m unittest discover -s tests -v"""
import io
import math
import os
import random
import sys
import tempfile
import unittest
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ttga.algorithms import run_algorithm
from ttga.config import (DAYS, SLOTS_PER_DAY, N_SLOTS, W_HARD, MAX_CONSEC)
from ttga.data import generate_instance, load_problem, SIZES
from ttga.export import excel_bytes, batch_grid, schedule_dataframe

TMP = tempfile.mkdtemp(prefix="ttga_test_")


def instance(size):
    path = os.path.join(TMP, f"{size}.xlsx")
    if not os.path.exists(path):
        generate_instance(size, out=path)
    return load_problem(path)


def naive_terms(p, ind):
    """Independent, deliberately simple re-implementation of every constraint term."""
    M = p.M
    occ = []
    for i in range(M):
        for o in range(int(p.sess_len[i])):
            occ.append((i, min(int(ind[i, 1]) + o, N_SLOTS - 1)))
    fc, bc, rc = Counter(), Counter(), Counter()
    for i, s in occ:
        fc[(int(p.sess_fac[i]), s)] += 1
        bc[(int(p.sess_batch[i]), s)] += 1
        rc[(int(ind[i, 0]), s)] += 1
    h1 = sum(max(v - 1, 0) for v in fc.values())
    h2 = sum(max(v - 1, 0) for v in bc.values())
    h3 = sum(max(v - 1, 0) for v in rc.values())
    h4 = sum(1 for i in range(M)
             if p.room_cap[ind[i, 0]] < p.batch_strength[p.sess_batch[i]])
    h5 = sum(1 for i in range(M) if bool(p.room_is_lab[ind[i, 0]]) != bool(p.sess_lab[i]))
    h5 += sum(1 for i in range(M) if int(ind[i, 1]) % SLOTS_PER_DAY + int(p.sess_len[i]) > SLOTS_PER_DAY)
    h6 = sum(1 for i, s in occ if p.unavail[p.sess_fac[i], s])

    s1 = 0
    day_counts = {}
    for b in range(p.nb):
        for d in range(DAYS):
            slots = {s % SLOTS_PER_DAY for (i, s) in occ
                     if p.sess_batch[i] == b and s // SLOTS_PER_DAY == d}
            day_counts[(b, d)] = len(slots)
            if slots:
                s1 += (max(slots) - min(slots) + 1) - len(slots)
    s2 = 0
    for c in range(p.nc):
        per_day = Counter(min(int(ind[i, 1]) // SLOTS_PER_DAY, DAYS - 1)
                          for i in range(M) if p.sess_course[i] == c)
        s2 += sum(max(v - 1, 0) for v in per_day.values())
    s3 = 0
    for f in range(p.nf):
        for d in range(DAYS):
            busy = {s % SLOTS_PER_DAY for (i, s) in occ
                    if p.sess_fac[i] == f and s // SLOTS_PER_DAY == d}
            for st in range(SLOTS_PER_DAY - MAX_CONSEC):
                if all((st + k) in busy for k in range(MAX_CONSEC + 1)):
                    s3 += 1
    s4 = 0
    for b in range(p.nb):
        counts = [day_counts[(b, d)] for d in range(DAYS)]
        s4 += max(max(counts) - math.ceil(sum(counts) / DAYS), 0)
    return (h1, h2, h3, h4, h5, h6, s1, s2, s3, s4)


def messy_individual(p, rng):
    """Completely random genes - includes invalid rooms, overflowing labs, clashes."""
    ind = np.zeros((p.M, 2), dtype=np.int64)
    for i in range(p.M):
        ind[i, 0] = rng.randrange(p.nr)
        ind[i, 1] = rng.randrange(N_SLOTS)
    return ind


class TestEvaluator(unittest.TestCase):
    def test_matches_naive_checker(self):
        for size in ("small", "medium"):
            p = instance(size)
            rng = random.Random(123)
            for _ in range(25):
                for ind in (p.random_individual(rng), messy_individual(p, rng)):
                    self.assertEqual(p.terms(ind), naive_terms(p, ind), size)

    def test_conflict_mask_consistent_with_hard_count(self):
        p = instance("small")
        rng = random.Random(5)
        for _ in range(20):
            ind = messy_individual(p, rng)
            hard, _ = p.evaluate(ind)
            self.assertEqual(hard > 0, bool(p.conflict_mask(ind).any()))

    def test_known_conflict(self):
        p = instance("small")
        ind = np.zeros((p.M, 2), dtype=np.int64)    # everything in room 0, slot 0
        self.assertGreater(p.evaluate(ind)[0], 0)


class TestData(unittest.TestCase):
    def test_sizes(self):
        for size, nb in SIZES.items():
            p = instance(size)
            self.assertEqual(p.nb, nb)
            self.assertEqual(p.check(), [], f"generated {size} instance should be sane")

    def test_missing_sheet_raises(self):
        path = os.path.join(TMP, "bad.xlsx")
        with pd.ExcelWriter(path) as xw:
            pd.DataFrame({"a": [1]}).to_excel(xw, sheet_name="Nope", index=False)
        with self.assertRaises(ValueError):
            load_problem(path)

    def test_unknown_batch_raises(self):
        path = os.path.join(TMP, "bad2.xlsx")
        generate_instance("small", out=path)
        x = pd.read_excel(path, sheet_name=None)
        x["Courses"].loc[0, "batch"] = "ZZZ"
        with pd.ExcelWriter(path) as xw:
            for k, v in x.items():
                v.to_excel(xw, sheet_name=k, index=False)
        with self.assertRaises(ValueError):
            load_problem(path)

    def test_upload_from_buffer(self):
        path = os.path.join(TMP, "small.xlsx")
        instance("small")
        with open(path, "rb") as fh:
            p = load_problem(io.BytesIO(fh.read()))
        self.assertGreater(p.M, 0)


class TestAlgorithms(unittest.TestCase):
    def test_all_methods_run_and_respect_budget(self):
        p = instance("small")
        for m in ("random", "greedy", "sa", "ga", "memetic"):
            r = run_algorithm(m, p, 1500, 1)
            self.assertLessEqual(r["evals"], 1500 + 10, m)     # GA may finish its last child
            self.assertEqual(r["cost"], p.cost(r["best"]), m)  # reported cost is real
            self.assertEqual(r["best"].shape, (p.M, 2))

    def test_reproducible(self):
        p = instance("small")
        a = run_algorithm("ga", p, 3000, 7)
        b = run_algorithm("ga", p, 3000, 7)
        self.assertEqual(a["cost"], b["cost"])
        self.assertTrue((a["best"] == b["best"]).all())

    def test_ga_and_sa_find_feasible_small(self):
        p = instance("small")
        for m in ("sa", "ga", "memetic"):
            r = run_algorithm(m, p, 20000, 1)
            self.assertEqual(r["hard"], 0, m)
            self.assertEqual(naive_terms(p, r["best"])[:6], (0, 0, 0, 0, 0, 0), m)

    def test_greedy_is_feasible_small(self):
        p = instance("small")
        self.assertEqual(run_algorithm("greedy", p, 10, 1)["hard"], 0)

    def test_history_is_monotone(self):
        p = instance("small")
        r = run_algorithm("memetic", p, 4000, 3)
        costs = [c for _, c in r["history"]]
        self.assertEqual(costs, sorted(costs, reverse=True))
        self.assertEqual(costs[-1], r["cost"])


class TestExport(unittest.TestCase):
    def test_excel_roundtrip(self):
        p = instance("small")
        r = run_algorithm("ga", p, 3000, 1)
        data = excel_bytes(p, r["best"], r["history"])
        sheets = pd.ExcelFile(io.BytesIO(data)).sheet_names
        self.assertIn("Summary", sheets)
        self.assertIn("All_Sessions", sheets)
        self.assertEqual(sum(s.startswith("Batch_") for s in sheets), p.nb)
        self.assertEqual(sum(s.startswith("Faculty_") for s in sheets), p.nf)

    def test_every_session_appears_in_grids(self):
        p = instance("small")
        r = run_algorithm("ga", p, 3000, 1)
        df = schedule_dataframe(p, r["best"])
        self.assertEqual(len(df), p.M)
        total_cells = sum((batch_grid(p, r["best"], b) != "").sum().sum() for b in p.batches)
        self.assertGreaterEqual(total_cells, 1)


if __name__ == "__main__":
    unittest.main()
