"""Problem model and fast, vectorised constraint evaluation.

A solution ("individual") is an integer array of shape (M, 2), one row per weekly
session: [room_index, start_slot]. start_slot is 0..N_SLOTS-1 (day*SLOTS_PER_DAY + slot).
A lab block of length L occupies start_slot .. start_slot+L-1 (must stay inside one day).
"""
import numpy as np

from .config import (DAYS, SLOTS_PER_DAY, N_SLOTS, W_HARD, W_GAP, W_SAMEDAY,
                     W_CONSEC, W_BALANCE, MAX_CONSEC)

TERM_NAMES = [
    "H1_faculty_clash",
    "H2_batch_clash",
    "H3_room_clash",
    "H4_room_capacity",
    "H5_room_type_or_lab_block",
    "H6_faculty_unavailable",
    "S1_gaps_in_batch_day",
    "S2_same_course_same_day",
    "S3_long_teaching_streak",
    "S4_uneven_daily_load",
]


class Problem:
    def __init__(self, courses, rooms, batches, unavailable=None, name="instance"):
        self.name = name
        self.batches = [str(b) for b in batches["batch"]]
        b_idx = {b: i for i, b in enumerate(self.batches)}
        self.batch_strength = np.array(batches["strength"], dtype=np.int64)

        self.rooms = [str(r) for r in rooms["room"]]
        self.room_cap = np.array(rooms["capacity"], dtype=np.int64)
        self.room_is_lab = np.array([str(t).strip().lower() == "lab" for t in rooms["type"]],
                                    dtype=bool)

        self.faculty = sorted(set(str(f) for f in courses["faculty"]))
        f_idx = {f: i for i, f in enumerate(self.faculty)}
        self.unavail = np.zeros((len(self.faculty), N_SLOTS), dtype=bool)
        if unavailable is not None:
            for f, d, s in zip(unavailable["faculty"], unavailable["day"], unavailable["slot"]):
                f = str(f)
                if f in f_idx:
                    self.unavail[f_idx[f], (int(d) - 1) * SLOTS_PER_DAY + int(s) - 1] = True

        self.nb, self.nr, self.nf = len(self.batches), len(self.rooms), len(self.faculty)

        # ---- expand courses into weekly sessions --------------------------------------
        self.courses = []
        sc, sb, sf, sl, slab = [], [], [], [], []
        for _, row in courses.iterrows():
            length = int(row["consecutive_slots"])
            is_lab = str(row["type"]).strip().lower() == "lab"
            cidx = len(self.courses)
            self.courses.append({
                "course_id": str(row["course_id"]), "course": str(row["course"]),
                "batch": str(row["batch"]), "faculty": str(row["faculty"]),
                "type": "lab" if is_lab else "theory", "length": length,
                "sessions": int(row["sessions_per_week"]),
            })
            for _k in range(int(row["sessions_per_week"])):
                sc.append(cidx)
                sb.append(b_idx[str(row["batch"])])
                sf.append(f_idx[str(row["faculty"])])
                sl.append(length)
                slab.append(is_lab)
        self.sess_course = np.array(sc, dtype=np.int64)
        self.sess_batch = np.array(sb, dtype=np.int64)
        self.sess_fac = np.array(sf, dtype=np.int64)
        self.sess_len = np.array(sl, dtype=np.int64)
        self.sess_lab = np.array(slab, dtype=bool)
        self.M = len(sc)
        self.nc = len(self.courses)
        self.fac_load = np.bincount(self.sess_fac, weights=self.sess_len, minlength=self.nf)

        # one entry per occupied (session, offset) cell - used for vectorised counting
        self.occ_sess = np.repeat(np.arange(self.M), self.sess_len)
        self.occ_off = np.concatenate([np.arange(int(l)) for l in self.sess_len])

        # ---- constraint-aware search space (valid rooms / start slots per session) ----
        self.allowed_rooms, self.allowed_starts, self.start_sets = [], [], []
        for i in range(self.M):
            lab = bool(self.sess_lab[i])
            length = int(self.sess_len[i])
            strength = self.batch_strength[self.sess_batch[i]]
            type_ok = [r for r in range(self.nr) if bool(self.room_is_lab[r]) == lab]
            cap_ok = [r for r in type_ok if self.room_cap[r] >= strength]
            self.allowed_rooms.append(cap_ok or type_ok or list(range(self.nr)))
            valid = [s for s in range(N_SLOTS) if (s % SLOTS_PER_DAY) + length <= SLOTS_PER_DAY]
            f = self.sess_fac[i]
            free = [s for s in valid if not any(self.unavail[f, s + o] for o in range(length))]
            starts = free or valid
            self.allowed_starts.append(starts)
            self.start_sets.append(set(starts))
        # sessions that may swap start slots with each other
        self.groups = {}
        for i in range(self.M):
            self.groups.setdefault((int(self.sess_len[i]), bool(self.sess_lab[i])), []).append(i)

    # ------------------------------------------------------------------ sanity checks
    def check(self):
        """Return a list of human-readable warnings about obviously infeasible input."""
        w = []
        if self.M == 0:
            w.append("No sessions to schedule.")
        for b in range(self.nb):
            hours = int(self.sess_len[self.sess_batch == b].sum())
            if hours > N_SLOTS:
                w.append(f"Batch {self.batches[b]} needs {hours} slots but only {N_SLOTS} exist.")
        for f in range(self.nf):
            free = int(N_SLOTS - self.unavail[f].sum())
            if self.fac_load[f] > free:
                w.append(f"Faculty {self.faculty[f]} has load {int(self.fac_load[f])} "
                         f"but only {free} free slots.")
        if self.sess_lab.any() and not self.room_is_lab.any():
            w.append("Lab courses exist but there is no room of type 'lab'.")
        if (~self.sess_lab).any() and self.room_is_lab.all():
            w.append("Theory courses exist but there is no 'classroom' room.")
        for i in range(self.M):
            strength = self.batch_strength[self.sess_batch[i]]
            if all(self.room_cap[r] < strength for r in self.allowed_rooms[i]):
                c = self.courses[self.sess_course[i]]
                w.append(f"No suitable room is big enough for {c['course_id']} "
                         f"(batch {c['batch']}, strength {int(strength)}).")
                break
        return w

    # ------------------------------------------------------------------ evaluation
    def _cells(self, ind):
        room, start = ind[:, 0], ind[:, 1]
        osess = self.occ_sess
        slot = np.minimum(start[osess] + self.occ_off, N_SLOTS - 1)
        return room, start, slot, self.sess_fac[osess], self.sess_batch[osess], room[osess]

    def terms(self, ind):
        """Raw (unweighted) violation counts, ordered as TERM_NAMES."""
        room, start, slot, fac, bat, rm = self._cells(ind)
        fc = np.bincount(fac * N_SLOTS + slot, minlength=self.nf * N_SLOTS)
        bc = np.bincount(bat * N_SLOTS + slot, minlength=self.nb * N_SLOTS)
        rc = np.bincount(rm * N_SLOTS + slot, minlength=self.nr * N_SLOTS)

        h1 = int(np.maximum(fc - 1, 0).sum())
        h2 = int(np.maximum(bc - 1, 0).sum())
        h3 = int(np.maximum(rc - 1, 0).sum())
        h4 = int((self.room_cap[room] < self.batch_strength[self.sess_batch]).sum())
        h5 = int((self.room_is_lab[room] != self.sess_lab).sum()
                 + ((start % SLOTS_PER_DAY) + self.sess_len > SLOTS_PER_DAY).sum())
        h6 = int(self.unavail[fac, slot].sum())

        # S1 gaps + S4 balance (batch occupancy)
        occ = (bc > 0).reshape(self.nb, DAYS, SLOTS_PER_DAY)
        cnt = occ.sum(axis=2)
        idx = np.arange(SLOTS_PER_DAY)
        first = np.where(occ, idx, SLOTS_PER_DAY).min(axis=2)
        last = np.where(occ, idx, -1).max(axis=2)
        s1 = int(np.where(cnt > 0, last - first + 1 - cnt, 0).sum())
        total = cnt.sum(axis=1)
        s4 = int(np.maximum(cnt.max(axis=1) - np.ceil(total / DAYS), 0).sum())

        # S2 same course twice on a day
        day = np.minimum(start // SLOTS_PER_DAY, DAYS - 1)
        cd = np.bincount(self.sess_course * DAYS + day, minlength=self.nc * DAYS)
        s2 = int(np.maximum(cd - 1, 0).sum())

        # S3 teacher teaching more than MAX_CONSEC consecutive slots
        fo = (fc > 0).reshape(self.nf, DAYS, SLOTS_PER_DAY)
        w = MAX_CONSEC + 1
        n_win = SLOTS_PER_DAY - w + 1
        s3 = 0
        if n_win > 0:
            acc = fo[:, :, 0:n_win].copy()
            for k in range(1, w):
                acc &= fo[:, :, k:n_win + k]
            s3 = int(acc.sum())
        return (h1, h2, h3, h4, h5, h6, s1, s2, s3, s4)

    def evaluate(self, ind):
        """Return (hard_violations, weighted_soft_penalty)."""
        t = self.terms(ind)
        hard = t[0] + t[1] + t[2] + t[3] + t[4] + t[5]
        soft = W_GAP * t[6] + W_SAMEDAY * t[7] + W_CONSEC * t[8] + W_BALANCE * t[9]
        return hard, soft

    def cost(self, ind):
        h, s = self.evaluate(ind)
        return W_HARD * h + s

    def breakdown(self, ind):
        t = self.terms(ind)
        d = dict(zip(TERM_NAMES, t))
        hard, soft = self.evaluate(ind)
        d["hard_total"] = hard
        d["soft_weighted"] = soft
        d["cost"] = W_HARD * hard + soft
        return d

    def conflict_mask(self, ind):
        """Boolean mask over sessions that take part in a hard violation."""
        room, start, slot, fac, bat, rm = self._cells(ind)
        fc = np.bincount(fac * N_SLOTS + slot, minlength=self.nf * N_SLOTS)
        bc = np.bincount(bat * N_SLOTS + slot, minlength=self.nb * N_SLOTS)
        rc = np.bincount(rm * N_SLOTS + slot, minlength=self.nr * N_SLOTS)
        bad = ((fc[fac * N_SLOTS + slot] > 1) | (bc[bat * N_SLOTS + slot] > 1)
               | (rc[rm * N_SLOTS + slot] > 1) | self.unavail[fac, slot])
        mask = np.bincount(self.occ_sess, weights=bad.astype(float), minlength=self.M) > 0
        mask |= self.room_is_lab[room] != self.sess_lab
        mask |= self.room_cap[room] < self.batch_strength[self.sess_batch]
        mask |= (start % SLOTS_PER_DAY) + self.sess_len > SLOTS_PER_DAY
        return mask

    # ------------------------------------------------------------------ construction
    def random_individual(self, rng):
        ind = np.empty((self.M, 2), dtype=np.int64)
        for i in range(self.M):
            ind[i, 0] = rng.choice(self.allowed_rooms[i])
            ind[i, 1] = rng.choice(self.allowed_starts[i])
        return ind

    def summary(self):
        return (f"{self.name}: {self.nb} batches, {self.nc} courses, {self.M} sessions/week, "
                f"{self.nf} faculty, {self.nr} rooms")
