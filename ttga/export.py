"""Turn a solution into readable timetables and an Excel workbook."""
import io
import re

import pandas as pd

from .config import DAY_NAMES, DAYS, SLOTS_PER_DAY, N_SLOTS


def schedule_dataframe(problem, ind):
    rows = []
    for i in range(problem.M):
        c = problem.courses[problem.sess_course[i]]
        start = int(ind[i, 1])
        day, s = divmod(start, SLOTS_PER_DAY)
        length = int(problem.sess_len[i])
        rows.append({
            "batch": c["batch"], "course_id": c["course_id"], "course": c["course"],
            "type": c["type"], "faculty": c["faculty"],
            "room": problem.rooms[int(ind[i, 0])],
            "day_no": min(day, DAYS - 1) + 1, "day": DAY_NAMES[min(day, DAYS - 1)],
            "start_slot": s + 1, "end_slot": min(s + length, SLOTS_PER_DAY),
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["batch", "day_no", "start_slot"]).reset_index(drop=True)


def _grid(cells):
    data = {DAY_NAMES[d]: [" // ".join(cells.get((d, s), [])) for s in range(SLOTS_PER_DAY)]
            for d in range(DAYS)}
    return pd.DataFrame(data, index=[f"Slot {s + 1}" for s in range(SLOTS_PER_DAY)])


def batch_grid(problem, ind, batch_name):
    b = problem.batches.index(batch_name)
    cells = {}
    for i in range(problem.M):
        if problem.sess_batch[i] != b:
            continue
        c = problem.courses[problem.sess_course[i]]
        label = f"{c['course']} | {c['faculty']} | {problem.rooms[int(ind[i, 0])]}"
        for o in range(int(problem.sess_len[i])):
            slot = int(ind[i, 1]) + o
            if slot < N_SLOTS:
                d, s = divmod(slot, SLOTS_PER_DAY)
                cells.setdefault((d, s), []).append(label)
    return _grid(cells)


def faculty_grid(problem, ind, faculty_name):
    f = problem.faculty.index(faculty_name)
    cells = {}
    for i in range(problem.M):
        if problem.sess_fac[i] != f:
            continue
        c = problem.courses[problem.sess_course[i]]
        label = f"{c['course']} ({c['batch']}) | {problem.rooms[int(ind[i, 0])]}"
        for o in range(int(problem.sess_len[i])):
            slot = int(ind[i, 1]) + o
            if slot < N_SLOTS:
                d, s = divmod(slot, SLOTS_PER_DAY)
                cells.setdefault((d, s), []).append(label)
    return _grid(cells)


def _sheet_name(prefix, name, used):
    base = re.sub(r"[\[\]:*?/\\]", "_", f"{prefix}_{name}")[:31]
    out, n = base, 2
    while out in used:
        out = f"{base[:28]}_{n}"
        n += 1
    used.add(out)
    return out


def write_excel(problem, ind, target, history=None):
    """Write the timetable workbook to a path or file-like object."""
    used = {"Summary", "All_Sessions", "Convergence"}
    summary = pd.DataFrame(list(problem.breakdown(ind).items()), columns=["metric", "value"])
    with pd.ExcelWriter(target, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="Summary", index=False)
        schedule_dataframe(problem, ind).drop(columns=["day_no"]).to_excel(
            xw, sheet_name="All_Sessions", index=False)
        grid_sheets = []
        for b in problem.batches:
            name = _sheet_name("Batch", b, used)
            batch_grid(problem, ind, b).to_excel(xw, sheet_name=name)
            grid_sheets.append(name)
        for f in problem.faculty:
            name = _sheet_name("Faculty", f, used)
            faculty_grid(problem, ind, f).to_excel(xw, sheet_name=name)
            grid_sheets.append(name)
        if history:
            pd.DataFrame(history, columns=["evaluations", "best_cost"]).to_excel(
                xw, sheet_name="Convergence", index=False)
        from openpyxl.styles import Alignment
        for name in grid_sheets:
            ws = xw.sheets[name]
            ws.column_dimensions["A"].width = 10
            for col in "BCDEF":
                ws.column_dimensions[col].width = 34
            for row in ws.iter_rows(min_row=2, min_col=2):
                for cell in row:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
        for name in xw.sheets:
            if name not in grid_sheets:
                xw.sheets[name].column_dimensions["A"].width = 30


def excel_bytes(problem, ind, history=None):
    buf = io.BytesIO()
    write_excel(problem, ind, buf, history)
    return buf.getvalue()


def save_convergence_plot(histories, path, budget=None, title="Convergence"):
    """histories: dict label -> list of (evaluations, best_cost)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, hist in histories.items():
        if not hist:
            continue
        xs = [e for e, _ in hist]
        ys = [c + 1 for _, c in hist]
        end = budget or xs[-1]
        xs.append(max(end, xs[-1]))
        ys.append(ys[-1])
        ax.step(xs, ys, where="post", label=label)
    ax.set_yscale("log")
    ax.set_xlabel("cost evaluations")
    ax.set_ylabel("best cost + 1 (log)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
