"""Synthetic instance generator and validated Excel loader.

Excel format (sheet names matter, column names matter):
  Courses            : course_id, batch, course, sessions_per_week, type, consecutive_slots, faculty
  Rooms              : room, capacity, type            (type = classroom | lab)
  Batches            : batch, strength
  FacultyUnavailable : faculty, day, slot              (day 1..5, slot 1..6) - optional sheet
"""
import math
import os
import random

import pandas as pd

from .config import DAYS, SLOTS_PER_DAY
from .model import Problem

SIZES = {"small": 3, "medium": 6, "large": 12}

THEORY = [
    "Data Structures", "Algorithms", "Operating Systems", "Database Systems",
    "Computer Networks", "Machine Learning", "Soft Computing", "Software Engineering",
    "Compiler Design", "Artificial Intelligence", "Web Technologies", "Cloud Computing",
    "Cyber Security", "Data Mining", "Computer Architecture",
]
LABS = [
    "Data Structures Lab", "DBMS Lab", "ML Lab", "Networks Lab",
    "OS Lab", "Web Tech Lab", "AI Lab",
]

REQUIRED_COLUMNS = {
    "Courses": ["course_id", "batch", "course", "sessions_per_week", "type",
                "consecutive_slots", "faculty"],
    "Rooms": ["room", "capacity", "type"],
    "Batches": ["batch", "strength"],
}


def instance_path(size, out_dir="data"):
    return os.path.join(out_dir, f"timetable_input_{size}.xlsx")


def generate_instance(size="medium", seed=7, out_dir="data", out=None):
    """Create a realistic synthetic instance and write it as an Excel file."""
    if size not in SIZES:
        raise ValueError(f"size must be one of {list(SIZES)}")
    rnd = random.Random(seed)
    n_batches = SIZES[size]
    os.makedirs(out_dir, exist_ok=True)
    out = out or instance_path(size, out_dir)

    batches = pd.DataFrame({
        "batch": [f"B{i + 1}" for i in range(n_batches)],
        "strength": [rnd.randint(38, 65) for _ in range(n_batches)],
    })

    rows = []
    for b in batches["batch"]:
        for name in rnd.sample(THEORY, 5):
            rows.append([b, name, rnd.choice([3, 3, 4]), "theory", 1])
        for name in rnd.sample(LABS, 2):
            rows.append([b, name, 1, "lab", 2])          # one 2-slot block per week
    courses = pd.DataFrame(rows, columns=["batch", "course", "sessions_per_week",
                                          "type", "consecutive_slots"])
    courses.insert(0, "course_id", [f"C{i + 1:03d}" for i in range(len(courses))])

    hours = courses["sessions_per_week"] * courses["consecutive_slots"]
    n_fac = math.ceil(hours.sum() / 15) + 1
    load = {f"F{i + 1}": 0 for i in range(n_fac)}
    assigned = []
    for h in hours:
        f = min(load, key=lambda k: load[k] + rnd.random())
        load[f] += int(h)
        assigned.append(f)
    courses["faculty"] = assigned

    n_class = max(3, math.ceil(n_batches * 0.7))
    n_lab = max(2, math.ceil(n_batches * 0.4))
    room_rows = [[f"R{101 + i}", rnd.choice([45, 60, 70, 70]), "classroom"] for i in range(n_class)]
    room_rows += [[f"L{i + 1}", 70, "lab"] for i in range(n_lab)]
    rooms = pd.DataFrame(room_rows, columns=["room", "capacity", "type"])
    rooms.loc[0, "capacity"] = 70

    un = []
    for f in load:
        for _ in range(rnd.randint(2, 4)):
            un.append([f, rnd.randint(1, DAYS), rnd.randint(1, SLOTS_PER_DAY)])
    unavailable = pd.DataFrame(un, columns=["faculty", "day", "slot"]).drop_duplicates()

    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        courses.to_excel(xw, sheet_name="Courses", index=False)
        rooms.to_excel(xw, sheet_name="Rooms", index=False)
        batches.to_excel(xw, sheet_name="Batches", index=False)
        unavailable.to_excel(xw, sheet_name="FacultyUnavailable", index=False)
    return out


def _clean(df, sheet, str_cols, int_cols):
    missing = [c for c in REQUIRED_COLUMNS[sheet] if c not in df.columns]
    if missing:
        raise ValueError(f"Sheet '{sheet}' is missing column(s): {missing}")
    df = df[REQUIRED_COLUMNS[sheet]].dropna(how="all").copy()
    for c in str_cols:
        if df[c].isna().any():
            raise ValueError(f"Sheet '{sheet}': column '{c}' has empty cells")
        df[c] = df[c].astype(str).str.strip()
    for c in int_cols:
        try:
            df[c] = pd.to_numeric(df[c]).astype(int)
        except Exception:
            raise ValueError(f"Sheet '{sheet}': column '{c}' must contain whole numbers")
    return df.reset_index(drop=True)


def load_problem(source, name=None):
    """Load and validate an Excel instance. `source` is a path or a file-like object."""
    if name is None:
        name = os.path.splitext(os.path.basename(source))[0] if isinstance(source, str) \
            else "uploaded"
    try:
        with pd.ExcelFile(source) as xl:
            sheet_names = list(xl.sheet_names)
            for sheet in REQUIRED_COLUMNS:
                if sheet not in sheet_names:
                    raise ValueError(f"Missing sheet '{sheet}'. Found sheets: {sheet_names}")
            raw_courses = xl.parse("Courses")
            raw_rooms = xl.parse("Rooms")
            raw_batches = xl.parse("Batches")
            raw_un = xl.parse("FacultyUnavailable") if "FacultyUnavailable" in sheet_names else None
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Could not read the Excel file: {e}")

    courses = _clean(raw_courses, "Courses",
                     ["course_id", "batch", "course", "type", "faculty"],
                     ["sessions_per_week", "consecutive_slots"])
    rooms = _clean(raw_rooms, "Rooms", ["room", "type"], ["capacity"])
    batches = _clean(raw_batches, "Batches", ["batch"], ["strength"])

    un = pd.DataFrame(columns=["faculty", "day", "slot"])
    if raw_un is not None:
        raw_un = raw_un.dropna(how="all")
        if len(raw_un):
            for c in ("faculty", "day", "slot"):
                if c not in raw_un.columns:
                    raise ValueError(f"Sheet 'FacultyUnavailable' is missing column '{c}'")
            un = raw_un[["faculty", "day", "slot"]].copy()
            un["faculty"] = un["faculty"].astype(str).str.strip()
            un["day"] = pd.to_numeric(un["day"]).astype(int)
            un["slot"] = pd.to_numeric(un["slot"]).astype(int)
            if not un["day"].between(1, DAYS).all() or not un["slot"].between(1, SLOTS_PER_DAY).all():
                raise ValueError(f"FacultyUnavailable: day must be 1..{DAYS}, slot 1..{SLOTS_PER_DAY}")

    # ---- validation ----
    if len(courses) == 0 or len(rooms) == 0 or len(batches) == 0:
        raise ValueError("Courses, Rooms and Batches sheets must not be empty")
    if courses["course_id"].duplicated().any():
        raise ValueError("Duplicate course_id values in sheet 'Courses'")
    if batches["batch"].duplicated().any():
        raise ValueError("Duplicate batch names in sheet 'Batches'")
    if rooms["room"].duplicated().any():
        raise ValueError("Duplicate room names in sheet 'Rooms'")
    unknown = sorted(set(courses["batch"]) - set(batches["batch"]))
    if unknown:
        raise ValueError(f"Courses refer to batches missing from sheet 'Batches': {unknown}")
    if (courses["sessions_per_week"] < 1).any():
        raise ValueError("sessions_per_week must be at least 1")
    if not courses["consecutive_slots"].between(1, SLOTS_PER_DAY).all():
        raise ValueError(f"consecutive_slots must be between 1 and {SLOTS_PER_DAY}")
    bad = sorted(set(courses["type"].str.lower()) - {"theory", "lab"})
    if bad:
        raise ValueError(f"Course type must be 'theory' or 'lab', found: {bad}")
    bad = sorted(set(rooms["type"].str.lower()) - {"classroom", "lab"})
    if bad:
        raise ValueError(f"Room type must be 'classroom' or 'lab', found: {bad}")
    if (rooms["capacity"] < 1).any() or (batches["strength"] < 1).any():
        raise ValueError("Room capacity and batch strength must be positive")

    return Problem(courses, rooms, batches, un, name=name)
