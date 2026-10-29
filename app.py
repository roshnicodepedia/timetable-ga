"""Streamlit app:  streamlit run app.py"""
import io
import os
import time

import numpy as np
import pandas as pd
import streamlit as st

from ttga.algorithms import run_algorithm
from ttga.data import SIZES, generate_instance, instance_path, load_problem
from ttga.export import batch_grid, excel_bytes, faculty_grid, schedule_dataframe

st.set_page_config(page_title="Timetable Generator", layout="wide")
st.title("Timetable Generator (Genetic Algorithm)")
st.caption("Soft Computing project - constraint-based timetable scheduling with GA, "
           "GA + local search, Simulated Annealing and baselines.")

ALGOS = {
    "Genetic Algorithm + local search (recommended)": "memetic",
    "Genetic Algorithm": "ga",
    "Simulated Annealing": "sa",
    "Greedy heuristic": "greedy",
    "Random search": "random",
}
DEFAULT_BUDGET = {"small": 20000, "medium": 40000, "large": 80000}

# ------------------------------------------------------------------------- sidebar: data
st.sidebar.header("1. Input data")
source = st.sidebar.radio("Data source", ["Generated sample", "Upload Excel file"])
problem, size_key, sample_bytes = None, "medium", None
try:
    if source == "Generated sample":
        size_key = st.sidebar.selectbox("Instance size", list(SIZES), index=1)
        path = instance_path(size_key)
        if not os.path.exists(path):
            generate_instance(size_key)
        problem = load_problem(path)
        with open(path, "rb") as fh:
            sample_bytes = fh.read()
        st.sidebar.download_button("Download this input file", sample_bytes,
                                   file_name=os.path.basename(path),
                                   mime="application/vnd.openxmlformats-officedocument."
                                        "spreadsheetml.sheet")
    else:
        up = st.sidebar.file_uploader("Excel file (.xlsx)", type=["xlsx"])
        if up is None:
            st.info("Upload an Excel file with sheets: Courses, Rooms, Batches and "
                    "(optional) FacultyUnavailable. Or switch to 'Generated sample'.")
            st.stop()
        problem = load_problem(io.BytesIO(up.getvalue()), name=os.path.splitext(up.name)[0])
except ValueError as e:
    st.error(f"Input problem: {e}")
    st.stop()

st.subheader("Instance")
st.write(problem.summary())
for w in problem.check():
    st.warning(w)

# ------------------------------------------------------------------------- sidebar: solver
st.sidebar.header("2. Solver")
algo_label = st.sidebar.selectbox("Algorithm", list(ALGOS))
budget = st.sidebar.slider("Search budget (cost evaluations)", 2000, 200000,
                           DEFAULT_BUDGET.get(size_key, 40000) if source == "Generated sample"
                           else 40000, step=2000)
seed = st.sidebar.number_input("Random seed", min_value=0, value=1, step=1)
run = st.sidebar.button("Generate timetable", type="primary")

if run:
    with st.spinner("Searching for a timetable ..."):
        t0 = time.perf_counter()
        res = run_algorithm(ALGOS[algo_label], problem, int(budget), int(seed))
        res["wall"] = time.perf_counter() - t0
    st.session_state["res"] = {"problem": problem, "result": res, "algo": algo_label}

# ------------------------------------------------------------------------- results
stored = st.session_state.get("res")
if stored is None:
    st.info("Choose the settings in the sidebar and press 'Generate timetable'.")
    st.stop()

problem, res = stored["problem"], stored["result"]
best = res["best"]
st.subheader(f"Result - {stored['algo']}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Hard violations", res["hard"])
c2.metric("Soft penalty", res["soft"])
c3.metric("Evaluations used", res["evals"])
c4.metric("Time (s)", f"{res['wall']:.1f}")
if res["hard"] == 0:
    st.success("Feasible timetable: no clashes, all capacity / lab / availability rules satisfied.")
else:
    st.error("The timetable still has hard violations. Increase the budget or use the "
             "GA + local search algorithm.")

bd = problem.breakdown(best)
breakdown_df = pd.DataFrame([(k, v) for k, v in bd.items()], columns=["constraint", "count"])

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Batch timetables", "Faculty timetables", "All sessions", "Constraint breakdown",
     "Convergence"])
with tab1:
    b = st.selectbox("Batch", problem.batches)
    st.table(batch_grid(problem, best, b))
with tab2:
    f = st.selectbox("Faculty", problem.faculty)
    st.table(faculty_grid(problem, best, f))
with tab3:
    st.dataframe(schedule_dataframe(problem, best).drop(columns=["day_no"]))
with tab4:
    st.table(breakdown_df)
with tab5:
    hist = pd.DataFrame(res["history"], columns=["evaluations", "best_cost"])
    hist["log10(best_cost + 1)"] = np.log10(hist["best_cost"] + 1)
    st.line_chart(hist.set_index("evaluations")["log10(best_cost + 1)"])

st.download_button("Download timetable (Excel)", excel_bytes(problem, best, res["history"]),
                   file_name="timetable.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
