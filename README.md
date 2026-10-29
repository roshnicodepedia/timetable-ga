# Constraint-Based University Timetable Scheduling using Genetic Algorithm
### with a Comparative Study against Simulated Annealing (Soft Computing course project)

Generates clash-free weekly timetables (batches, faculty, rooms, labs, faculty availability) and
compares a **Genetic Algorithm** (core technique), a **GA + local-search (memetic) hybrid**,
**Simulated Annealing**, a **greedy heuristic** and **random search**.

---------------------------------------------------------------------------------------------
## 1. Setup (once)

Python 3.9 or newer.

    python -m venv venv
    venv\Scripts\activate            (Windows)      source venv/bin/activate   (Linux / macOS)
    pip install -r requirements.txt

All commands below are run from this folder.

## 2. Quick start (2 minutes)

    python generate_data.py          # creates data/timetable_input_{small,medium,large}.xlsx
    python main.py                   # solves 'medium', writes output/timetable.xlsx + convergence.png
    streamlit run app.py             # web app: upload / generate / view / download
    python -m unittest discover -s tests -v     # 14 tests (should all pass)

Useful variants:

    python main.py --instance large --algo ga --budget 80000 --seed 3
    python main.py --instance path/to/your_real_data.xlsx --algo memetic

## 3. Input format (Excel)

| Sheet | Columns | Notes |
|---|---|---|
| Courses | course_id, batch, course, sessions_per_week, type, consecutive_slots, faculty | type = `theory` or `lab`; a lab block uses `consecutive_slots` = 2 |
| Rooms | room, capacity, type | type = `classroom` or `lab` |
| Batches | batch, strength | number of students |
| FacultyUnavailable (optional) | faculty, day, slot | day 1-5, slot 1-6 |

The week is 5 days x 6 slots (change in `ttga/config.py`). The files in `data/` are examples.
To test on your own class timetable, copy its subjects / faculty / lecture counts into the same format.

## 4. Problem definition

A solution assigns every weekly session a (room, start slot).

Hard constraints (must be 0 for a valid timetable)
- H1 a teacher is never in two places at once
- H2 a batch never has two sessions at once
- H3 a room is never used twice at once
- H4 room capacity >= batch strength
- H5 lab sessions use lab rooms (theory uses classrooms) and a lab block stays inside one day
- H6 teachers are not scheduled in their unavailable slots

Soft constraints (minimised, weights in `ttga/config.py`)
- S1 idle gaps in a batch's day
- S2 the same course twice on one day
- S3 a teacher teaching more than 3 slots in a row
- S4 uneven daily load of a batch

    cost = 10000 x (hard violations) + S1 + 2 x S2 + S3 + S4          fitness = 1 / (1 + cost)

## 4b. How the methods work

- **Encoding**: one gene per session = (room, start slot). The search space is constraint-aware:
  genes only take rooms of the right type / capacity and start slots that fit the lab block and the
  teacher's availability (H4-H6 are therefore satisfied by construction, but are still checked by the
  evaluator). Clashes H1-H3 are handled by the penalty in the cost.
- **GA**: tournament selection (k=3), **batch-wise crossover** (a whole batch's schedule is inherited
  intact from one parent), mutation (change room / change slot / swap slots of two sessions), elitism.
- **GA + local search (memetic)**: after mutation, half of the children get a short hill-climb that
  focuses on sessions involved in clashes (repair).
- **Simulated Annealing**: same moves, geometric cooling, same cost function.
- **Greedy**: places hardest sessions first at the best-scoring free slot (no search).
- **Random search**: lower bound.
- **Fair comparison**: every method gets the same number of cost evaluations (the budget).

## 5. Experiments for the report

    python run_experiments.py --exp main            # E1 comparison, E2 convergence, E5 equal budget
    python run_experiments.py --exp sensitivity     # E3 population / mutation / crossover study
    python run_experiments.py --exp ablation        # E4 crossover type, elitism, local search
    python run_experiments.py --exp all             # everything
    python run_experiments.py --exp all --quick     # ONLY a smoke test - do not use its numbers

Options: `--seeds 10` (default), `--instances small medium large <your.xlsx>` (E6: real case),
`--workers N` (parallel processes; default = CPU cores - 1), `--budget-scale 0.5`.

Outputs in `results/`: `main_summary.csv`, `main_runs.csv`, `main_wilcoxon.csv` (paired significance
tests), `convergence_*.png`, `boxplot_soft.png`, `sensitivity_*`, `ablation_*`.

Runtime (single core): `--exp main` about 20-25 min for 10 seeds, sensitivity about 7 min,
ablation about 4 min. More cores divide this. Start with `--seeds 3` if you are in a hurry.

Metrics to report: success rate (runs with 0 hard violations), soft penalty, evaluations to first
feasible solution, runtime, Wilcoxon p-values.

## 6. Folder structure

    ttga/config.py        constants and weights
    ttga/model.py         problem model + fast vectorised constraint evaluation
    ttga/data.py          instance generator + validated Excel loader
    ttga/algorithms.py    random, greedy, SA, GA, memetic GA (shared budgeted evaluator)
    ttga/export.py        timetable grids, Excel export, convergence plot
    generate_data.py      create sample instances
    main.py               solve one instance
    run_experiments.py    all experiments, statistics, plots
    app.py                Streamlit app
    tests/                unit tests (incl. independent brute-force constraint checker)

## 7. Troubleshooting

- `ModuleNotFoundError`: activate the virtual environment and run `pip install -r requirements.txt`.
- "Missing sheet / column" error: compare your Excel file with the table in section 3.
- Hard violations remain: increase `--budget` or use `--algo memetic`.
- Excel file locked on Windows: close it in Excel before re-running.
