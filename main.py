"""Demo: solve one instance and write the timetable to Excel.

    python main.py                              # medium instance, memetic GA
    python main.py --instance large --algo ga
    python main.py --instance my_real_data.xlsx --algo memetic --budget 60000
"""
import argparse
import os

from ttga.algorithms import run_algorithm, METHODS
from ttga.data import SIZES, generate_instance, instance_path, load_problem
from ttga.export import write_excel, save_convergence_plot

DEFAULT_BUDGET = {"small": 20000, "medium": 40000, "large": 80000}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--instance", default="medium", help="small | medium | large | path to .xlsx")
    ap.add_argument("--algo", default="memetic", choices=list(METHODS))
    ap.add_argument("--budget", type=int, default=None, help="number of cost evaluations")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="output")
    a = ap.parse_args()

    if a.instance in SIZES:
        path = instance_path(a.instance)
        if not os.path.exists(path):
            generate_instance(a.instance)
    else:
        path = a.instance
    problem = load_problem(path)
    budget = a.budget or DEFAULT_BUDGET.get(a.instance, 40000)

    print(problem.summary())
    for w in problem.check():
        print("WARNING:", w)
    print(f"Running {a.algo} (budget {budget} evaluations, seed {a.seed}) ...")
    r = run_algorithm(a.algo, problem, budget, a.seed)

    print(f"Result: hard violations = {r['hard']}, soft penalty = {r['soft']}, "
          f"evaluations used = {r['evals']}, time = {r['time']:.1f}s")
    print("First feasible solution after", r["first_feasible"], "evaluations")
    print("Constraint breakdown:")
    for k, v in problem.breakdown(r["best"]).items():
        print(f"  {k:28s} {v}")

    os.makedirs(a.out, exist_ok=True)
    xlsx = os.path.join(a.out, "timetable.xlsx")
    png = os.path.join(a.out, "convergence.png")
    write_excel(problem, r["best"], xlsx, r["history"])
    save_convergence_plot({a.algo: r["history"]}, png, budget=r["evals"],
                          title=f"{problem.name} - {a.algo}")
    print(f"Saved {xlsx} and {png}")
    if r["hard"] != 0:
        print("NOTE: the timetable still has hard violations; try a larger --budget or --algo memetic.")


if __name__ == "__main__":
    main()
