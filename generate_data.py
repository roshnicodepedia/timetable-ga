"""Generate synthetic input instances.

    python generate_data.py            # creates small, medium and large in ./data
    python generate_data.py medium     # only one size
"""
import sys

from ttga.data import generate_instance, SIZES, load_problem

if __name__ == "__main__":
    sizes = sys.argv[1:] or list(SIZES)
    for size in sizes:
        path = generate_instance(size)
        print(f"{path}  ->  {load_problem(path).summary()}")
