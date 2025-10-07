import importlib


def check_package(my_package):
    try:
        importlib.import_module(my_package)
        print(f"[INFO] {my_package} is installed correctly.")
        return 0
    except ImportError:
        print(f"Error: {my_package} is not installed.")
        return 1

packages = ["torch", "matplotlib", "numpy", "jax", "tqdm"]
flag = 0
for package in packages:
    out = check_package(package)
    flag += out
if flag != 0:

    print("[WARNING] At least one package needs to be reinstalled.")
else:
    print("All packages installed correctly.")
