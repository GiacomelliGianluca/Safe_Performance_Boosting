import os
import sys
import subprocess

ENV_NAME = ".venv" if os.name != "nt" else "venv"

def create_virtualenv():
    """Creates a virtual environment if it does not exist."""
    if not os.path.exists(ENV_NAME):
        print(f"Creating virtual environment '{ENV_NAME}'...")
        subprocess.run([sys.executable, "-m", "venv", ENV_NAME])
    else:
        print(f"The virtual environment '{ENV_NAME}' already exists.")

def install_dependencies():
    """Installs dependencies from requirements.txt."""
    pip_executable = os.path.join(ENV_NAME, "bin", "pip") if os.name != "nt" else os.path.join(ENV_NAME, "Scripts", "pip")
    print("Installing dependencies from requirements.txt...")
    subprocess.run([pip_executable, "install", "--upgrade", "pip"])
    subprocess.run([pip_executable, "install", "-r", "requirements.txt"])

def main():
    create_virtualenv()
    install_dependencies()
    print("\nInstallation complete. Please activate the virtual environment in PyCharm.")
    # activation_cmd = f"source {ENV_NAME}/bin/activate" if os.name != "nt" else f"{ENV_NAME}\\Scripts\\activate"
    # print("\nInstallation complete. To activate the virtual environment, run:")
    # print(f"\n    {activation_cmd}\n")

if __name__ == "__main__":
    main()
