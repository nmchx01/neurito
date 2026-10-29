"""Command-line entry point for running Neurito from the project checkout."""

from pathlib import Path
import runpy


def main() -> None:
    project_main = Path(__file__).resolve().parents[2] / "main.py"
    runpy.run_path(str(project_main), run_name="__main__")
