"""Robot-dog stair-climbing RL project (MuJoCo + Gymnasium + Stable-Baselines3).

Importing this package also repairs the console encoding, so Chinese log output
survives Windows redirection (see :mod:`robotdog.console`).
"""

from robotdog.console import setup_utf8_console

__version__ = "0.1.0"

setup_utf8_console()

__all__ = ["__version__", "setup_utf8_console"]
