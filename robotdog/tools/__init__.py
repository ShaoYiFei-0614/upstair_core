"""Reusable, dependency-light helpers for the robot-dog RL project.

Everything in this subpackage is a *tool*: it knows nothing about PPO, the
curriculum or the training loop, and none of it imports :mod:`robotdog.env`.
The core environment / training modules live one level up in
:mod:`robotdog`.

* :mod:`robotdog.tools.paths`   -- canonical filesystem locations
* :mod:`robotdog.tools.console` -- the UTF-8 console fix
* :mod:`robotdog.tools.png`     -- dependency-free PNG read/write
* :mod:`robotdog.tools.terrain` -- parametric stair geometry + MJCF terrain
* :mod:`robotdog.tools.touch`   -- contact forces and the forward rangefinder

Design notes for each module are in ``docs/robotdog/tools/``.
"""

from __future__ import annotations

from robotdog.tools import console, paths, png, terrain, touch

__all__ = ["console", "paths", "png", "terrain", "touch"]
