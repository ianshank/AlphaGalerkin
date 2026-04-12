"""AlphaGalerkin dashboard tab modules.

Each module exposes a ``create_*_tab()`` function that adds one Gradio
``gr.Tab`` block inside an existing ``gr.Blocks`` context.

Modules
-------
game_tab
    Human vs AI and AI vs AI Go game.
pde_tab
    Interactive Poisson equation solver with resolution comparison.
poc_tab
    PoC scenario runner (Complexity, LBB Stability, Transfer milestone).
training_tab
    Training dashboard — architecture summary, loss breakdown, training curves.
reentry_tab
    Reentry Thermal Protection System (TPS) heat-diffusion analysis.
wildfire_tab
    Wildfire spread simulation via advection-diffusion with combustion.
missile_defense_tab
    Missile defense intercept trajectory analysis with potential flow.

"""

from __future__ import annotations

from dashboard.tabs.game_tab import create_game_tab
from dashboard.tabs.missile_defense_tab import create_missile_defense_tab
from dashboard.tabs.pde_tab import create_pde_tab
from dashboard.tabs.poc_tab import create_poc_tab
from dashboard.tabs.reentry_tab import create_reentry_tab
from dashboard.tabs.training_tab import create_training_tab
from dashboard.tabs.wildfire_tab import create_wildfire_tab

__all__ = [
    "create_game_tab",
    "create_missile_defense_tab",
    "create_pde_tab",
    "create_poc_tab",
    "create_reentry_tab",
    "create_training_tab",
    "create_wildfire_tab",
]
