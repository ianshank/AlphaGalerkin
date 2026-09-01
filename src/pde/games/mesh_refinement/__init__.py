"""Mesh Refinement Game for adaptive FEM/DG methods.

This package provides a PDEGame where:
- State: Current mesh + solution quality indicators
- Actions: Refine specific elements (h or p refinement)
- Reward: Error reduction per DOF added
- Terminal: Error < tolerance or DOF budget exhausted

MCTS can look ahead multiple refinement steps to find optimal
refinement sequences, outperforming single-step error indicators.

Note:
    The current Mesh implementation supports 2D quadrilateral elements only.
    For 1D (intervals), 3D (hexahedra), or higher dimensions, a specialized
    mesh class would be required. This limitation is validated at runtime.

This module was originally a single flat file (``src/pde/games/mesh_refinement.py``,
docs/CODE_HYGIENE_AUDIT.md B4/§3.1: "a pure quadtree data structure (``Mesh``,
~313 lines) and the MCTS game (~700 lines) in one file"). It is now a package
with two files: ``mesh.py`` (the domain-free ``ActionKind``/``MeshElement``/
``Mesh`` quadtree, with no dependency on MCTS or ``torch``) and ``game.py``
(the MCTS-facing ``MeshRefinementGame``, built on ``mesh.py``). This file
re-exports every name the old flat module exposed, so
``from src.pde.games.mesh_refinement import MeshRefinementGame`` (and every
other existing import site across ``src/``, ``tests/``, ``dashboard/``, and
``hf_space/``) continues to work unchanged -- the split is transparent to
callers.

The imports below intentionally mirror the *exact* top-of-file import block
of the old monolithic module (down to the incidental names it leaked into its
namespace, e.g. ``copy``, ``Tensor``, ``np``), so that every *public* name
(``[n for n in dir(src.pde.games.mesh_refinement) if not n.startswith("_")]``)
is unchanged before and after the split. **Not** every name in the raw
``dir()`` output: becoming a package unavoidably adds ``__path__``, and the
explicit ``__all__`` below (absent from the old flat module) adds itself as
an attribute -- both dunders, both harmless, since nothing in this codebase
introspects ``dir()`` on this module. See the ``src/pde/operators.py ->
operators/`` split (docs/CODE_HYGIENE_AUDIT.md B4/B21) for the precedent this
recipe follows, including the corrected, narrower claim about what "public
API unchanged" actually means -- see
``tests/pde/test_mesh_refinement.py::TestMeshRefinementPackagePublicAPI`` for
the test that encodes the real guarantee.

"""

from __future__ import annotations

import copy
import itertools
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from torch import Tensor

from src.pde.config import MeshRefinementConfig, PDEGameConfig, RefinementStrategy
from src.pde.game import GamePhase, PDEGame, PDEState
from src.pde.reward import log_reward

logger = structlog.get_logger(__name__)

from src.pde.games.mesh_refinement.game import MeshRefinementGame  # noqa: E402
from src.pde.games.mesh_refinement.mesh import ActionKind, Mesh, MeshElement  # noqa: E402

# ``from src.pde.games.mesh_refinement.<submodule> import ...`` above has an
# unavoidable side effect documented in the Python import system: each
# submodule is bound as an attribute of this package (e.g.
# ``src.pde.games.mesh_refinement.mesh`` becomes accessible as
# ``mesh_refinement.mesh``), which the old flat module never exposed.
# Deleting those bindings here (the submodules stay importable and fully
# cached in ``sys.modules`` -- only this package's own namespace loses the
# attribute) keeps every *public* name in ``dir(src.pde.games.mesh_refinement)``
# unchanged from the pre-split module (see the module docstring above and
# ``tests/pde/test_mesh_refinement.py::TestMeshRefinementPackagePublicAPI``
# for the actual, narrower guarantee -- not literal ``dir()`` byte-identity,
# which a package's ``__path__``/``__all__`` dunders make impossible).
del game, mesh

__all__ = [
    "ActionKind",
    "Any",
    "Float",
    "GamePhase",
    "IntEnum",
    "LinearNDInterpolator",
    "Mesh",
    "MeshElement",
    "MeshRefinementConfig",
    "MeshRefinementGame",
    "NDArray",
    "NearestNDInterpolator",
    "PDEGame",
    "PDEGameConfig",
    "PDEState",
    "RefinementStrategy",
    "TYPE_CHECKING",
    "Tensor",
    "copy",
    "dataclass",
    "field",
    "itertools",
    "log_reward",
    "logger",
    "np",
    "structlog",
    "time",
    "torch",
]
