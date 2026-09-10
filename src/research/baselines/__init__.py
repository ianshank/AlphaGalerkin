"""Classical PDE solver baselines for benchmarking.

Provides reference implementations against which AlphaGalerkin's
MCTS-guided approach is compared in SBIR proposals.

Baselines:
- UniformFDMSolver: Finite difference on uniform grid (scipy.sparse)
- DorflerAMRSolver: Dorfler marking adaptive mesh refinement
- SimplePINNSolver: Physics-Informed Neural Network baseline

This module was originally a single flat file (``src/research/baselines.py``,
docs/CODE_HYGIENE_AUDIT.md B34). It is now a package with schemas, predicates,
one solver class per file, and a registry. This file re-exports every name the
old module exposed, so ``from src.research.baselines import DorflerAMRSolver``
(and every other existing import site) continues to work unchanged.

The imports below intentionally mirror the *exact* top-of-file import block
of the old monolithic module (down to the incidental names it leaked into its
namespace, e.g. ``ABC``, ``np``, ``resolve_device``), so that every *public*
name (``[n for n in dir(src.research.baselines) if not n.startswith("_")]``)
is unchanged before and after the split. **Not** every name in the raw
``dir()`` output: becoming a package unavoidably adds ``__path__``, and the
explicit ``__all__`` below (absent from the old flat module) adds itself as
an attribute. See ``tests/research/test_baselines.py::TestBaselinesPackagePublicAPI``.

``extra_solvers`` register into :data:`SOLVER_REGISTRY` by importing it from
this package. This ``__init__`` binds the registry before any extra-solver
import, and does **not** import ``extra_solvers`` itself (the old module did
not either).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from src.device import resolve_device
from src.pde.config import PDEType
from src.pde.operators import PDEOperator
from src.research.gpu_profiler import GpuUtilizationProfiler
from src.research.marking import dorfler_mark

logger = structlog.get_logger(__name__)

from src.research.baselines.amr import DorflerAMRSolver  # noqa: E402
from src.research.baselines.base import BaseSolver  # noqa: E402
from src.research.baselines.fdm import UniformFDMSolver  # noqa: E402
from src.research.baselines.navier_stokes import NavierStokesFDMSolver  # noqa: E402
from src.research.baselines.pinn import SimplePINNSolver  # noqa: E402
from src.research.baselines.predicates import (  # noqa: E402
    InsidePredicate,
    element_inside_mask,
    nodal_rms_l2_error,
    require_exact_solution,
    require_measurable_l2,
)
from src.research.baselines.registry import (  # noqa: E402
    SOLVER_REGISTRY,
    get_solver,
    list_solvers,
)
from src.research.baselines.schemas import (  # noqa: E402
    AMRConfig,
    FDMConfig,
    NavierStokesConfig,
    PINNConfig,
    SolverConfig,
    SolverResult,
)

# ``from src.research.baselines.<submodule> import ...`` binds each submodule
# as an attribute of this package, which the old flat module never exposed.
# Deleting those bindings (the submodules stay importable and cached in
# ``sys.modules``) keeps every *public* name in ``dir(src.research.baselines)``
# equal to the pre-split freeze. Not literal ``dir()`` byte-identity — a
# package's ``__path__`` / ``__all__`` dunders make that impossible.
del (
    amr,
    base,
    fdm,
    navier_stokes,
    pinn,
    predicates,
    registry,
    schemas,
)

__all__ = [
    "ABC",
    "AMRConfig",
    "Any",
    "BaseModel",
    "BaseSolver",
    "Callable",
    "ConfigDict",
    "DorflerAMRSolver",
    "FDMConfig",
    "Field",
    "GpuUtilizationProfiler",
    "InsidePredicate",
    "NDArray",
    "NavierStokesConfig",
    "NavierStokesFDMSolver",
    "PDEOperator",
    "PDEType",
    "PINNConfig",
    "SOLVER_REGISTRY",
    "SimplePINNSolver",
    "SolverConfig",
    "SolverResult",
    "UniformFDMSolver",
    "abstractmethod",
    "dataclass",
    "dorfler_mark",
    "element_inside_mask",
    "field",
    "get_solver",
    "list_solvers",
    "logger",
    "nodal_rms_l2_error",
    "np",
    "require_exact_solution",
    "require_measurable_l2",
    "resolve_device",
    "structlog",
    "time",
    "torch",
]
