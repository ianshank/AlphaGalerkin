"""PDE Operator definitions with automatic differentiation.

This package provides abstract and concrete PDE operators for:
- Defining PDE equations declaratively
- Computing residuals via automatic differentiation
- Supporting time-dependent and steady-state PDEs

Each operator implements:
- residual(): Computes PDE residual at collocation points
- exact_solution(): Optional analytical solution for testing
- source_term(): Source/forcing term
- boundary_condition(): Boundary value function

Supported PDEs:
- Poisson: ∇²u = f
- Burgers: u_t + u·∇u = ν∇²u
- Advection-Diffusion: u_t + a·∇u = ν∇²u + f
- Heat: u_t = κ∇²u + f
- Wave: u_tt = c²∇²u + f

This module was originally a single flat file (``src/pde/operators.py``,
docs/CODE_HYGIENE_AUDIT.md B4/§3.1: "10 PDE-family classes in one flat
namespace"). It is now a package with one concrete operator class per file
(``poisson.py``, ``burgers.py``, ``advection_diffusion.py``, ``heat.py``,
``navier_stokes.py``, ``lshaped_poisson.py``, ``helmholtz.py``,
``biharmonic.py``) plus the shared ``PDEResidual``/``PDEOperator`` base in
``base.py``. This file re-exports every name the old module exposed, so
``from src.pde.operators import PoissonOperator`` (and every other existing
import site across ``src/``, ``tests/``, ``dashboard/``, and ``hf_space/``)
continues to work unchanged -- the split is transparent to callers.

The imports below intentionally mirror the *exact* top-of-file import block
of the old monolithic module (down to the incidental names it leaked into its
namespace, e.g. ``ABC``, ``Tensor``, ``np``), so that every *public* name
(``[n for n in dir(src.pde.operators) if not n.startswith("_")]``) is
unchanged before and after the split. **Not** every name in the raw
``dir()`` output: becoming a package unavoidably adds ``__path__``, and the
explicit ``__all__`` below (absent from the old flat module) adds itself as
an attribute -- both dunders, both harmless, since nothing in this codebase
introspects ``dir()`` on this module. An earlier revision of this docstring
claimed the stronger, false "``dir()`` is byte-identical" property; that
claim was not what was actually checked and did not hold (verified 2026-09-01
via a peer review of PR #140) -- see ``tests/pde/test_operators.py`` for the
test that encodes the real, narrower, true guarantee.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import structlog
import torch
from numpy.typing import NDArray
from scipy.special import ive
from torch import Tensor

from src.constants import DEFAULT_BOUNDARY_TOLERANCE
from src.pde.config import BoundaryCondition, PDEConfig, PDEType
from src.pde.geometry import (
    DomainGeometry,
    GeometryType,
    LShapedDomain,
    create_geometry,
)

logger = structlog.get_logger(__name__)

from src.pde.operators.advection_diffusion import (  # noqa: E402
    GAUSSIAN_PULSE_WIDTH_FRACTION,
    AdvectionDiffusionOperator,
)
from src.pde.operators.base import PDEOperator, PDEResidual  # noqa: E402
from src.pde.operators.biharmonic import BiharmonicOperator  # noqa: E402
from src.pde.operators.burgers import (  # noqa: E402
    COLE_HOPF_CLAMP_EPS,
    COLE_HOPF_COEFFICIENT_CACHE_SIZE,
    COLE_HOPF_MAX_TERMS,
    COLE_HOPF_MIN_RESOLVED_VISCOSITY,
    COLE_HOPF_N_TERMS,
    COLE_HOPF_TERM_TOLERANCE,
    BurgersOperator,
    _cole_hopf_coefficients,
)
from src.pde.operators.heat import HeatOperator  # noqa: E402
from src.pde.operators.helmholtz import (  # noqa: E402
    DEFAULT_HELMHOLTZ_WAVENUMBER,
    HelmholtzOperator,
)
from src.pde.operators.lshaped_poisson import LShapedPoissonOperator  # noqa: E402
from src.pde.operators.navier_stokes import NavierStokesOperator  # noqa: E402
from src.pde.operators.poisson import PoissonOperator  # noqa: E402

# ``from src.pde.operators.<submodule> import ...`` above has an unavoidable
# side effect documented in the Python import system: each submodule is bound
# as an attribute of this package (e.g. ``src.pde.operators.poisson`` becomes
# accessible as ``operators.poisson``), which the old flat module never
# exposed. Deleting those bindings here (the submodules stay importable and
# fully cached in ``sys.modules`` -- only this package's own namespace loses
# the attribute) keeps every *public* name in ``dir(src.pde.operators)``
# unchanged from the pre-split module (see the module docstring above and
# ``tests/pde/test_operators.py::TestOperatorsPackagePublicAPI`` for the
# actual, narrower guarantee -- not literal ``dir()`` byte-identity, which a
# package's ``__path__``/``__all__`` dunders make impossible).
del (
    advection_diffusion,
    base,
    biharmonic,
    burgers,
    heat,
    helmholtz,
    lshaped_poisson,
    navier_stokes,
    poisson,
)

__all__ = [
    "ABC",
    "AdvectionDiffusionOperator",
    "Any",
    "BiharmonicOperator",
    "BoundaryCondition",
    "BurgersOperator",
    "COLE_HOPF_CLAMP_EPS",
    "COLE_HOPF_COEFFICIENT_CACHE_SIZE",
    "COLE_HOPF_MAX_TERMS",
    "COLE_HOPF_MIN_RESOLVED_VISCOSITY",
    "COLE_HOPF_N_TERMS",
    "COLE_HOPF_TERM_TOLERANCE",
    "Callable",
    "DEFAULT_BOUNDARY_TOLERANCE",
    "DEFAULT_HELMHOLTZ_WAVENUMBER",
    "DomainGeometry",
    "GAUSSIAN_PULSE_WIDTH_FRACTION",
    "GeometryType",
    "HeatOperator",
    "HelmholtzOperator",
    "LShapedDomain",
    "LShapedPoissonOperator",
    "NDArray",
    "NavierStokesOperator",
    "PDEConfig",
    "PDEOperator",
    "PDEResidual",
    "PDEType",
    "PoissonOperator",
    "Tensor",
    "_cole_hopf_coefficients",
    "abstractmethod",
    "create_geometry",
    "dataclass",
    "ive",
    "logger",
    "lru_cache",
    "np",
    "structlog",
    "torch",
]
