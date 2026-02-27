"""PRNG key manager for JAX's explicit-key random number generation.

JAX uses a functional PRNG model where every random call requires an
explicit key.  This module provides a stateful wrapper that splits keys
automatically, so the rest of the codebase can request keys without
manually threading state.

Example:
    from src.backend.rng import KeyManager

    km = KeyManager(seed=42)
    key1 = km.next()          # First subkey
    key2 = km.next()          # Second subkey (derived from split)
    keys = km.split(num=4)    # Four independent subkeys
    km.reset(seed=0)          # Re-seed entirely

"""

from __future__ import annotations

__all__ = ["KeyManager"]

from typing import Any


class KeyManager:
    """Manages JAX PRNG keys throughout the pipeline.

    JAX requires explicit PRNG keys (no global state). This class
    provides a stateful wrapper that splits keys automatically.

    The internal state is a single JAX PRNG key. Each call to
    :meth:`next` or :meth:`split` advances the state by splitting
    the current key, consuming one subkey for the caller and
    retaining the other for future use.

    Attributes:
        _key: The current JAX PRNG key (advanced on every consumption).

    """

    def __init__(self, seed: int) -> None:
        """Initialize the key manager with a seed.

        Args:
            seed: Integer seed for the initial PRNG key.

        Raises:
            ImportError: If JAX is not installed.

        """
        self._jax_random = _get_jax_random()
        self._key = self._jax_random.PRNGKey(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def next(self) -> Any:
        """Return the next PRNG subkey and advance internal state.

        Returns:
            A JAX PRNG key suitable for passing to ``jax.random.*``
            functions.

        """
        self._key, subkey = self._jax_random.split(self._key)
        return subkey

    def split(self, num: int = 2) -> list[Any]:
        """Split the current key into *num* independent subkeys.

        The internal state is advanced so that subsequent calls
        produce fresh keys.

        Args:
            num: Number of subkeys to generate. Must be >= 1.

        Returns:
            List of *num* independent PRNG keys.

        Raises:
            ValueError: If *num* < 1.

        """
        if num < 1:
            msg = f"num must be >= 1, got {num}"
            raise ValueError(msg)

        # Split into num + 1 keys: one to keep, num to return.
        all_keys = self._jax_random.split(self._key, num=num + 1)
        # First key becomes the new internal state.
        self._key = all_keys[0]
        return [all_keys[i] for i in range(1, num + 1)]

    def reset(self, seed: int) -> None:
        """Reset the key manager with a new seed.

        Args:
            seed: Integer seed for the new PRNG key.

        """
        self._key = self._jax_random.PRNGKey(seed)

    @property
    def current(self) -> Any:
        """Return the current key (read-only, does NOT advance state).

        This is primarily useful for inspection and debugging.
        Prefer :meth:`next` when you actually need a key for
        random number generation, since reusing the same key
        produces identical draws.

        Returns:
            The current internal PRNG key.

        """
        return self._key


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _get_jax_random() -> Any:
    """Lazily import ``jax.random`` and return the module.

    Raises:
        ImportError: If JAX is not installed, with a helpful message
            explaining how to install it.

    """
    try:
        import jax.random

        return jax.random
    except ImportError:
        msg = (
            "JAX is required for KeyManager but is not installed. "
            "Install it with:  pip install jax jaxlib\n"
            "For GPU support:  pip install jax[cuda12]\n"
            "See https://jax.readthedocs.io/en/latest/installation.html"
        )
        raise ImportError(msg) from None
