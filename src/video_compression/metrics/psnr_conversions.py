"""Closed-form PSNR <-> MSE conversion helpers.

Both rdcurve.py (zoo) and h265_baseline.py (zoo) need to back-derive
an MSE-style "distortion" surrogate from a PSNR (dB) measurement so
the existing :class:`~src.video_compression.metrics.rd_curves.RDPoint`
``distortion`` field stays populated. Pulling the formula into one
place keeps the two callers (and any future ones — e.g. VMAF, LPIPS)
consistent and the formula testable in isolation.

Mathematical reference:
  PSNR_dB = 10 * log10(MAX^2 / MSE)
  =>  MSE = MAX^2 * 10^(-PSNR_dB / 10)

The two constants below come straight from that definition; they are
*not* arbitrary tuning knobs.
"""

from __future__ import annotations

#: Decibel base — the 10 in PSNR's ``10 * log10(...)`` definition.
PSNR_DB_LOG_BASE: float = 10.0
#: Default peak signal value for normalized [0, 1] tensors.
DEFAULT_MAX_SIGNAL: float = 1.0


def psnr_db_to_mse_surrogate(
    psnr_db: float,
    *,
    max_signal: float = DEFAULT_MAX_SIGNAL,
) -> float:
    """Convert a PSNR measurement in dB back to the implied MSE.

    Args:
        psnr_db: PSNR in dB. Must be finite; a value of ``+inf`` returns
            ``0.0`` (perfect reconstruction).
        max_signal: Peak signal value used when computing PSNR. Defaults
            to 1.0 (normalized tensors).

    Returns:
        The MSE surrogate ``MAX^2 * 10^(-PSNR_dB / 10)`` as a float.

    Raises:
        ValueError: When ``max_signal`` is not strictly positive.

    """
    if max_signal <= 0.0:
        raise ValueError(f"max_signal must be > 0; got {max_signal!r}")
    if psnr_db == float("inf"):
        return 0.0
    return float(max_signal**2) * float(PSNR_DB_LOG_BASE ** (-psnr_db / PSNR_DB_LOG_BASE))


__all__ = [
    "DEFAULT_MAX_SIGNAL",
    "PSNR_DB_LOG_BASE",
    "psnr_db_to_mse_surrogate",
]
