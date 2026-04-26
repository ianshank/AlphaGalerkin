"""Entropy coding for lossless compression of quantized symbols.

Implements range coding (a variant of arithmetic coding) for
encoding quantized latent symbols to a bitstream.

The entropy coder must be lossless:
    decode(encode(symbols)) == symbols
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class EncodedBitstream:
    """Container for encoded bitstream and metadata."""

    data: bytes
    shape: tuple[int, ...]
    min_val: int
    max_val: int
    num_symbols: int


class RangeEncoder:
    """Range encoder for entropy coding.

    Uses 32-bit precision for the range state and a separate CDF
    precision (default 12 bits) to ensure cdf_total << max_range,
    preventing range collapse to zero during encoding.
    """

    def __init__(self, precision: int = 32, cdf_precision: int = 12) -> None:
        """Initialize range encoder.

        Args:
            precision: Precision bits for range state (must be >= 16).
            cdf_precision: Precision bits for CDF values. Must be
                strictly less than ``precision`` to guarantee
                ``range_size = range // cdf_total >= 1`` after
                renormalization.

        """
        if cdf_precision >= precision:
            raise ValueError(f"cdf_precision ({cdf_precision}) must be < precision ({precision})")
        self.precision = precision
        self.cdf_precision = cdf_precision
        self.max_range = 1 << precision

        # State
        self.low = 0
        self.range = self.max_range
        self.buffer: list[int] = []

    def encode_symbol(
        self,
        symbol: int,
        cdf_low: int,
        cdf_high: int,
        cdf_total: int,
    ) -> None:
        """Encode a single symbol.

        Args:
            symbol: Symbol to encode.
            cdf_low: CDF value at symbol (lower bound).
            cdf_high: CDF value at symbol + 1 (upper bound).
            cdf_total: Total CDF range.

        """
        range_size = self.range // cdf_total
        if range_size == 0:
            # Safety: force renormalize if range collapsed
            logger.debug(
                "Range collapse detected (range=%d, total=%d), forcing renormalization",
                self.range,
                cdf_total,
            )
            self._renormalize()
            range_size = self.range // cdf_total
            if range_size == 0:
                range_size = 1

        self.low += range_size * cdf_low
        self.range = range_size * (cdf_high - cdf_low)

        # Guard: ensure range is always positive
        if self.range <= 0:
            self.range = 1

        self._renormalize()

    def _renormalize(self) -> None:
        """Emit bytes while range is below threshold."""
        while self.range < (self.max_range >> 8):
            self.buffer.append((self.low >> (self.precision - 8)) & 0xFF)
            self.low = (self.low << 8) & (self.max_range - 1)
            self.range <<= 8

    def encode_symbols(
        self,
        symbols: Tensor,
        cdfs: Tensor,
    ) -> bytes:
        """Encode multiple symbols with their CDFs.

        Args:
            symbols: Integer symbols to encode.
            cdfs: Cumulative distribution functions (N, num_bins).

        Returns:
            Encoded bytes.

        """
        symbols_np = symbols.flatten().cpu().numpy().astype(np.int32)
        cdfs_np = cdfs.cpu().numpy().astype(np.int32)

        self.low = 0
        self.range = self.max_range
        self.buffer = []

        for i, sym in enumerate(symbols_np):
            cdf = cdfs_np[i] if cdfs_np.ndim > 1 else cdfs_np
            sym = int(sym) + len(cdf) // 2  # Offset for signed symbols
            sym = max(0, min(len(cdf) - 2, sym))  # Clamp to valid range

            self.encode_symbol(
                sym,
                int(cdf[sym]),
                int(cdf[sym + 1]),
                int(cdf[-1]),
            )

        # Flush remaining bits
        for shift in range(self.precision - 8, -1, -8):
            self.buffer.append((self.low >> shift) & 0xFF)

        return bytes(self.buffer)

    def finalize(self) -> bytes:
        """Finalize encoding and return bitstream.

        Returns:
            Encoded bytes.

        """
        for _ in range(4):
            self.buffer.append((self.low >> (self.precision - 8)) & 0xFF)
            self.low = (self.low << 8) & (self.max_range - 1)

        return bytes(self.buffer)


class RangeDecoder:
    """Range decoder for entropy coding.

    Decodes symbols from bitstream using learned CDFs.
    Must use the same precision and cdf_precision as the encoder.
    """

    def __init__(self, precision: int = 32, cdf_precision: int = 12) -> None:
        """Initialize range decoder.

        Args:
            precision: Precision bits for range state (must match encoder).
            cdf_precision: Precision bits for CDF values (must match encoder).

        """
        self.precision = precision
        self.cdf_precision = cdf_precision
        self.max_range = 1 << precision

        # State
        self.low = 0
        self.range = self.max_range
        self.code = 0
        self.data: bytes = b""
        self.pos = 0

    def init_from_bytes(self, data: bytes) -> None:
        """Initialize decoder from bitstream.

        Args:
            data: Encoded bytes.

        """
        self.data = data
        self.pos = 0
        self.low = 0
        self.range = self.max_range

        # Read initial code (precision // 8 bytes)
        self.code = 0
        for _ in range(self.precision // 8):
            self.code = (self.code << 8) | self._read_byte()

    def _read_byte(self) -> int:
        """Read next byte from bitstream."""
        if self.pos < len(self.data):
            byte = self.data[self.pos]
            self.pos += 1
            return byte
        return 0

    def _renormalize(self) -> None:
        """Read bytes while range is below threshold."""
        while self.range < (self.max_range >> 8):
            self.code = ((self.code << 8) | self._read_byte()) & (self.max_range - 1)
            self.low = (self.low << 8) & (self.max_range - 1)
            self.range <<= 8

    def decode_symbol(
        self,
        cdf: np.ndarray[np.int32, np.dtype[np.int32]],
    ) -> int:
        """Decode a single symbol.

        Args:
            cdf: Cumulative distribution function.

        Returns:
            Decoded symbol.

        """
        cdf_total = int(cdf[-1])
        range_size = self.range // cdf_total
        if range_size == 0:
            self._renormalize()
            range_size = self.range // cdf_total
            if range_size == 0:
                range_size = 1

        # Find symbol via scaled value lookup
        scaled_value = (self.code - self.low) // range_size
        symbol = int(np.searchsorted(cdf[:-1], scaled_value, side="right") - 1)
        symbol = max(0, min(len(cdf) - 2, symbol))

        # Update range
        self.low += range_size * int(cdf[symbol])
        self.range = range_size * (int(cdf[symbol + 1]) - int(cdf[symbol]))
        if self.range <= 0:
            self.range = 1

        self._renormalize()

        # Return signed symbol
        return symbol - len(cdf) // 2

    def decode_symbols(
        self,
        num_symbols: int,
        cdfs: np.ndarray[np.int32, np.dtype[np.int32]],
    ) -> np.ndarray[np.int32, np.dtype[np.int32]]:
        """Decode multiple symbols.

        Args:
            num_symbols: Number of symbols to decode.
            cdfs: CDFs for each symbol (N, num_bins) or (num_bins,).

        Returns:
            Decoded symbols.

        """
        symbols = np.zeros(num_symbols, dtype=np.int32)

        for i in range(num_symbols):
            cdf = cdfs[i] if cdfs.ndim > 1 else cdfs
            symbols[i] = self.decode_symbol(cdf)

        return symbols


class EntropyCoder:
    """High-level entropy coder combining encoder and decoder.

    Provides a simple interface for encoding/decoding symbol tensors.

    The *range_precision* controls the encoder/decoder integer range
    (32 bits by default).  The *cdf_precision* controls how finely
    probability distributions are quantised (12 bits by default).
    ``cdf_precision`` must be strictly less than ``range_precision``
    so that ``range // cdf_total >= 1`` is always satisfied after
    renormalization.
    """

    def __init__(
        self,
        range_precision: int = 32,
        cdf_precision: int = 12,
    ) -> None:
        """Initialize entropy coder.

        Args:
            range_precision: Precision bits for range state.
            cdf_precision: Precision bits for CDF quantization.
                Must be < range_precision.

        """
        if cdf_precision >= range_precision:
            raise ValueError(
                f"cdf_precision ({cdf_precision}) must be < range_precision ({range_precision})"
            )
        self.range_precision = range_precision
        self.cdf_precision = cdf_precision
        self.encoder = RangeEncoder(range_precision, cdf_precision)
        self.decoder = RangeDecoder(range_precision, cdf_precision)

    def encode(
        self,
        symbols: Tensor,
        scales: Tensor | None = None,
    ) -> EncodedBitstream:
        """Encode symbols to bitstream.

        Args:
            symbols: Integer symbols to encode.
            scales: Optional scale parameters for Gaussian model.

        Returns:
            Encoded bitstream.

        """
        # Get symbol statistics
        symbols_int = symbols.to(torch.int32)
        min_val = int(symbols_int.min())
        max_val = int(symbols_int.max())
        num_symbols = symbols.numel()

        # Build CDFs
        if scales is not None:
            cdfs = self._build_gaussian_cdfs(scales, min_val, max_val)
        else:
            cdfs = self._build_uniform_cdfs(min_val, max_val, num_symbols)

        # Encode
        data = self.encoder.encode_symbols(symbols_int, cdfs)

        return EncodedBitstream(
            data=data,
            shape=tuple(symbols.shape),
            min_val=min_val,
            max_val=max_val,
            num_symbols=num_symbols,
        )

    def decode(
        self,
        bitstream: EncodedBitstream,
        scales: Tensor | None = None,
    ) -> Tensor:
        """Decode symbols from bitstream.

        Args:
            bitstream: Encoded bitstream.
            scales: Optional scale parameters for Gaussian model.

        Returns:
            Decoded symbols.

        """
        # Initialize decoder
        self.decoder.init_from_bytes(bitstream.data)

        # Build CDFs
        if scales is not None:
            cdfs = (
                self._build_gaussian_cdfs(scales, bitstream.min_val, bitstream.max_val)
                .cpu()
                .numpy()
            )
        else:
            cdfs = (
                self._build_uniform_cdfs(
                    bitstream.min_val, bitstream.max_val, bitstream.num_symbols
                )
                .cpu()
                .numpy()
            )

        # Decode
        symbols_np = self.decoder.decode_symbols(bitstream.num_symbols, cdfs)

        # Reshape
        symbols = torch.from_numpy(symbols_np).reshape(bitstream.shape)

        return symbols

    def _build_gaussian_cdfs(
        self,
        scales: Tensor,
        min_val: int,
        max_val: int,
    ) -> Tensor:
        """Build Gaussian CDFs for given scales.

        CDFs are quantized to ``cdf_precision`` bits so that the
        total CDF value is at most ``(1 << cdf_precision) - 1``,
        keeping it well below the encoder's range precision.

        Args:
            scales: Scale (std dev) parameters.
            min_val: Minimum symbol value.
            max_val: Maximum symbol value.

        Returns:
            CDF tensor (N, num_bins + 1).

        """
        import math

        scales = scales.flatten().clamp(min=0.11)
        num_symbols = scales.numel()
        num_bins = max_val - min_val + 1

        # Create bin edges
        edges = torch.arange(min_val - 0.5, max_val + 1.5, device=scales.device)

        # Compute CDFs
        cdfs = torch.zeros(num_symbols, num_bins + 1, device=scales.device)

        for i in range(num_bins + 1):
            z = edges[i] / scales
            cdfs[:, i] = 0.5 * (1 + torch.erf(z / math.sqrt(2)))

        # Scale to cdf_precision (NOT range_precision)
        cdf_max = (1 << self.cdf_precision) - 1
        cdfs = (cdfs * cdf_max).to(torch.int32).clamp(min=0)

        # Ensure strict monotonicity
        for i in range(1, num_bins + 1):
            cdfs[:, i] = torch.maximum(cdfs[:, i], cdfs[:, i - 1] + 1)

        return cdfs

    def _build_uniform_cdfs(
        self,
        min_val: int,
        max_val: int,
        num_symbols: int,
    ) -> Tensor:
        """Build uniform CDFs.

        Args:
            min_val: Minimum symbol value.
            max_val: Maximum symbol value.
            num_symbols: Number of symbols.

        Returns:
            CDF tensor (num_bins + 1,).

        """
        num_bins = max_val - min_val + 1
        cdf_max = (1 << self.cdf_precision) - 1
        step = cdf_max // num_bins

        # Ensure step >= 1 for monotonicity
        step = max(step, 1)

        cdf = torch.arange(0, num_bins + 1) * step
        cdf[-1] = cdf_max

        return cdf.to(torch.int32)
