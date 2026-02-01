"""Entropy coding for lossless compression of quantized symbols.

Implements range coding (a variant of arithmetic coding) for
encoding quantized latent symbols to a bitstream.

The entropy coder must be lossless:
    decode(encode(symbols)) == symbols
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import numpy as np


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

    Implements asymmetric numeral systems (ANS) style range coding
    for efficient compression of symbols with learned distributions.
    """

    def __init__(self, precision: int = 16) -> None:
        """Initialize range encoder.

        Args:
            precision: Precision bits for probability representation.
        """
        self.precision = precision
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
        # Update range
        range_size = self.range // cdf_total
        self.low += range_size * cdf_low
        self.range = range_size * (cdf_high - cdf_low)

        # Renormalize
        while self.range < (self.max_range >> 8):
            self.buffer.append(self.low >> (self.precision - 8))
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

        # Encode each symbol
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
        self.buffer.append(self.low >> (self.precision - 8))
        self.buffer.append((self.low >> (self.precision - 16)) & 0xFF)

        return bytes(self.buffer)

    def finalize(self) -> bytes:
        """Finalize encoding and return bitstream.

        Returns:
            Encoded bytes.
        """
        # Flush remaining state
        for _ in range(4):
            self.buffer.append(self.low >> (self.precision - 8))
            self.low = (self.low << 8) & (self.max_range - 1)

        return bytes(self.buffer)


class RangeDecoder:
    """Range decoder for entropy coding.

    Decodes symbols from bitstream using learned CDFs.
    """

    def __init__(self, precision: int = 16) -> None:
        """Initialize range decoder.

        Args:
            precision: Precision bits for probability representation.
        """
        self.precision = precision
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

        # Read initial code
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

    def decode_symbol(
        self,
        cdf: np.ndarray,
    ) -> int:
        """Decode a single symbol.

        Args:
            cdf: Cumulative distribution function.

        Returns:
            Decoded symbol.
        """
        cdf_total = int(cdf[-1])
        range_size = self.range // cdf_total

        # Find symbol
        scaled_value = (self.code - self.low) // range_size
        symbol = np.searchsorted(cdf[:-1], scaled_value, side="right") - 1
        symbol = max(0, symbol)

        # Update range
        self.low += range_size * int(cdf[symbol])
        self.range = range_size * (int(cdf[symbol + 1]) - int(cdf[symbol]))

        # Renormalize
        while self.range < (self.max_range >> 8):
            self.code = ((self.code << 8) | self._read_byte()) & (self.max_range - 1)
            self.low = (self.low << 8) & (self.max_range - 1)
            self.range <<= 8

        # Return signed symbol
        return symbol - len(cdf) // 2

    def decode_symbols(
        self,
        num_symbols: int,
        cdfs: np.ndarray,
    ) -> np.ndarray:
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
    """

    def __init__(self, precision: int = 16) -> None:
        """Initialize entropy coder.

        Args:
            precision: Precision bits for probability representation.
        """
        self.precision = precision
        self.encoder = RangeEncoder(precision)
        self.decoder = RangeDecoder(precision)

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
            cdfs = self._build_gaussian_cdfs(
                scales, bitstream.min_val, bitstream.max_val
            ).cpu().numpy()
        else:
            cdfs = self._build_uniform_cdfs(
                bitstream.min_val, bitstream.max_val, bitstream.num_symbols
            ).cpu().numpy()

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
            # Standard normal CDF
            z = edges[i] / scales
            cdfs[:, i] = 0.5 * (1 + torch.erf(z / math.sqrt(2)))

        # Scale to precision
        cdfs = (cdfs * ((1 << self.precision) - 1)).to(torch.int32)
        cdfs = cdfs.clamp(min=0)

        # Ensure monotonicity
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
        step = (1 << self.precision) // num_bins

        cdf = torch.arange(0, num_bins + 1) * step
        cdf[-1] = (1 << self.precision) - 1

        return cdf.to(torch.int32)
