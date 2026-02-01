"""GOP (Group of Pictures) manager for video compression.

Handles frame scheduling, reference frame management, and
I/P/B frame structure for temporal compression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator

from torch import Tensor


class FrameType(str, Enum):
    """Frame types in video compression."""

    I = "I"  # Intra frame (no references)
    P = "P"  # Predicted frame (forward reference)
    B = "B"  # Bidirectional frame (forward + backward reference)


@dataclass
class FrameInfo:
    """Information about a video frame."""

    index: int  # Frame index in video
    gop_index: int  # Index within current GOP
    frame_type: FrameType
    display_order: int  # Display order in GOP
    encode_order: int  # Encoding order in GOP
    forward_ref: int | None = None  # Forward reference frame index
    backward_ref: int | None = None  # Backward reference frame index
    qp: int | None = None  # Quantization parameter

    @property
    def is_reference(self) -> bool:
        """Check if this frame is used as a reference."""
        return self.frame_type != FrameType.B


@dataclass
class ReferenceBuffer:
    """Buffer for storing decoded reference frames."""

    capacity: int = 2  # Usually 2 for forward/backward

    # Storage
    frames: dict[int, Tensor] = field(default_factory=dict)
    latents: dict[int, Tensor] = field(default_factory=dict)

    def add(
        self,
        frame_idx: int,
        decoded: Tensor,
        latent: Tensor | None = None,
    ) -> None:
        """Add a reference frame.

        Args:
            frame_idx: Frame index.
            decoded: Decoded frame tensor.
            latent: Optional latent tensor.
        """
        self.frames[frame_idx] = decoded
        if latent is not None:
            self.latents[frame_idx] = latent

        # Remove old references if over capacity
        if len(self.frames) > self.capacity:
            oldest = min(self.frames.keys())
            del self.frames[oldest]
            if oldest in self.latents:
                del self.latents[oldest]

    def get(self, frame_idx: int) -> Tensor | None:
        """Get reference frame by index.

        Args:
            frame_idx: Frame index.

        Returns:
            Reference frame tensor or None.
        """
        return self.frames.get(frame_idx)

    def get_latent(self, frame_idx: int) -> Tensor | None:
        """Get reference latent by index.

        Args:
            frame_idx: Frame index.

        Returns:
            Reference latent tensor or None.
        """
        return self.latents.get(frame_idx)

    def clear(self) -> None:
        """Clear all references."""
        self.frames.clear()
        self.latents.clear()


class GOPManager:
    """Manages GOP structure and frame scheduling.

    Handles:
    - Frame type assignment (I/P/B)
    - Encoding order vs display order
    - Reference frame tracking
    - GOP boundary detection
    """

    def __init__(
        self,
        gop_size: int = 16,
        i_frame_interval: int = 16,
        use_b_frames: bool = True,
        b_frame_count: int = 3,
    ) -> None:
        """Initialize GOP manager.

        Args:
            gop_size: Number of frames in a GOP.
            i_frame_interval: Interval between I-frames.
            use_b_frames: Whether to use B-frames.
            b_frame_count: Number of B-frames between references.
        """
        self.gop_size = gop_size
        self.i_frame_interval = i_frame_interval
        self.use_b_frames = use_b_frames
        self.b_frame_count = b_frame_count

        # State
        self.frame_count = 0
        self.current_gop = 0
        self.reference_buffer = ReferenceBuffer()

    def get_frame_info(self, frame_idx: int) -> FrameInfo:
        """Get frame info for a given frame index.

        Args:
            frame_idx: Global frame index.

        Returns:
            FrameInfo for the frame.
        """
        gop_idx = frame_idx // self.gop_size
        gop_position = frame_idx % self.gop_size

        # Determine frame type
        if gop_position == 0 or frame_idx % self.i_frame_interval == 0:
            frame_type = FrameType.I
            forward_ref = None
            backward_ref = None
        elif self.use_b_frames and gop_position % (self.b_frame_count + 1) != 0:
            frame_type = FrameType.B
            # Find nearest reference frames
            ref_interval = self.b_frame_count + 1
            forward_ref = frame_idx - (gop_position % ref_interval)
            backward_ref = forward_ref + ref_interval
            if backward_ref >= (gop_idx + 1) * self.gop_size:
                backward_ref = None  # No backward ref at GOP end
        else:
            frame_type = FrameType.P
            # Forward reference is previous reference frame
            ref_interval = self.b_frame_count + 1 if self.use_b_frames else 1
            forward_ref = frame_idx - ref_interval
            if forward_ref < gop_idx * self.gop_size:
                forward_ref = gop_idx * self.gop_size  # First frame of GOP
            backward_ref = None

        # Compute encode order (B-frames are encoded after their references)
        display_order = gop_position
        encode_order = self._compute_encode_order(gop_position)

        return FrameInfo(
            index=frame_idx,
            gop_index=gop_position,
            frame_type=frame_type,
            display_order=display_order,
            encode_order=encode_order,
            forward_ref=forward_ref,
            backward_ref=backward_ref,
        )

    def _compute_encode_order(self, display_order: int) -> int:
        """Compute encoding order from display order.

        B-frames are encoded after their reference frames.

        Args:
            display_order: Display order within GOP.

        Returns:
            Encoding order within GOP.
        """
        if not self.use_b_frames:
            return display_order

        ref_interval = self.b_frame_count + 1

        # Reference frames first, then B-frames
        if display_order % ref_interval == 0:
            # This is a reference frame
            return display_order // ref_interval
        else:
            # This is a B-frame
            num_refs = (display_order // ref_interval) + 1
            b_offset = (display_order % ref_interval) - 1
            return num_refs + b_offset

    def get_gop_frames(self, gop_start: int) -> list[FrameInfo]:
        """Get all frame infos for a GOP.

        Args:
            gop_start: Starting frame index of GOP.

        Returns:
            List of FrameInfo for all frames in GOP.
        """
        frames = []
        for i in range(self.gop_size):
            frames.append(self.get_frame_info(gop_start + i))
        return frames

    def get_encoding_order(self, gop_start: int) -> list[FrameInfo]:
        """Get frames in encoding order for a GOP.

        Args:
            gop_start: Starting frame index of GOP.

        Returns:
            List of FrameInfo sorted by encoding order.
        """
        frames = self.get_gop_frames(gop_start)
        return sorted(frames, key=lambda f: f.encode_order)

    def iter_frames(
        self,
        start: int = 0,
        end: int | None = None,
    ) -> Iterator[FrameInfo]:
        """Iterate over frames in encoding order.

        Args:
            start: Starting frame index.
            end: Ending frame index (exclusive).

        Yields:
            FrameInfo for each frame in encoding order.
        """
        current = start
        while end is None or current < end:
            gop_start = (current // self.gop_size) * self.gop_size
            gop_frames = self.get_encoding_order(gop_start)

            for frame_info in gop_frames:
                if frame_info.index >= current and (end is None or frame_info.index < end):
                    yield frame_info

            current = gop_start + self.gop_size

            if end is not None and current >= end:
                break

    def is_gop_boundary(self, frame_idx: int) -> bool:
        """Check if frame is at a GOP boundary.

        Args:
            frame_idx: Frame index.

        Returns:
            True if frame starts a new GOP.
        """
        return frame_idx % self.gop_size == 0

    def reset(self) -> None:
        """Reset GOP manager state."""
        self.frame_count = 0
        self.current_gop = 0
        self.reference_buffer.clear()
