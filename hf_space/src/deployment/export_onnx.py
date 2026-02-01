"""ONNX export utilities for AlphaGalerkin models.

This module provides utilities for exporting PyTorch models to ONNX
format with support for various optimization and export methods.

Features:
    - Multiple export methods (trace, script, dynamo)
    - Dynamic shape support for resolution independence
    - Automatic optimization
    - Metadata embedding
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
import torch
from torch import Tensor, nn

from src.deployment.config import ExportConfig

if TYPE_CHECKING:
    from src.modeling.model import AlphaGalerkinModel

logger = structlog.get_logger(__name__)


class ONNXExporter:
    """Exports PyTorch models to ONNX format.

    Handles the complexity of ONNX export including:
    - Dynamic axis configuration
    - Operator compatibility
    - Optimization passes
    - Metadata embedding

    Attributes:
        config: Export configuration.

    """

    def __init__(self, config: ExportConfig | None = None) -> None:
        """Initialize exporter.

        Args:
            config: Export configuration. Uses defaults if None.

        """
        self.config = config or ExportConfig()
        self._logger = structlog.get_logger(__name__).bind(
            opset_version=self.config.opset_version,
            export_method=self.config.export_method,
        )

    def export(
        self,
        model: AlphaGalerkinModel | nn.Module,
        sample_input: Tensor | tuple[Tensor, ...] | dict[str, Tensor],
        output_path: str | Path,
        **kwargs: Any,
    ) -> Path:
        """Export model to ONNX format.

        Args:
            model: PyTorch model to export.
            sample_input: Sample input for tracing.
            output_path: Path for output ONNX file.
            **kwargs: Additional torch.onnx.export arguments.

        Returns:
            Path to exported ONNX model.

        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure model is in eval mode
        model.eval()

        # Prepare input
        if isinstance(sample_input, dict):
            input_tensors = tuple(sample_input.values())
        elif isinstance(sample_input, Tensor):
            input_tensors = (sample_input,)
        else:
            input_tensors = sample_input

        self._logger.info(
            "starting_export",
            output_path=str(output_path),
            input_shapes=[tuple(t.shape) for t in input_tensors],
        )

        # Select export method
        if self.config.export_method == "dynamo":
            self._export_dynamo(model, input_tensors, output_path, **kwargs)
        elif self.config.export_method == "script":
            self._export_script(model, input_tensors, output_path, **kwargs)
        else:
            self._export_trace(model, input_tensors, output_path, **kwargs)

        # Optimize if requested
        if self.config.optimization_level != "none":
            self._optimize(output_path)

        # Add metadata
        self._add_metadata(output_path)

        self._logger.info(
            "export_completed",
            output_path=str(output_path),
            file_size_mb=output_path.stat().st_size / (1024 * 1024),
        )

        return output_path

    def _export_trace(
        self,
        model: nn.Module,
        input_tensors: tuple[Tensor, ...],
        output_path: Path,
        **kwargs: Any,
    ) -> None:
        """Export using torch.jit.trace.

        Args:
            model: Model to export.
            input_tensors: Input tensors for tracing.
            output_path: Output path.
            **kwargs: Additional arguments.

        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

            torch.onnx.export(
                model,
                input_tensors,
                str(output_path),
                input_names=self.config.input_names,
                output_names=self.config.output_names,
                dynamic_axes=self.config.dynamic_axes,
                opset_version=self.config.opset_version,
                do_constant_folding=self.config.do_constant_folding,
                export_params=self.config.export_params,
                verbose=self.config.verbose,
                **kwargs,
            )

    def _export_script(
        self,
        model: nn.Module,
        input_tensors: tuple[Tensor, ...],
        output_path: Path,
        **kwargs: Any,
    ) -> None:
        """Export using torch.jit.script.

        Args:
            model: Model to export.
            input_tensors: Input tensors for shape inference.
            output_path: Output path.
            **kwargs: Additional arguments.

        """
        scripted_model = torch.jit.script(model)

        torch.onnx.export(
            scripted_model,
            input_tensors,
            str(output_path),
            input_names=self.config.input_names,
            output_names=self.config.output_names,
            dynamic_axes=self.config.dynamic_axes,
            opset_version=self.config.opset_version,
            do_constant_folding=self.config.do_constant_folding,
            export_params=self.config.export_params,
            verbose=self.config.verbose,
            **kwargs,
        )

    def _export_dynamo(
        self,
        model: nn.Module,
        input_tensors: tuple[Tensor, ...],
        output_path: Path,
        **kwargs: Any,
    ) -> None:
        """Export using torch.onnx.dynamo_export (PyTorch 2.0+).

        Args:
            model: Model to export.
            input_tensors: Input tensors for tracing.
            output_path: Output path.
            **kwargs: Additional arguments.

        """
        try:
            export_output = torch.onnx.dynamo_export(
                model,
                *input_tensors,
            )
            export_output.save(str(output_path))
        except AttributeError:
            # Fall back to trace if dynamo_export not available
            self._logger.warning(
                "dynamo_export_not_available",
                message="Falling back to trace export",
            )
            self._export_trace(model, input_tensors, output_path, **kwargs)

    def _optimize(self, model_path: Path) -> None:
        """Apply ONNX optimizations.

        Args:
            model_path: Path to ONNX model.

        """
        try:
            import onnx
            from onnx import optimizer

            model = onnx.load(str(model_path))

            # Apply optimization passes based on level
            passes = []

            if self.config.optimization_level in ("basic", "extended", "full"):
                passes.extend([
                    "eliminate_identity",
                    "eliminate_nop_dropout",
                    "eliminate_nop_pad",
                    "eliminate_nop_transpose",
                    "eliminate_unused_initializer",
                    "fuse_bn_into_conv",
                    "fuse_consecutive_squeezes",
                    "fuse_consecutive_transposes",
                ])

            if self.config.optimization_level in ("extended", "full"):
                passes.extend([
                    "fuse_add_bias_into_conv",
                    "fuse_matmul_add_bias_into_gemm",
                    "fuse_transpose_into_gemm",
                ])

            if self.config.optimization_level == "full":
                passes.extend([
                    "fuse_consecutive_concats",
                    "fuse_consecutive_log_softmax",
                    "fuse_consecutive_reduce_unsqueeze",
                ])

            if passes:
                optimized_model = optimizer.optimize(model, passes)
                onnx.save(optimized_model, str(model_path))
                self._logger.debug("optimization_applied", passes=passes)

        except ImportError:
            self._logger.warning(
                "onnx_optimizer_not_available",
                message="Install onnx package for optimization",
            )
        except Exception as e:
            self._logger.warning(
                "optimization_failed",
                error=str(e),
            )

    def _add_metadata(self, model_path: Path) -> None:
        """Add metadata to ONNX model.

        Args:
            model_path: Path to ONNX model.

        """
        try:
            import onnx

            model = onnx.load(str(model_path))

            # Add model info
            model.doc_string = f"AlphaGalerkin {self.config.model_version}"

            # Add metadata key-value pairs
            meta_info = [
                ("model_name", self.config.model_name),
                ("model_version", self.config.model_version),
                ("export_method", self.config.export_method),
                ("opset_version", str(self.config.opset_version)),
            ]

            for key, value in meta_info:
                meta = model.metadata_props.add()
                meta.key = key
                meta.value = value

            onnx.save(model, str(model_path))

        except ImportError:
            self._logger.debug("onnx_not_available_for_metadata")

    def create_sample_input(
        self,
        batch_size: int = 1,
        board_size: int = 19,
        channels: int = 17,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Create a sample input tensor for export.

        Args:
            batch_size: Batch size.
            board_size: Board size (height = width).
            channels: Number of input channels.
            device: Device for the tensor.

        Returns:
            Sample input tensor.

        """
        return torch.randn(
            batch_size,
            channels,
            board_size,
            board_size,
            device=device,
        )

    def verify_export(
        self,
        model_path: str | Path,
    ) -> bool:
        """Verify the exported ONNX model.

        Args:
            model_path: Path to ONNX model.

        Returns:
            True if model is valid.

        """
        try:
            import onnx

            model = onnx.load(str(model_path))
            onnx.checker.check_model(model)

            self._logger.info("model_verified", path=str(model_path))
            return True

        except ImportError:
            self._logger.warning(
                "onnx_not_available",
                message="Cannot verify without onnx package",
            )
            return False
        except Exception as e:
            self._logger.error(
                "verification_failed",
                error=str(e),
            )
            return False

    def get_model_info(self, model_path: str | Path) -> dict[str, Any]:
        """Get information about an ONNX model.

        Args:
            model_path: Path to ONNX model.

        Returns:
            Dictionary with model information.

        """
        try:
            import onnx

            model = onnx.load(str(model_path))
            graph = model.graph

            # Get input info
            inputs = []
            for inp in graph.input:
                shape = []
                for dim in inp.type.tensor_type.shape.dim:
                    if dim.dim_param:
                        shape.append(dim.dim_param)
                    else:
                        shape.append(dim.dim_value)
                inputs.append({"name": inp.name, "shape": shape})

            # Get output info
            outputs = []
            for out in graph.output:
                shape = []
                for dim in out.type.tensor_type.shape.dim:
                    if dim.dim_param:
                        shape.append(dim.dim_param)
                    else:
                        shape.append(dim.dim_value)
                outputs.append({"name": out.name, "shape": shape})

            # Get metadata
            metadata = {}
            for prop in model.metadata_props:
                metadata[prop.key] = prop.value

            return {
                "inputs": inputs,
                "outputs": outputs,
                "opset_version": model.opset_import[0].version if model.opset_import else None,
                "ir_version": model.ir_version,
                "producer_name": model.producer_name,
                "producer_version": model.producer_version,
                "doc_string": model.doc_string,
                "metadata": metadata,
                "num_nodes": len(graph.node),
                "num_initializers": len(graph.initializer),
            }

        except ImportError:
            return {"error": "onnx package not available"}
        except Exception as e:
            return {"error": str(e)}


def create_exporter(
    opset_version: int = 17,
    **kwargs: Any,
) -> ONNXExporter:
    """Factory function to create ONNX exporter.

    Args:
        opset_version: ONNX opset version.
        **kwargs: Additional configuration options.

    Returns:
        Configured ONNXExporter instance.

    """
    config = ExportConfig(opset_version=opset_version, **kwargs)
    return ONNXExporter(config)


def export_model(
    model: nn.Module,
    output_path: str | Path,
    sample_input: Tensor | None = None,
    board_size: int = 19,
    input_channels: int = 17,
    **kwargs: Any,
) -> Path:
    """Convenience function to export a model.

    Args:
        model: PyTorch model to export.
        output_path: Path for output ONNX file.
        sample_input: Optional sample input tensor.
        board_size: Board size for sample input.
        input_channels: Input channels for sample input.
        **kwargs: Additional export options.

    Returns:
        Path to exported model.

    """
    exporter = create_exporter(**kwargs)

    if sample_input is None:
        sample_input = exporter.create_sample_input(
            batch_size=1,
            board_size=board_size,
            channels=input_channels,
        )

    return exporter.export(model, sample_input, output_path)
