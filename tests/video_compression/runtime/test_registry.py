"""Tests for the runtime registry and BaseDecoderRuntime ABC."""

from __future__ import annotations

import pytest

from src.video_compression.runtime import (
    BaseDecoderRuntime,
    RuntimeRegistry,
    create_runtime,
    register_runtime,
)
from src.video_compression.runtime.metadata import CompiledArtifactMetadata
from src.video_compression.runtime.protocol import DecoderRuntimeContext

# -------------------------------------------------------- BaseDecoderRuntime


class TestBaseDecoderRuntime:
    """ABC contract tests using a minimal subclass."""

    class _MinimalRuntime(BaseDecoderRuntime):
        runtime_name = "minimal-test-runtime"

        def prepare(self, *, ctx: DecoderRuntimeContext) -> None:
            self._prepared_ctx = ctx
            self._metadata = CompiledArtifactMetadata(
                name="m",
                runtime_name=self.runtime_name,
                backend="pytorch",
                precision=ctx.dtype,
                model_hash=ctx.model_hash,
                device_label=ctx.device,
                batch_size=ctx.batch_size,
                latent_channels=ctx.latent_channels,
                latent_height=ctx.latent_height,
                latent_width=ctx.latent_width,
            )

        def decode(self, latent):  # type: ignore[no-untyped-def]
            return latent

    def test_name_property(self, tiny_context) -> None:
        rt = self._MinimalRuntime()
        assert rt.name == "minimal-test-runtime"
        rt.prepare(ctx=tiny_context)
        meta = rt.metadata
        assert meta is not None
        assert meta.runtime_name == "minimal-test-runtime"

    def test_metadata_none_before_prepare(self) -> None:
        rt = self._MinimalRuntime()
        assert rt.metadata is None

    def test_teardown_clears_metadata(self, tiny_context) -> None:
        rt = self._MinimalRuntime()
        rt.prepare(ctx=tiny_context)
        assert rt.metadata is not None
        rt.teardown()
        assert rt.metadata is None

    def test_unset_runtime_name_raises(self) -> None:
        class NoName(BaseDecoderRuntime):
            # Deliberately leaves runtime_name = "" (the ABC default)
            def prepare(self, *, ctx: DecoderRuntimeContext) -> None:
                pass

            def decode(self, latent):  # type: ignore[no-untyped-def]
                return latent

        rt = NoName()
        with pytest.raises(RuntimeError, match="runtime_name not set"):
            _ = rt.name


# ------------------------------------------------------------ RuntimeRegistry


class TestRuntimeRegistry:
    def test_eager_runtime_registered_at_import(self) -> None:
        from src.video_compression.runtime import (
            PYTORCH_EAGER_RUNTIME_NAME,
            PyTorchEagerRuntime,
        )

        assert RuntimeRegistry().is_registered(PYTORCH_EAGER_RUNTIME_NAME)
        cls = RuntimeRegistry().get(PYTORCH_EAGER_RUNTIME_NAME)
        assert cls is PyTorchEagerRuntime

    def test_list_items_sorted(self) -> None:
        items = RuntimeRegistry().list_items()
        assert items == sorted(items)
        assert "pytorch-eager" in items

    def test_create_runtime_returns_instance(self) -> None:
        rt = create_runtime("pytorch-eager")
        assert isinstance(rt, BaseDecoderRuntime)
        assert rt.name == "pytorch-eager"

    def test_create_runtime_unknown_name_raises_with_available(self) -> None:
        with pytest.raises(KeyError, match="not registered"):
            create_runtime("non-existent-runtime")

    def test_create_runtime_error_lists_available(self) -> None:
        try:
            create_runtime("does-not-exist")
        except KeyError as exc:
            # The available list must surface in the message so the
            # caller can debug a typo without grep'ing the source.
            assert "pytorch-eager" in str(exc)
        else:  # pragma: no cover
            pytest.fail("expected KeyError")


class TestRegisterDecorator:
    """The decorator path is the supported registration mechanism."""

    def test_decorator_registers_class(self) -> None:
        @register_runtime("decorator-test-runtime")
        class _ToyRuntime(BaseDecoderRuntime):
            runtime_name = "decorator-test-runtime"

            def prepare(self, *, ctx: DecoderRuntimeContext) -> None:
                self._metadata = CompiledArtifactMetadata(
                    name="m",
                    runtime_name=self.runtime_name,
                    backend="pytorch",
                    precision=ctx.dtype,
                    model_hash=ctx.model_hash,
                    device_label=ctx.device,
                    batch_size=ctx.batch_size,
                    latent_channels=ctx.latent_channels,
                    latent_height=ctx.latent_height,
                    latent_width=ctx.latent_width,
                )

            def decode(self, latent):  # type: ignore[no-untyped-def]
                return latent

        try:
            assert RuntimeRegistry().is_registered("decorator-test-runtime")
            cls = RuntimeRegistry().get("decorator-test-runtime")
            assert cls is _ToyRuntime
        finally:
            # Clean up so test isolation is preserved across tests
            RuntimeRegistry()._items.pop("decorator-test-runtime", None)

    def test_double_registration_raises(self) -> None:
        @register_runtime("double-register-test")
        class _A(BaseDecoderRuntime):
            runtime_name = "double-register-test"

            def prepare(self, *, ctx: DecoderRuntimeContext) -> None:
                pass

            def decode(self, latent):  # type: ignore[no-untyped-def]
                return latent

        try:
            with pytest.raises(ValueError, match="already registered"):

                @register_runtime("double-register-test")
                class _B(BaseDecoderRuntime):  # noqa: F841
                    runtime_name = "double-register-test"

                    def prepare(self, *, ctx: DecoderRuntimeContext) -> None:
                        pass

                    def decode(self, latent):  # type: ignore[no-untyped-def]
                        return latent
        finally:
            RuntimeRegistry()._items.pop("double-register-test", None)
