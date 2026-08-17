"""Smoke tests for Pydantic configuration schemas.

This ensures that all configuration models can be instantiated with their default
values and that they properly serialize and deserialize.
"""

import pytest
from pydantic import BaseModel

# We import inside try/except or tests to avoid failing the whole test suite
# if there's an import error (which should be caught by test_import_smoke.py).


def get_config_schemas() -> list[tuple[str, type[BaseModel]]]:
    """Helper to safely fetch schema classes for parametrization."""
    schemas: list[tuple[str, type[BaseModel]]] = []

    try:
        from config.schemas import OperatorConfig

        schemas.append(("OperatorConfig", OperatorConfig))
    except ImportError:
        pass

    try:
        from src.alphagalerkin.solver import AlphaGalerkinConfig

        schemas.append(("AlphaGalerkinConfig", AlphaGalerkinConfig))
    except ImportError:
        pass

    try:
        from src.pde.config import PDEConfig, PDEGameConfig

        schemas.append(("PDEConfig", PDEConfig))
        schemas.append(("PDEGameConfig", PDEGameConfig))
    except ImportError:
        pass

    return schemas


SCHEMAS = get_config_schemas()


@pytest.mark.parametrize("schema_name, schema_cls", SCHEMAS, ids=[name for name, _ in SCHEMAS])
def test_config_schema_defaults_and_roundtrip(
    schema_name: str, schema_cls: type[BaseModel]
) -> None:
    """Test that the config schema can be instantiated with default arguments.

    Ensures that it can survive a round-trip serialization to/from a dictionary.
    """
    # 1. Instantiate with defaults
    try:
        instance = schema_cls()
    except Exception as e:
        # Some models have required fields. We'll provide minimal mock data for them.
        try:
            if schema_name == "PDEConfig":
                instance = schema_cls(name="test_pde", pde_type="heat")
            elif schema_name == "PDEGameConfig":
                from src.pde.config import PDEConfig

                instance = schema_cls(
                    name="test_game", pde_config=PDEConfig(name="test_pde", pde_type="heat")
                )
            else:
                pytest.fail(f"Failed to instantiate {schema_name} with defaults: {e}")
        except Exception as inner_e:
            pytest.fail(f"Failed to instantiate {schema_name} even with mocks: {inner_e}")

    # 2. Serialize to dictionary
    try:
        model_dict = instance.model_dump()
    except AttributeError:
        # Fallback for older pydantic versions if needed, though project uses v2
        model_dict = instance.dict()

    assert isinstance(model_dict, dict)

    # 3. Deserialize back to model
    try:
        roundtrip_instance = schema_cls.model_validate(model_dict)
    except AttributeError:
        roundtrip_instance = schema_cls.parse_obj(model_dict)

    # 4. Ensure equality
    assert instance == roundtrip_instance
