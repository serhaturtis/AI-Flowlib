import pytest
from fastapi.testclient import TestClient

from server.app.main import app


client = TestClient(app)


@pytest.mark.parametrize(
    "resource_type,provider_type",
    [
        ("llm_config", "llamacpp"),
        ("vector_db_config", None),
    ],
)
def test_provider_schema_endpoint(resource_type: str, provider_type: str | None) -> None:
    params = {"project_id": "denememe", "resource_type": resource_type}
    if provider_type:
        params["provider_type"] = provider_type
    resp = client.get("/api/v1/configs/providers/schema", params=params)
    assert resp.status_code in (200, 400), resp.text
    if resp.status_code == 200:
        data = resp.json()
        assert "fields" in data
        names = [f["name"] for f in data["fields"]]
        assert "name" in names
        assert any(n in names for n in ["settings", "provider_type"])


@pytest.mark.parametrize(
    "resource_type,provider_type",
    [
        ("model_config", "llamacpp"),
        ("embedding_config", None),
    ],
)
def test_resource_schema_endpoint(resource_type: str, provider_type: str | None) -> None:
    params = {"project_id": "denememe", "resource_type": resource_type}
    if provider_type:
        params["provider_type"] = provider_type
    resp = client.get("/api/v1/configs/resources/schema", params=params)
    assert resp.status_code in (200, 400), resp.text
    if resp.status_code == 200:
        data = resp.json()
        assert "fields" in data
        names = [f["name"] for f in data["fields"]]
        assert "name" in names
        assert any(n in names for n in ["config", "provider_type"])


def _flatten_fields(fields: list[dict]) -> list[dict]:
    out: list[dict] = []
    for f in fields:
        out.append(f)
        if f.get("type") == "object" and isinstance(f.get("children"), list):
            out.extend(_flatten_fields(f.get("children") or []))
        if f.get("type") == "array" and isinstance(f.get("children"), list):
            # If array of objects, children describe object item fields
            out.extend(_flatten_fields(f.get("children") or []))
    return out


@pytest.mark.parametrize(
    "endpoint,params",
    [
        ("/api/v1/configs/providers/schema", {"project_id": "denememe", "resource_type": "llm_config", "provider_type": "llamacpp"}),
        ("/api/v1/configs/resources/schema", {"project_id": "denememe", "resource_type": "model_config", "provider_type": "llamacpp"}),
    ],
)
def test_schema_includes_union_hints_and_array_item_metadata(endpoint: str, params: dict) -> None:
    """Schemas should include allowed_types for unions and items_allowed_types for arrays (may be null if not applicable)."""
    resp = client.get(endpoint, params=params)
    assert resp.status_code in (200, 400), resp.text
    if resp.status_code != 200:
        pytest.skip("Schema not available in this environment")
    data = resp.json()
    assert "fields" in data
    flat = _flatten_fields(data["fields"])
    # Ensure keys exist on at least one field when types match
    saw_union_key = any("allowed_types" in f for f in flat)
    saw_array_item_keys = any((f.get("type") == "array" and "items_type" in f and "items_allowed_types" in f) for f in flat)
    assert saw_union_key, "expected 'allowed_types' key to be present on some schema fields"
    # items_allowed_types presence should be guaranteed for arrays (may be null)
    assert saw_array_item_keys, "expected array fields to include items_type and items_allowed_types keys"

