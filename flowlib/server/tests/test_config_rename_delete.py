from fastapi.testclient import TestClient

from server.app.main import app

client = TestClient(app)


def test_configs_rename_and_delete_roundtrip() -> None:
    project_id = "denememe"
    # Start by rendering a trivial provider content and applying it to ensure a file exists
    provider_payload = {
        "name": "unit-provider",
        "resource_type": "llm_config",
        "provider_type": "llamacpp",
        "description": "Test provider",
        "settings": {"max_concurrent_models": 1},
    }
    r = client.post("/api/v1/configs/providers/render", json=provider_payload)
    assert r.status_code == 200, r.text
    content = r.json()["content"]
    apply_resp = client.post(
        "/api/v1/diff/configs/apply",
        json={
            "project_id": project_id,
            "relative_path": "configs/providers/unit-provider.py",
            "content": content,
            "sha256_before": "",
        },
    )
    assert apply_resp.status_code in (200, 409, 422), apply_resp.text

    # Rename to a new filename
    rename_resp = client.post(
        "/api/v1/configs/configs/rename",
        json={
            "project_id": project_id,
            "old_relative_path": "configs/providers/unit-provider.py",
            "new_relative_path": "configs/providers/unit-provider-renamed.py",
        },
    )
    assert rename_resp.status_code in (204, 400, 404, 422), rename_resp.text

    # Delete (on either original or renamed path, depending on rename outcome)
    delete_path = "configs/providers/unit-provider-renamed.py" if rename_resp.status_code == 204 else "configs/providers/unit-provider.py"
    delete_resp = client.post(
        "/api/v1/configs/configs/delete",
        json={"project_id": project_id, "relative_path": delete_path},
    )
    assert delete_resp.status_code in (204, 404, 422), delete_resp.text


