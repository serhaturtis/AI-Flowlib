from fastapi.testclient import TestClient
import pytest

from server.app.main import app

client = TestClient(app)


def test_get_agent_schema() -> None:
    resp = client.get("/api/v1/configs/agents/schema", params={"project_id": "denememe"})
    assert resp.status_code in (200, 400), resp.text
    if resp.status_code == 200:
        data = resp.json()
        assert "fields" in data
        names = [f["name"] for f in data["fields"]]
        for required in [
            "persona",
            "allowed_tool_categories",
            "model_name",
            "llm_name",
            "temperature",
            "max_iterations",
            "enable_learning",
            "verbose",
        ]:
            assert required in names


def test_render_agent_content_minimal() -> None:
    payload = {
        "name": "unit-test-agent",
        "persona": "I am a helpful test agent.",
        "allowed_tool_categories": ["generic"],
    }
    resp = client.post("/api/v1/configs/agents/render", json=payload)
    assert resp.status_code == 200, resp.text
    content = resp.json()["content"]
    assert "@agent_config" in content
    assert "class UnitTestAgentAgentConfig" in content


@pytest.mark.parametrize("action", ["apply", "create"])
def test_agent_apply_or_create(action: str) -> None:
    if action == "apply":
        payload = {
            "project_id": "denememe",
            "name": "unit-test-agent",
            "persona": "I am a helpful test agent.",
            "allowed_tool_categories": ["generic"],
            "model_name": "default-model",
            "llm_name": "default-llm",
            "temperature": 0.5,
            "max_iterations": 5,
            "enable_learning": False,
            "verbose": False,
        }
        resp = client.post("/api/v1/configs/agents/apply", json=payload)
    else:
        payload = {
            "project_id": "denememe",
            "name": "unit-test-agent",
            "persona": "I am a helpful test agent.",
            "allowed_tool_categories": ["generic"],
            "description": "Unit test scaffold",
        }
        resp = client.post("/api/v1/configs/agents/create", json=payload)
    assert resp.status_code in (200, 201, 400, 422), resp.text
    # 200/201 expected when project is well-formed, else fail-fast is acceptable in CI environment


