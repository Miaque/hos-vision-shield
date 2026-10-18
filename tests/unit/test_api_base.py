import httpx

from vision_shield.config import Settings
from vision_shield.main import create_app


async def test_liveness_is_public_and_missing_dependencies_are_not_ready():
    app = create_app(Settings(_env_file=None, redis_url="redis://127.0.0.1:1/0"))
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.get("/health/live")).json() == {"status": "alive"}
        response = await client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "dependency_unavailable"
