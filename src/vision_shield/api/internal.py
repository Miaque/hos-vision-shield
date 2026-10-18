from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field

from vision_shield.media.publisher import Publisher

router = APIRouter()


class MediaCheck(BaseModel):
    user: str = Field(default="", max_length=128)
    password: str = Field(default="", max_length=128)
    action: str = Field(max_length=32)
    path: str = Field(max_length=256)
    id: str = Field(pattern=r"^[0-9a-fA-F-]{36}$")


@router.post("/internal/media/publish-check", include_in_schema=False)
async def check(payload: MediaCheck, request: Request):
    approved = await Publisher(request.app.state.store, request.app.state.settings).authorize(payload.model_dump())
    return Response(status_code=204 if approved else 401)
