from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class FileUrlInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["image", "video"]
    url: Annotated[str, Field(min_length=1, max_length=4096)]


class StreamInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: Annotated[str, Field(min_length=1, max_length=4096)]


class DesiredStateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    desired_state: Literal["cancelled", "stopped"]


class TaskError(BaseModel):
    code: str
    message: str


class Stats(BaseModel):
    processed_frames: Annotated[int, Field(ge=0)] = 0
    dropped_frames: Annotated[int, Field(ge=0)] = 0
    elapsed_seconds: Annotated[float, Field(ge=0)] = 0


class TaskResponse(BaseModel):
    id: str
    kind: Literal["image", "video", "stream"]
    state: Literal["queued", "processing", "succeeded", "failed", "cancelled", "connecting", "running", "recovering", "stopped"]
    desired_state: Literal["run", "cancelled", "stopped"]
    ready: bool
    created_at: str
    updated_at: str
    result_url: str | None
    playback_url: str | None
    error: TaskError | None
    stats: Stats

    @classmethod
    def from_record(cls, record, rtsp_base):
        fields = {key: record[key] for key in ("id", "kind", "state", "desired_state", "ready", "created_at", "updated_at", "error", "stats")}
        fields["result_url"] = f"/v1/tasks/{record['id']}/result" if record["state"] == "succeeded" else None
        fields["playback_url"] = f"{rtsp_base}/redacted/{record['id']}" if record["kind"] == "stream" else None
        return cls(**fields)
