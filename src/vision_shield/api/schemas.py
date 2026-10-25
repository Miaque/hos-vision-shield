from datetime import timedelta, timezone
from typing import Annotated, Literal
from uuid import UUID
from urllib.parse import urlsplit

from pydantic import AwareDatetime, BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer, WithJsonSchema, field_validator


UTC8 = timezone(timedelta(hours=8))
ResponseDatetime = Annotated[
    AwareDatetime,
    PlainSerializer(lambda value: value.astimezone(UTC8).strftime("%Y-%m-%d %H:%M:%S"),
                    return_type=str, when_used="json"),
    WithJsonSchema({"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$",
                    "description": "UTC+8，格式为 %Y-%m-%d %H:%M:%S",
                    "examples": ["2026-10-22 09:00:00"]}, mode="serialization"),
]


class FileUrlInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["image", "video"]
    url: Annotated[str, Field(min_length=1, max_length=4096)]
    redaction_mode: Literal["solid", "mosaic"] = "solid"


class StreamInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: Annotated[str, Field(min_length=1, max_length=4096)]
    publish_url: Annotated[str | None, Field(min_length=1, max_length=4096,
        description="可选完整 RTSP/RTMP 推流地址；省略时使用默认 RTSP 输出")] = None
    playback_url: Annotated[str | None, Field(min_length=1, max_length=4096,
        description="可选完整 HTTP/HTTPS/RTSP/RTMP 播放地址；省略时沿用推流地址或默认公开 RTSP 地址")] = None
    redaction_mode: Literal["solid", "mosaic"] = "solid"
    max_bitrate: Annotated[int | None, Field(default=None, ge=1,
        description="可选实时编码码率上限，单位 bit/s；省略时不设置码率上限")] = None

    @field_validator('playback_url')
    @classmethod
    def validate_playback_url(cls, value):
        if value is not None:
            url = urlsplit(value)
            if url.scheme not in {'http', 'https', 'rtsp', 'rtmp'} or not url.hostname or url.fragment:
                raise ValueError('Invalid playback URL')
        return value


class DesiredStateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    desired_state: Literal["cancelled", "stopped"]


class TaskError(BaseModel):
    code: str
    message: str


class StreamOutputSample(BaseModel):
    sequence: int = Field(ge=1)
    source_pts: str
    timeline_pts: str | None = None
    timestamp_repaired: bool = False
    encoded_bytes: int | None = Field(default=None, ge=0)
    encoded_vcl_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    received_monotonic: float | None
    encoded_monotonic: float
    latency_seconds: float | None


class StreamStats(BaseModel):
    attempt_id: str
    started_monotonic: float
    sampled_monotonic: float
    input_frames: int = Field(ge=0)
    encoded_frames: int = Field(ge=0)
    encoded_packets: int = Field(ge=0)
    unmatched_packets: int = Field(ge=0)
    missing_receive_frames: int = Field(ge=0)
    pending_encoder_frames: int = Field(ge=0)
    queue_frames: int = Field(ge=0)
    audio_queue_bytes: int = Field(ge=0)
    failed_frames: int = Field(ge=0)
    protected_frames: int = Field(ge=0)
    timestamp_repairs: int = Field(default=0, ge=0)
    # Redis Lua cjson round-trips an empty array as an empty object.
    recent_outputs: Annotated[list[StreamOutputSample], BeforeValidator(lambda value: [] if value == {} else value)] = Field(max_length=512)


class Stats(BaseModel):
    processed_frames: Annotated[int, Field(ge=0)] = 0
    dropped_frames: Annotated[int, Field(ge=0)] = 0
    elapsed_seconds: Annotated[float, Field(ge=0)] = 0
    stream: StreamStats | None = Field(default=None, exclude_if=lambda value: value is None)


class CpuUsage(BaseModel):
    scope: Literal['cgroup_v2']
    percent: float | None
    limit_cores: float | None
    unavailable_reason: str | None


class MemoryUsage(BaseModel):
    scope: Literal['cgroup_v2']
    used_bytes: int | None
    limit_bytes: int | None
    unavailable_reason: str | None


class GpuDevice(BaseModel):
    uuid: str
    name: str
    utilization_percent: float | None
    memory_used_bytes: int | None
    memory_total_bytes: int | None
    power_watts: float | None


class GpuUsage(BaseModel):
    scope: Literal['device']
    devices: list[GpuDevice]
    unavailable_reason: str | None


class ResourceSnapshot(BaseModel):
    sampled_at_unix: float
    sample_window_seconds: float | None
    cpu: CpuUsage
    memory: MemoryUsage
    gpu: GpuUsage


class WorkerResources(BaseModel):
    worker_id: str
    pool: Literal['files', 'streams']
    age_seconds: float | None
    stale: bool
    unavailable_reason: str | None
    snapshot: ResourceSnapshot | None


class ResourcesResponse(BaseModel):
    sampled_at_unix: float
    workers: list[WorkerResources]


class StreamHealth(BaseModel):
    """健康期限由服务端租约和实际输出计算，不能按查询时刻续期。"""

    observed_at: ResponseDatetime
    last_output_at: ResponseDatetime | None
    valid_until: ResponseDatetime | None
    lease_valid: bool
    generation: int


class TaskResponse(BaseModel):
    sequence: int = Field(default=1, ge=1)
    health: StreamHealth | None = None
    redaction_mode: Literal["solid", "mosaic"] = "solid"
    id: str
    kind: Literal["image", "video", "stream"]
    state: Literal["queued", "processing", "succeeded", "failed", "cancelled", "connecting", "running", "recovering", "stopped"]
    desired_state: Literal["run", "cancelled", "stopped"]
    ready: bool
    created_at: ResponseDatetime
    updated_at: ResponseDatetime
    result_url: str | None
    playback_url: str | None
    error: TaskError | None
    stats: Stats

    @classmethod
    def from_record(cls, record, rtsp_base):
        fields = {key: record[key] for key in ("id", "kind", "state", "desired_state", "ready", "created_at", "updated_at", "error", "stats")}
        fields["redaction_mode"] = record.get("redaction_mode", "solid")
        fields["sequence"] = record.get("sequence", 1)
        fields["health"] = record.get("health")
        fields["result_url"] = f"/v1/tasks/{record['id']}/result" if record["state"] == "succeeded" else None
        fields["playback_url"] = (record.get('playback_url') or record.get('publish_url') or
                                  f"{rtsp_base}/redacted/{record['id']}") if record["kind"] == "stream" else None
        return cls(**fields)


class TaskListResponse(BaseModel):
    current: list[TaskResponse]
    queued: list[TaskResponse]


class TaskQueryInput(BaseModel):
    """限制单次查询工作量，UUID 验证发生在读取 Redis 之前。"""

    model_config = ConfigDict(extra="forbid")
    task_ids: list[UUID] = Field(min_length=1, max_length=100)


class TaskQueryResponse(BaseModel):
    items: list[TaskResponse]
    missing_ids: list[UUID]
