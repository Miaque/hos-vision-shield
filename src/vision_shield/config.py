"""Validated deployment settings; no network or model loading at import time."""

import ipaddress
import re
from pathlib import Path
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PositiveSeconds = Annotated[float, Field(gt=0, allow_inf_nan=False)]
PositiveCount = Annotated[int, Field(gt=0)]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", hide_input_in_errors=True
    )

    debug: bool = False
    redis_url: SecretStr = SecretStr("redis://127.0.0.1:16379/0")
    media_root: Path = Path("data")
    detection_backend: Literal["quality", "quality_yolo", "cpu", "egoblur"] = "quality"
    inference_device: Literal["cpu", "cuda"] = "cpu"
    face_model_path: Path = Path("models/yunet_face.onnx")
    plate_model_path: Path = Path("models/hyperlpr_plate.onnx")
    egoblur_face_model_path: Path = Path("models/face.jit")
    egoblur_plate_model_path: Path = Path("models/license_plate.jit")
    detection_max_side: Annotated[int, Field(ge=320, le=1920)] = 1280
    detection_box_padding: Annotated[float, Field(ge=0, le=0.5, allow_inf_nan=False)] = 0.15
    mediamtx_control_url: str = "http://127.0.0.1:9997"
    rtsp_public_base_url: str = "rtsp://127.0.0.1:8554"
    rtsp_publish_base_url: str | None = None
    source_allowlist: list[str] = Field(default_factory=list)

    file_task_limit: PositiveCount = 100
    stream_task_limit: PositiveCount = 100
    message_limit: PositiveCount = 100
    notification_limit: PositiveCount = 1000
    media_disk_high_watermark: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)] = 0.95
    notification_targets: list[Annotated[str, Field(pattern=r"^[a-z][a-z0-9-]*(?:\.[a-z][a-z0-9-]*)*$",
                                                     max_length=64)]] = Field(default_factory=list)
    vision_shield_notification_amqp_url: SecretStr = SecretStr("")
    frame_buffer_size: PositiveCount = 2
    stream_output_pacing: bool = True
    stream_diagnostics: bool = True
    audio_buffer_seconds: PositiveSeconds = 16
    audio_buffer_bytes: PositiveCount = 4 * 1024 * 1024
    max_image_bytes: PositiveCount = 20 * 1024 * 1024
    max_image_pixels: PositiveCount = 40_000_000
    max_video_bytes: PositiveCount = 2 * 1024 * 1024 * 1024
    max_video_seconds: PositiveSeconds = 3600
    max_video_width: PositiveCount = 3840
    max_video_height: PositiveCount = 2160

    http_connect_timeout: PositiveSeconds = 5
    http_read_timeout: PositiveSeconds = 30
    download_timeout: PositiveSeconds = 1800
    upload_timeout: PositiveSeconds = 1800
    rtsp_connect_timeout: PositiveSeconds = 10
    rtsp_read_timeout: PositiveSeconds = 10
    heartbeat_seconds: PositiveSeconds = 5
    lease_seconds: PositiveSeconds = 20
    reconcile_seconds: PositiveSeconds = 5
    process_stop_timeout: PositiveSeconds = 10

    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(cls, value: SecretStr) -> SecretStr:
        url = urlsplit(value.get_secret_value())
        if url.scheme not in {"redis", "rediss"} or not url.hostname:
            raise ValueError("REDIS_URL must use redis:// or rediss://")
        return value

    @field_validator("vision_shield_notification_amqp_url")
    @classmethod
    def validate_amqp_url(cls, value: SecretStr) -> SecretStr:
        """独立通知连接只允许 AMQP，凭据不进入错误信息。"""
        raw = value.get_secret_value()
        if raw and (urlsplit(raw).scheme not in {"amqp", "amqps"} or not urlsplit(raw).hostname):
            raise ValueError("通知连接必须使用 amqp:// 或 amqps://")
        return value

    @field_validator("mediamtx_control_url", "rtsp_public_base_url", "rtsp_publish_base_url")
    @classmethod
    def validate_service_url(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        url = urlsplit(value)
        schemes = {"http", "https"} if info.field_name == "mediamtx_control_url" else {"rtsp"}
        if (
            url.scheme not in schemes
            or not url.hostname
            or url.username is not None
            or url.query
            or url.fragment
        ):
            raise ValueError("Invalid service URL (credentials and query are not allowed)")
        return value.rstrip("/")

    @field_validator("source_allowlist")
    @classmethod
    def validate_allowlist(cls, values: list[str]) -> list[str]:
        for value in values:
            try:
                network = ipaddress.ip_network(value)
            except ValueError:
                if not re.fullmatch(r"[a-zA-Z0-9]+(?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?", value):
                    raise ValueError("Allowlist requires exact hosts, IP addresses or CIDRs")
            else:
                if network.prefixlen == 0:
                    raise ValueError("An unrestricted network is not an allowlist")
        return values

    @model_validator(mode="after")
    def validate_lease(self) -> Self:
        if self.notification_targets and not self.vision_shield_notification_amqp_url.get_secret_value():
            raise ValueError("AMQP 通知必须配置 VISION_SHIELD_NOTIFICATION_AMQP_URL")
        if self.detection_backend == "quality_yolo" and self.inference_device != "cuda":
            raise ValueError("quality_yolo requires CUDA inference")
        if self.detection_backend == "cpu" and self.inference_device == "cuda":
            raise ValueError("轻量 cpu 检测策略不支持 CUDA，请选择 quality 或 egoblur")
        if self.heartbeat_seconds >= self.lease_seconds:
            raise ValueError("HEARTBEAT_SECONDS must be less than LEASE_SECONDS")
        return self
