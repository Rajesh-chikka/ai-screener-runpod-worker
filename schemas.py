from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImageUrl(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str


class ContentItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["image_url", "text"]
    image_url: ImageUrl | None = None
    text: str | None = None

    @model_validator(mode="after")
    def validate_content(self) -> "ContentItem":
        if self.type == "image_url" and self.image_url is None:
            raise ValueError("image_url content requires image_url")
        if self.type == "text" and not (self.text or "").strip():
            raise ValueError("text content requires non-empty text")
        return self


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user"]
    content: list[ContentItem]


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=2048, gt=0)
    temperature: float = Field(default=0.1, ge=0)


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: AssistantMessage
    finish_reason: Literal["stop"] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    choices: list[ChatCompletionChoice]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    device: str
    loaded_models: list[str]
