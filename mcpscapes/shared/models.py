from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class ServerRegistration(BaseModel):
    id: str
    name: str
    description: str
    url: str
    centroid: list[float] | None = None
    registered_at: datetime = Field(default_factory=datetime.utcnow)


class MemoryNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    domain_weights: dict[str, float]
    embedding: list[float] | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class RouteResult(BaseModel):
    server_id: str
    score: float
    connection_info: dict
