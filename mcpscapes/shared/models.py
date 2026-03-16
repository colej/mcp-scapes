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
