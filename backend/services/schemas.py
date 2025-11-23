from pydantic import BaseModel, Field

class UserCreate(BaseModel):
    id: int = Field(..., description="ID do usuário (inteiro)")

class RatingCreate(BaseModel):
    user_id: int
    movie_id: int
    rating: float = Field(..., ge=0.5, le=5.0)
