from pydantic import BaseModel

class RatingCreate(BaseModel):
    user_id: int
    book_id: str
    rating: int

class Rating(BaseModel):
    user_id: int
    book_id: str
    rating: int
