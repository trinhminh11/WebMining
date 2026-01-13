from pydantic import BaseModel
from typing import Optional

class BookBase(BaseModel):
    book_id: str
    title: str
    author: Optional[str] = None
    year_of_publication: Optional[int] = None
    publisher: Optional[str] = None
    image_url_s: Optional[str] = None
    image_url_m: Optional[str] = None
    image_url_l: Optional[str] = None
    average_rating: Optional[float] = None
    rating_count: Optional[int] = 0

class Book(BookBase):
    pass

class BookList(BaseModel):
    books: list[Book]
    total_count: int
    page: int
    limit: int
