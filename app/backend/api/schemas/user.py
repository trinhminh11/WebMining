from pydantic import BaseModel
from typing import Optional

class UserLogin(BaseModel):
    user_id: int # User ID is used as username
    password: str

class UserRegister(BaseModel):
    user_id: int
    password: str
    location: Optional[str] = None
    age: Optional[int] = None

class User(BaseModel):
    user_id: int
    location: Optional[str] = None
    age: Optional[int] = None
