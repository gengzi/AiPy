from pydantic import BaseModel

class UserCreate(BaseModel):
    """
    DTO for user creation
    """
    username : str;
    email : str;
    age : int;

class UserResponse(UserCreate):
    id:int
    class Config:
        from_attributes = True
        orm_mode = True

