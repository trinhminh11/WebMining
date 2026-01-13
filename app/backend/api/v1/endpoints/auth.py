from fastapi import APIRouter, HTTPException, Depends
from ...schemas.user import UserLogin, UserRegister, User
from ....database import get_db, get_db_connection
from psycopg2.extensions import connection

router = APIRouter()

@router.post("/login", response_model=User)
def login(user_in: UserLogin, db: connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        query = "SELECT user_id, location, age, password FROM users WHERE user_id = %s"
        cursor.execute(query, (user_in.user_id,))
        result = cursor.fetchone()

        if not result:
            raise HTTPException(status_code=400, detail="Incorrect user_id or password")

        user_id, location, age, db_password = result

        # Simple password check as requested
        if str(db_password) != user_in.password:
            raise HTTPException(status_code=400, detail="Incorrect user_id or password")

        return User(user_id=user_id, location=location, age=age)

    finally:
        cursor.close()

@router.post("/register", response_model=User)
def register(user_in: UserRegister, db: connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        # Check if user exists
        check_query = "SELECT user_id FROM users WHERE user_id = %s"
        cursor.execute(check_query, (user_in.user_id,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="User already exists")

        # Insert new user
        insert_query = """
            INSERT INTO users (user_id, password, location, age)
            VALUES (%s, %s, %s, %s)
            RETURNING user_id, location, age
        """
        cursor.execute(insert_query, (user_in.user_id, user_in.password, user_in.location, user_in.age))
        db.commit()

        new_user = cursor.fetchone()
        return User(user_id=new_user[0], location=new_user[1], age=new_user[2])

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
