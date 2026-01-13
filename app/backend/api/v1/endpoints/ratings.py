from fastapi import APIRouter, HTTPException, Depends
from ...schemas.rating import Rating, RatingCreate
from ....database import get_db
from psycopg2.extensions import connection

router = APIRouter()

@router.post("/", response_model=Rating)
def submit_rating(rating: RatingCreate, db: connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        # Check if user exists
        cursor.execute("SELECT user_id FROM users WHERE user_id = %s", (rating.user_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="User not found")

        # Check if book exists
        cursor.execute("SELECT book_id FROM books WHERE book_id = %s", (rating.book_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Book not found")

        # Insert or Update rating
        query = """
            INSERT INTO ratings (user_id, book_id, rating)
            VALUES (%s, %s, %s)
            ON CONFLICT (user_id, book_id)
            DO UPDATE SET rating = EXCLUDED.rating
            RETURNING user_id, book_id, rating
        """
        cursor.execute(query, (rating.user_id, rating.book_id, rating.rating))
        db.commit()

        row = cursor.fetchone()
        return Rating(user_id=row[0], book_id=row[1], rating=row[2])

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
