from fastapi import APIRouter, HTTPException, Depends
from ...schemas.book import Book
from ....database import get_db
from psycopg2.extensions import connection
import random

router = APIRouter()

@router.get("/{user_id}", response_model=list[Book])
def get_recommendations(user_id: int, db: connection = Depends(get_db)):
    """
    Get recommendations for a user.
    Currently returns a random selection of popular books as a placeholder
    while LightGCN integration is being set up.
    """
    cursor = db.cursor()
    try:
        # Placeholder: Get 10 random books that are well rated
        # In a real scenario, this would call the LightGCN model inference
        query = """
            SELECT b.book_id, b.title, b.author, b.year_of_publication, b.publisher,
                   b.image_url_s, b.image_url_m, b.image_url_l
            FROM books b
            JOIN ratings r ON b.book_id = r.book_id
            GROUP BY b.book_id
            HAVING AVG(r.rating) > 7
            ORDER BY RANDOM()
            LIMIT 10
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        books = []
        for row in rows:
            books.append(Book(
                book_id=row[0],
                title=row[1],
                author=row[2],
                year_of_publication=row[3],
                publisher=row[4],
                image_url_s=row[5],
                image_url_m=row[6],
                image_url_l=row[7]
            ))

        return books

    finally:
        cursor.close()
