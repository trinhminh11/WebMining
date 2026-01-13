from fastapi import APIRouter, HTTPException, Depends, Query
from ...schemas.book import Book, BookList
from ....database import get_db
from psycopg2.extensions import connection

router = APIRouter()

@router.get("/", response_model=BookList)
def get_books(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    db: connection = Depends(get_db)
):
    offset = (page - 1) * limit
    cursor = db.cursor()
    try:
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM books")
        total_count = cursor.fetchone()[0]

        # Get books with weighted rating
        # Weighted Rating (WR) = (v / (v+m)) * R + (m / (v+m)) * C
        # v = vote_count
        # m = minimum votes (e.g. 5)
        # R = average_rating
        # C = mean vote across the whole report
        m = 5

        # Calculate C (global mean)
        cursor.execute("SELECT AVG(rating) FROM ratings")
        result = cursor.fetchone()
        C = float(result[0]) if result and result[0] else 7.0

        query = """
            SELECT b.book_id, b.title, b.author, b.year_of_publication, b.publisher,
                   b.image_url_s, b.image_url_m, b.image_url_l,
                   COALESCE(AVG(r.rating), 0) as avg_rating,
                   COUNT(r.rating) as vote_count,
                   (
                       (COUNT(r.rating)::decimal / (COUNT(r.rating) + %s)) * COALESCE(AVG(r.rating), 0) +
                       (%s::decimal / (COUNT(r.rating) + %s)) * %s
                   ) as weighted_score
            FROM books b
            LEFT JOIN ratings r ON b.book_id = r.book_id
            GROUP BY b.book_id
            ORDER BY weighted_score DESC
            LIMIT %s OFFSET %s
        """
        cursor.execute(query, (m, m, m, C, limit, offset))
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
                image_url_l=row[7],
                average_rating=round(float(row[8]), 2) if row[8] is not None else 0.0,
                rating_count=row[9]
            ))

        return BookList(books=books, total_count=total_count, page=page, limit=limit)

    finally:
        cursor.close()

@router.get("/{book_id}", response_model=Book)
def get_book_details(book_id: str, db: connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        query = """
            SELECT book_id, title, author, year_of_publication, publisher,
                   image_url_s, image_url_m, image_url_l
            FROM books
            WHERE book_id = %s
        """
        cursor.execute(query, (book_id,))
        row = cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Book not found")

        return Book(
            book_id=row[0],
            title=row[1],
            author=row[2],
            year_of_publication=row[3],
            publisher=row[4],
            image_url_s=row[5],
            image_url_m=row[6],
            image_url_l=row[7]
        )

    finally:
        cursor.close()
