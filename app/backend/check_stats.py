import psycopg2
import os

def check_stats():
    try:
        conn = psycopg2.connect(
            dbname='postgres',
            user='postgres',
            password='admin',
            host='127.0.0.1',
            port='5432'
        )
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM books")
        books_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM ratings")
        ratings_count = cursor.fetchone()[0]

        print(f"Books: {books_count}")
        print(f"Ratings: {ratings_count}")

        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_stats()
