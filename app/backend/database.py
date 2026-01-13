import psycopg2
from psycopg2.extensions import connection
import os
from contextlib import contextmanager

def get_db_connection():
    """
    Establishes a connection to the PostgreSQL database.
    Configuration is hardcoded as per user request, but can be moved to env vars.
    """
    try:
        conn = psycopg2.connect(
            dbname='postgres',
            user='postgres',
            password='admin',
            host='127.0.0.1',
            port='5432'
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise e

def get_db():
    """
    Context manager for database connections.
    Ensures connection is closed after use.
    """
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()
