import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Function to establish database connection
def get_db():
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        print("Database connection error:", e)
        return None

# Function to fetch FAQ categories
def get_categories():
    conn = get_db()
    if not conn:
        return []

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, category_name FROM categories")
            categories = cur.fetchall()
            return [{"id": c["id"], "name": c["category_name"]} for c in categories]
    except Exception as e:
        print("Error fetching categories:", e)
        return []
    finally:
        conn.close()

# Function to fetch recommended questions by category
def get_recommended_questions(category_id):
    conn = get_db()
    if not conn:
        return []

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, question_text FROM recommended_questions WHERE category_id = %s", (category_id,))
            questions = cur.fetchall()
            return [{"id": q["id"], "text": q["question_text"]} for q in questions]
    except Exception as e:
        print("Error fetching recommended questions:", e)
        return []
    finally:
        conn.close()
