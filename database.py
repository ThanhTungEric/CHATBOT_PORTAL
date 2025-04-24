import os
from dotenv import load_dotenv
import mysql.connector

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

# Kết nối DB
def get_db():
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        print("Connected completed")
        return conn
    except mysql.connector.Error as e:
        print("Database connection error:", e)
        return None

# Tạo bảng nếu chưa có
def create_qa_table_if_not_exists():
    try:
        conn = get_db()
        if not conn:
            print("Failed to connect to the database.")
            return

        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS qa_pairs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                website_id INT NOT NULL,
                question_vi TEXT NOT NULL,
                answer_vi TEXT NOT NULL,
                question_en TEXT NOT NULL,
                answer_en TEXT NOT NULL,
                role VARCHAR(50) DEFAULT 'user',
                hidden BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        print("qa_pairs table checked/created successfully.")
    except Exception as e:
        print("Error creating qa_pairs table:", e)