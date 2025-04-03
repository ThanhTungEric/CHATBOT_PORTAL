import os
from dotenv import load_dotenv
import mysql.connector

# Load environment variables từ file .env
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
#print("Host:", DB_HOST, "User:", DB_USER, "Password:", DB_PASSWORD, "DB:", DB_NAME)

# Hàm kết nối database
def get_db():
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            #protocol="tcp"   
        )
        return conn
    except mysql.connector.Error as e:
        print("Database connection error:", e)
        return None

# Hàm lấy tất cả phòng ban
def get_departments():
    try:
        conn = get_db()  # Assign conn first
        if not conn:  # Check if connection failed
            print("Failed to connect to the database.")
            return []
        print("Connected successfully!")  # Optional: confirm connection
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, name_vi, name_en FROM departments")
        departments = cursor.fetchall()
        cursor.close()
        conn.close()
        return departments
    except Exception as e:
        print("Error fetching departments:", e)
        return []

# Hàm lấy danh sách website theo phòng ban
def get_websites_by_department(department_id):
    try:
        conn = get_db()
        if not conn:
            return []
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, name_vi, name_en FROM websites WHERE department_id = %s", (department_id,))
        websites = cursor.fetchall()
        cursor.close()
        conn.close()
        return websites
    except Exception as e:
        print("Error fetching websites:", e)
        return []

# Hàm lấy câu hỏi ngẫu nhiên theo website
def get_random_questions_by_website(website_id, limit=3):
    try:
        conn = get_db()
        if not conn:
            return []
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, question_vi, question_en FROM qa_pairs WHERE website_id = %s AND hidden = 0 ORDER BY RAND() LIMIT %s",
            (website_id, limit)
        )
        questions = cursor.fetchall()
        cursor.close()
        conn.close()
        return questions
    except Exception as e:
        print("Error fetching questions:", e)
        return []