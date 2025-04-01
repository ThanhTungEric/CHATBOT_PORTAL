from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
import database
from database import get_db, get_categories, get_recommended_questions, get_category_id
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Pydantic models for requests
class QuestionRequest(BaseModel):
    question: str

class NewQuestion(BaseModel):
    category: str
    question: str
    answer: str

class UpdateQuestion(BaseModel):
    category: str
    question: str
    answer: str

STOP_WORDS = {"what", "the", "is", "how", "a", "an", "to", "for", "of", "and", "in", "on", "at", "with", "by", "from"}

# Chatbot API endpoint for answering questions
@app.post("/chat")
async def chat_response(request: QuestionRequest):
    try:
        conn = get_db()
        cur = conn.cursor()
        
        # Step 1: Check for exact match
        cur.execute("SELECT answer FROM recommended_questions WHERE question_text = %s", (request.question,))
        result = cur.fetchone()
        
        if result:
            # Exact match found, return the answer
            answer = result["answer"]
            cur.close()
            conn.close()
            return {"answer": answer}
        else:
            # No exact match, filter stop words and search by keywords
            keywords = [kw for kw in request.question.lower().split() if kw not in STOP_WORDS]
            if not keywords:
                cur.close()
                conn.close()
                return {"suggestions": []}
            
            # Build a query to match all keywords in question_text
            query = "SELECT id, question_text FROM recommended_questions WHERE " + " OR ".join(["question_text ILIKE %s" for _ in keywords])
            params = ['%' + kw + '%' for kw in keywords]
            cur.execute(query, params)
            suggestions = cur.fetchall()
            
            cur.close()
            conn.close()
            
            # Return suggestions as a list of objects with id and text
            return {"suggestions": [{"id": s["id"], "text": s["question_text"]} for s in suggestions]}
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

# API to get all categories
@app.get("/all-categories")
async def fetch_categories():
    try:
        categories = get_categories()
        if not categories:
            raise HTTPException(status_code=404, detail="No categories found")
        return categories
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

# API to get recommended questions based on category
@app.get("/recommended-questions")
async def fetch_recommended_questions(category_id: int):
    questions = get_recommended_questions(category_id)
    if not questions:
        raise HTTPException(status_code=500, detail="Error fetching recommended questions")
    return questions

# API to get all questions
@app.get("/all-questions")
async def fetch_all_questions():
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT q.id, c.category_name AS category, q.question_text, q.answer, q.hidden
            FROM recommended_questions q
            JOIN categories c ON q.category_id = c.id
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        results = [
            {"id": row["id"], "category": row["category"], "question": row["question_text"], "answer": row["answer"], "hidden": row["hidden"]}
            for row in rows
        ]
        return results
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

# ✅ **Fixed: API to add a new question**
@app.post("/add-question")
async def add_question(new_question: NewQuestion):
    try:
        conn = get_db()
        cur = conn.cursor()

        # ✅ **Use `get_category_id` to ensure the category exists**
        category_id = get_category_id(new_question.category)
        if category_id is None:
            raise HTTPException(status_code=400, detail="Category does not exist")

        # ✅ **Insert new question into the database**
        cur.execute(
            "INSERT INTO recommended_questions (category_id, question_text, answer) VALUES (%s, %s, %s) RETURNING id",
            (category_id, new_question.question, new_question.answer),
        )
        question_id = cur.fetchone()["id"]

        conn.commit()
        cur.close()
        conn.close()

        return {"id": question_id, "message": "Question added successfully"}

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

# API to update a question
@app.put("/update-question/{id}")
async def update_question(id: int, updated_question: UpdateQuestion):
    try:
        conn = get_db()
        cur = conn.cursor()

        print(f"Received update request: {updated_question}")

        # Ensure category exists
        category_id = get_category_id(updated_question.category)
        if category_id is None:
            raise HTTPException(status_code=400, detail="Category does not exist")

        # Update question
        cur.execute(
            "UPDATE recommended_questions SET category_id = %s, question_text = %s, answer = %s WHERE id = %s",
            (category_id, updated_question.question, updated_question.answer, id),
        )
        conn.commit()

        cur.close()
        conn.close()

        return {"message": "Question updated successfully"}

    except Exception as e:
        print("Error updating question:", e)
        raise HTTPException(status_code=500, detail=str(e))



# API to hide a question
@app.patch("/hide-question/{question_id}")
async def hide_question(question_id: int):
    try:
        conn = get_db()
        cur = conn.cursor()

        # Toggle the hidden column
        cur.execute("""
            UPDATE recommended_questions 
            SET hidden = NOT hidden
            WHERE id = %s
        """, (question_id,))

        conn.commit()
        cur.close()
        conn.close()
        return {"message": "Question visibility toggled"}

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "Chatbot API is running!"}

# Get PORT from environment variables (default to 9090)
PORT = int(os.getenv("PORT", 9090))

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=PORT)
