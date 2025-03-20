from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
import database
from database import get_db, get_categories, get_recommended_questions
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

# Pydantic model for user input
class QuestionRequest(BaseModel):
    question: str

# Chatbot API endpoint for answering questions
@app.post("/chat")
async def chat_response(request: QuestionRequest):
    try:
        conn = get_db()
        cur = conn.cursor()

        # Query the database for the answer
        cur.execute("SELECT answer FROM recommended_questions WHERE question_text = %s", (request.question,))
        result = cur.fetchone()
        cur.close()
        conn.close()

        if result:
            answer = result["answer"]
        else:
            answer = "Sorry, I don't have an answer for that."

        return {"answer": answer}

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

# API to get all categories
@app.get("/categories")
async def fetch_categories():
    categories = get_categories()
    if not categories:
        raise HTTPException(status_code=500, detail="Error fetching categories")
    return categories

# API to get recommended questions based on category
@app.get("/recommended-questions")
async def fetch_recommended_questions(category_id: int):
    questions = get_recommended_questions(category_id)
    if not questions:
        raise HTTPException(status_code=500, detail="Error fetching recommended questions")
    return questions

@app.get("/")
def home():
    return {"message": "Chatbot API is running!"}

# Get PORT from environment variables (default to 9090)
PORT = int(os.getenv("PORT", 9090))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
