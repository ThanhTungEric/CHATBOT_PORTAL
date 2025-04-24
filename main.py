from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import database
from database import get_db, create_qa_table_if_not_exists
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from googletrans import Translator

app = FastAPI()

# Tạo bảng nếu chưa tồn tại
create_qa_table_if_not_exists()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QuestionRequest(BaseModel):
    question: str

class NewQuestion(BaseModel):
    website_id: int
    question_vi: str
    answer_vi: str
    question_en: str
    answer_en: str

class UpdateQuestion(BaseModel):
    website_id: int
    question_vi: str
    answer_vi: str
    question_en: str
    answer_en: str

# Load SBERT model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Global variables for questions and embeddings
questions = []
question_embeddings_vi = []
question_embeddings_en = []

# Load questions and compute embeddings
def load_embeddings():
    global questions, question_embeddings_vi, question_embeddings_en
    questions = get_all_questions()
    if questions:
        question_texts_vi = [q['question_vi'] for q in questions]
        question_texts_en = [q['question_en'] for q in questions]
        question_embeddings_vi = model.encode(question_texts_vi)
        question_embeddings_en = model.encode(question_texts_en)
    else:
        question_embeddings_vi = []
        question_embeddings_en = []

# Load once on startup
@app.on_event("startup")
async def startup_event():
    load_embeddings()

# Get all questions
def get_all_questions():
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, question_vi, question_en, answer_vi, answer_en FROM qa_pairs WHERE hidden = 0")
    questions = cursor.fetchall()
    cursor.close()
    conn.close()
    return questions

# Chat endpoint
@app.post("/chat")
async def chat_response(request: QuestionRequest):
    try:
        user_question = request.question
        lang = Translator.detect(user_question)

        if not questions:
            return {"answer": "Hiện chưa có dữ liệu câu hỏi."}

        if lang == "en":
            user_embedding = model.encode([user_question])
            similarities = cosine_similarity(user_embedding, question_embeddings_en)[0]
            top_texts = [q['question_en'] for q in questions]
            top_answers = [q['answer_en'] for q in questions]
            print("English similarities:", similarities)
          
        else:
            user_embedding = model.encode([user_question])
            similarities = cosine_similarity(user_embedding, question_embeddings_vi)[0]
            top_texts = [q['question_vi'] for q in questions]
            top_answers = [q['answer_vi'] for q in questions]
            print("Vietnamese similarities:", similarities)

        max_similarity_index = np.argmax(similarities)
        max_similarity = similarities[max_similarity_index]
        print(top_answers[max_similarity_index])

        if max_similarity > 0.7:
            return {"answer": top_answers[max_similarity_index]}
            
        else:
            top_indices = np.argsort(similarities)[-3:][::-1]
            suggestions = [top_texts[i] for i in top_indices if similarities[i] >= 0.5]
            if suggestions:
                return {
                    "answer": (
                        "I'm not sure, but maybe you are asking about one of these questions:"
                        if lang == "en"
                        else "Tôi không chắc chắn, nhưng có thể bạn đang hỏi về một trong những câu hỏi này:"
                    ),
                    "suggestions": suggestions
                }
            if lang == "en" or lang == "vi":
                return {
                    "answer": (
                        "AWGMQEfVmpJ8LGt2uhwpsE9M5p1Df7yyDcUYqQpdUiSZLGKZjZuCuKguEUSFGZ59"
                    )
                }
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-question")
async def add_question(new_question: NewQuestion):
    try:
        new_embedding_vi = model.encode([new_question.question_vi])[0]
        new_embedding_en = model.encode([new_question.question_en])[0]

        # Kiểm tra nếu embeddings hiện tại không rỗng
        if question_embeddings_vi.size > 0:
            sim_vi = cosine_similarity([new_embedding_vi], question_embeddings_vi)[0]
        else:
            sim_vi = np.array([0.0])

        if question_embeddings_en.size > 0:
            sim_en = cosine_similarity([new_embedding_en], question_embeddings_en)[0]
        else:
            sim_en = np.array([0.0])

        max_sim_vi = np.max(sim_vi).item()
        max_sim_en = np.max(sim_en).item()

        if max_sim_vi >= 0.9 or max_sim_en >= 0.9:
            vi_index = np.argmax(sim_vi)
            en_index = np.argmax(sim_en)
            #most_similar_vi = questions[vi_index]['question_vi']
            #most_similar_en = questions[en_index]['question_en']
            raise HTTPException(
                status_code=400,
                #detail=f"Câu hỏi đã tồn tại:\n- VI: {most_similar_vi}\n- EN: {most_similar_en}"
            )

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO qa_pairs (website_id, question_vi, answer_vi, question_en, answer_en, hidden)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            new_question.website_id,
            new_question.question_vi,
            new_question.answer_vi,
            new_question.question_en,
            new_question.answer_en,
            0  # hidden mặc định là 0
        ))
        conn.commit()
        question_id = cursor.lastrowid
        cursor.close()
        conn.close()

        load_embeddings()
        return {"id": question_id, "message": "Question added successfully"}

    except HTTPException as he:
        raise he
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))



@app.put("/update-question/{id}")
async def update_question(id: int, updated_question: UpdateQuestion):
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE qa_pairs SET website_id = %s, question_vi = %s, answer_vi = %s, question_en = %s, answer_en = %s WHERE id = %s
        """, (updated_question.website_id, updated_question.question_vi, updated_question.answer_vi, updated_question.question_en, updated_question.answer_en, id))
        conn.commit()
        cursor.close()
        conn.close()

        load_embeddings()
        return {"message": "Question updated successfully"}
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/hide-question/{question_id}")
async def hide_question(question_id: int):
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("UPDATE qa_pairs SET hidden = NOT hidden WHERE id = %s", (question_id,))
        conn.commit()
        cursor.close()
        conn.close()

        load_embeddings()
        return {"message": "Question visibility toggled"}
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/all-qa-pairs")
async def fetch_all_qa_pairs():
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, website_id, question_vi, answer_vi, question_en, answer_en, hidden FROM qa_pairs")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    PORT = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="127.0.0.1", port=PORT)
