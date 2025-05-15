import logging
import os
import json
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator

import database
from database import get_db, create_qa_table_if_not_exists
from DeepSeekV1 import DeepSeekHelper

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

translator = Translator()
DeepSeek = DeepSeekHelper()

# Ensure table exists
create_qa_table_if_not_exists()

# Load intent templates
try:
    with open("qa_data_fewshot.json", encoding="utf-8") as f:
        qa_data = json.load(f)
    intent_query_templates = qa_data.get("intent_query_templates", {})
except (FileNotFoundError, json.JSONDecodeError) as e:
    logger.error(f"Failed to load intent templates: {e}")
    intent_query_templates = {}

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

# Load SBERT
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

questions = []
question_embeddings_vi = []
question_embeddings_en = []

def get_all_questions():
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, question_vi, question_en, answer_vi, answer_en FROM qa_pairs WHERE hidden = 0")
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return data

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

@app.on_event("startup")
async def startup_event():
    load_embeddings()

@app.post("/chat")
async def chat_response(request: QuestionRequest):
    try:
        user_question = request.question
        lang = translator.detect(user_question).lang

        if not questions:
            answer = DeepSeek.call_model(user_question, lang=lang)
            return {
                "answer": answer or "Hiện chưa có dữ liệu câu hỏi.",
                "is_fallback": True
            }

        # Intent detection
        intents = []
        if "moodle" in user_question.lower():
            intents.append({"intent": "login", "entity": "MOODLE"})
        if "sis" in user_question.lower():
            intents.append({"intent": "login", "entity": "SIS"})

        if intents:
            combined_answer = []
            used_DeepSeek = False

            for intent_dict in intents:
                intent = intent_dict["intent"]
                entity = intent_dict["entity"]

                templates = intent_query_templates.get(lang, {})
                template = templates.get(intent, templates.get("default", "{intent} {entity}"))
                intent_question = template.format(intent=intent, entity=entity)

                user_embedding = model.encode([intent_question])
                if lang == "en":
                    similarities = cosine_similarity(user_embedding, question_embeddings_en)[0]
                    top_answers = [q['answer_en'] for q in questions]
                else:
                    similarities = cosine_similarity(user_embedding, question_embeddings_vi)[0]
                    top_answers = [q['answer_vi'] for q in questions]

                max_index = np.argmax(similarities)
                top_k = 3
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                combined_confidence = np.sqrt(sum([s ** 2 for s in [similarities[i] for i in top_indices]]))

                if similarities[max_index] > 0.7 or combined_confidence > 1.15:
                    combined_answer.append(top_answers[max_index])
                else:
                    fallback = DeepSeek.call_model(intent_question, lang=lang)
                    if fallback and not any(w in fallback.lower() for w in ["sorry", "xin lỗi", "unavailable", "không thể"]):
                        combined_answer.append(fallback)
                        used_DeepSeek = True
                    else:
                        msg = (
                            f"Không tìm thấy thông tin cho {intent_question}" if lang == "vi"
                            else f"No information found for {intent_question}"
                        )
                        combined_answer.append(msg)

            return {
                "answer": "\n".join(combined_answer),
                "is_fallback": used_DeepSeek
            }

        else:
            user_embedding = model.encode([user_question])
            if lang == "en":
                similarities = cosine_similarity(user_embedding, question_embeddings_en)[0]
                top_texts = [q['question_en'] for q in questions]
                top_answers = [q['answer_en'] for q in questions]
            else:
                similarities = cosine_similarity(user_embedding, question_embeddings_vi)[0]
                top_texts = [q['question_vi'] for q in questions]
                top_answers = [q['answer_vi'] for q in questions]

            max_index = np.argmax(similarities)
            top_k = 3
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            combined_confidence = np.sqrt(sum([s ** 2 for s in [similarities[i] for i in top_indices]]))

            if similarities[max_index] > 0.7 or combined_confidence > 1.15:
                return {"answer": top_answers[max_index], "is_fallback": False}

            suggestions = [top_texts[i] for i in top_indices if similarities[i] >= 0.5]
            answer = DeepSeek.call_model(user_question, lang=lang)

            if answer and not any(k in answer.lower() for k in ["sorry", "xin lỗi", "unavailable", "không thể"]):
                return {"answer": answer, "is_fallback": True}
            elif answer:
                return {"answer": answer, "is_fallback": True, "suggestions": suggestions or None}
            elif suggestions:
                return {
                    "answer": (
                        "I'm not sure, but maybe you're asking about:" if lang == "en"
                        else "Tôi không chắc, nhưng có thể bạn đang hỏi về:"
                    ),
                    "suggestions": suggestions,
                    "is_fallback": False
                }
            else:
                return {
                    "answer": (
                        "Sorry, the external AI is unavailable due to rate limits or high demand. Try rephrasing or contact ITD."
                        if lang == "en"
                        else "Xin lỗi, AI bên ngoài không khả dụng do giới hạn tốc độ hoặc nhu cầu cao. Hãy thử diễn đạt lại hoặc liên hệ ITD."
                    ),
                    "is_fallback": False
                }

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-question")
async def add_question(new_question: NewQuestion):
    try:
        new_emb_vi = model.encode([new_question.question_vi])[0]
        new_emb_en = model.encode([new_question.question_en])[0]

        sim_vi = cosine_similarity([new_emb_vi], question_embeddings_vi)[0] if question_embeddings_vi else [0.0]
        sim_en = cosine_similarity([new_emb_en], question_embeddings_en)[0] if question_embeddings_en else [0.0]

        if np.max(sim_vi) >= 0.9 or np.max(sim_en) >= 0.9:
            raise HTTPException(status_code=400, detail="Question already exists.")

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
            0
        ))
        conn.commit()
        question_id = cursor.lastrowid
        cursor.close()
        conn.close()

        load_embeddings()
        return {"id": question_id, "message": "Question added successfully"}

    except Exception as e:
        logger.error(f"Add question error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/update-question/{id}")
async def update_question(id: int, updated: UpdateQuestion):
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE qa_pairs 
            SET website_id = %s, question_vi = %s, answer_vi = %s, question_en = %s, answer_en = %s 
            WHERE id = %s
        """, (
            updated.website_id, updated.question_vi, updated.answer_vi,
            updated.question_en, updated.answer_en, id
        ))
        conn.commit()
        cursor.close()
        conn.close()

        load_embeddings()
        return {"message": "Question updated successfully"}

    except Exception as e:
        logger.error(f"Update question error: {e}")
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
        logger.error(f"Hide question error: {e}")
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
        logger.error(f"Fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    PORT = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="127.0.0.1", port=PORT)

if __name__ == "__main__":
    PORT = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="127.0.0.1", port=PORT)
