from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import database
from database import get_db, get_departments, get_websites_by_department, get_random_questions_by_website
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langdetect import detect  # cài: pip install langdetect

app = FastAPI()

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

# Load SBERT model and precompute embeddings
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # Mô hình đa ngôn ngữ

# Hàm lấy tất cả câu hỏi từ database
def get_all_questions():
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, question_vi, question_en, answer_vi, answer_en FROM qa_pairs WHERE hidden = 0")
    questions = cursor.fetchall()
    cursor.close()
    conn.close()
    return questions

# Tính trước embeddings khi khởi động ứng dụng
questions = get_all_questions()
question_texts_vi = [q['question_vi'] for q in questions]
question_texts_en = [q['question_en'] for q in questions]
question_embeddings_vi = model.encode(question_texts_vi)
question_embeddings_en = model.encode(question_texts_en)

# Các endpoint hiện có
@app.get("/departments")
async def fetch_departments():
    departments = get_departments()
    if not departments:
        raise HTTPException(status_code=404, detail="No departments found")
    return departments

@app.get("/websites/{department_id}")
async def fetch_websites(department_id: int):
    websites = get_websites_by_department(department_id)
    if not websites:
        raise HTTPException(status_code=404, detail="No websites found for this department")
    return websites

@app.get("/questions/{website_id}")
async def fetch_questions(website_id: int):
    questions = get_random_questions_by_website(website_id)
    if not questions:
        raise HTTPException(status_code=404, detail="No questions found for this website")
    return questions

# Endpoint chat cập nhật với SBERT
@app.post("/chat")
async def chat_response(request: QuestionRequest):
    try:
        user_question = request.question
        lang = detect(user_question)

        # Chọn embeddings phù hợp
        if lang == "en":
            user_embedding = model.encode([user_question])
            similarities = cosine_similarity(user_embedding, question_embeddings_en)[0]
            top_texts = [q['question_en'] for q in questions]
            top_answers = [q['answer_en'] for q in questions]
        else:
            user_embedding = model.encode([user_question])
            similarities = cosine_similarity(user_embedding, question_embeddings_vi)[0]
            top_texts = [q['question_vi'] for q in questions]
            top_answers = [q['answer_vi'] for q in questions]

        # Tìm câu hỏi gần nhất
        max_similarity_index = np.argmax(similarities)
        max_similarity = similarities[max_similarity_index]

        # Nếu độ giống cao thì trả lời luôn
        if max_similarity > 0.7:
            return {"answer": top_answers[max_similarity_index]}
        else:
            # Nếu độ giống trung bình thì đưa ra gợi ý
            top_indices = np.argsort(similarities)[-3:][::-1]
            suggestions = [top_texts[i] for i in top_indices if similarities[i] >= 0.5]

            if suggestions:
                return {"suggestions": suggestions}
            else:
                return {
                    "answer": "Sorry, I do not have an appropriate answer. Could you ask more clearly?"
                    if lang == "en" else
                    "Xin lỗi, tôi không có câu trả lời phù hợp. Bạn có thể hỏi chi tiết hơn không?"
                }
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/add-question")
async def add_question(new_question: NewQuestion):
    global questions, question_embeddings_vi, question_embeddings_en
    try:
        # Encode riêng cho từng ngôn ngữ
        new_embedding_vi = model.encode([new_question.question_vi])[0]
        new_embedding_en = model.encode([new_question.question_en])[0]

        # Tính độ giống riêng
        sim_vi = cosine_similarity([new_embedding_vi], question_embeddings_vi)[0]
        sim_en = cosine_similarity([new_embedding_en], question_embeddings_en)[0]

        max_sim_vi = np.max(sim_vi)
        max_sim_en = np.max(sim_en)

        # Kiểm tra trùng nếu 1 trong 2 cao > 0.9
        if max_sim_vi >= 0.9 or max_sim_en >= 0.9:
            vi_index = np.argmax(sim_vi)
            en_index = np.argmax(sim_en)
            most_similar_vi = questions[vi_index]['question_vi']
            most_similar_en = questions[en_index]['question_en']
            raise HTTPException(
                status_code=400,
                detail=f"Câu hỏi đã tồn tại với nội dung tương tự:\n- VI: {most_similar_vi}\n- EN: {most_similar_en}"
            )

        # Chưa trùng thì thêm vào DB
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO qa_pairs (website_id, question_vi, answer_vi, question_en, answer_en) VALUES (%s, %s, %s, %s, %s)",
            (new_question.website_id, new_question.question_vi, new_question.answer_vi, new_question.question_en, new_question.answer_en)
        )
        conn.commit()
        question_id = cursor.lastrowid
        cursor.close()
        conn.close()

        # Cập nhật lại questions & embeddings
        questions = get_all_questions()
        question_texts_vi = [q['question_vi'] for q in questions]
        question_texts_en = [q['question_en'] for q in questions]
        question_embeddings_vi = model.encode(question_texts_vi)
        question_embeddings_en = model.encode(question_texts_en)

        return {"id": question_id, "message": "Question added successfully"}
    except HTTPException as he:
        raise he
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/update-question/{id}")
async def update_question(id: int, updated_question: UpdateQuestion):
    global questions, question_embeddings_vi, question_embeddings_en
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE qa_pairs SET website_id = %s, question_vi = %s, answer_vi = %s, question_en = %s, answer_en = %s WHERE id = %s",
            (updated_question.website_id, updated_question.question_vi, updated_question.answer_vi, updated_question.question_en, updated_question.answer_en, id)
        )
        conn.commit()
        cursor.close()
        conn.close()
        
        # Cập nhật lại embeddings sau khi chỉnh sửa
        #global questions, question_embeddings
        index_map = {q["id"]: i for i, q in enumerate(questions)}
        if id in index_map:
            idx = index_map[id]
            # Cập nhật questions
            questions[idx]["question_vi"] = updated_question.question_vi
            questions[idx]["answer_vi"] = updated_question.answer_vi
            questions[idx]["question_en"] = updated_question.question_en
            questions[idx]["answer_en"] = updated_question.answer_en

            # Encode lại câu hỏi đã update
            question_embeddings_vi[idx] = model.encode([updated_question.question_vi])[0]
            question_embeddings_en[idx] = model.encode([updated_question.question_en])[0]
        
        return {"message": "Question updated successfully"}
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/hide-question/{question_id}")
async def hide_question(question_id: int):
    global questions, question_embeddings_vi, question_embeddings_en
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("UPDATE qa_pairs SET hidden = NOT hidden WHERE id = %s", (question_id,))
        conn.commit()
        cursor.close()
        conn.close()
        
        # Cập nhật lại embeddings sau khi ẩn/hiện
        #global questions, question_embeddings
        questions = get_all_questions()
        question_texts_vi = [q['question_vi'] for q in questions]
        question_texts_en = [q['question_en'] for q in questions]
        question_embeddings_vi = model.encode(question_texts_vi)
        question_embeddings_en = model.encode(question_texts_en)
        
        return {"message": "Question visibility toggled"}
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/all-qa-pairs")
async def fetch_all_qa_pairs():
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT q.id, d.id as department_id, d.name_vi as department, w.id as website_id, w.name_vi as website, q.question_vi, q.answer_vi, q.question_en, q.answer_en, q.hidden
            FROM qa_pairs q
            JOIN websites w ON q.website_id = w.id
            JOIN departments d ON w.department_id = d.id
        """)
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