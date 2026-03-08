from flask import Flask, flash, render_template, request, session, redirect, url_for
import pytesseract
import os
import pymupdf
import faiss
import numpy as np
import logging
from uuid import uuid4

from dotenv import load_dotenv
from mistralai import Mistral
from PIL import Image
from datetime import timedelta
from sentence_transformers import SentenceTransformer

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ["APP_SECRET_KEY"]
app.permanent_session_lifetime = timedelta(minutes=15)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-medium-latest"
client = Mistral(api_key=api_key)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

session_storage = {}

@app.before_request
def create_session():
     
    if 'session_id' not in session:
        session['session_id'] = str(uuid4())
        session.permanent = True
    
    session_id = session['session_id']
    if session_id not in session_storage:
        session_storage[session_id] = {
            'documents': [],
            'chunks': [],
            'faiss_index': None
        }

@app.route("/", methods=["GET"])
def home():
    session_id = session['session_id']
    storage = session_storage.get(session_id, {})
    
    documents_store = storage.get('documents', [])
    answer = session.get("answer")
    question = session.get("question")

    return render_template("index.html",
                           results=documents_store,
                           answer=answer,
                           question=question,
                           )

@app.route("/reset", methods=["POST"])
def reset():
    session_id = session['session_id']
    
    if session_id in session_storage:
        del session_storage[session_id]
        session_storage[session_id] = {
            'documents': [],
            'chunks': [],
            'faiss_index': None
        }
    
    session.pop("answer", None)
    session.pop("question", None)
    
    logger.info(f"Session {session_id} reset!")
    
    return redirect(url_for("home"))

@app.route("/upload", methods=["POST"])
def upload():
    session_id = session['session_id']
    
    files = request.files.getlist("files")

    if not files or all(f.filename == "" for f in files):
        flash("No files selected!", "warning")
        return redirect(url_for("home"))
    
    logger.info(f"[{session_id}] Starting file upload...")

    result = []

    for file in files:
        if file.filename == "":
            continue

        try:
            if file.content_type == "application/pdf":
                doc = pymupdf.open(stream=file.read(), filetype="pdf")

                text = "".join (page.get_text() for page in doc)

                result.append({
                    "filename": file.filename,
                    "text": text
                })

                logger.info(f"[{session_id}] Successfully processed PDF: {file.filename}")

            elif file.content_type.startswith("image/"):

                text = pytesseract.image_to_string(Image.open(file))

                result.append({
                    "filename": file.filename,
                    "text": text
                })

                logger.info(f"[{session_id}] Successfully processed image: {file.filename}")
            else:
                logger.warning(f"[{session_id}] Unsupported file type: {file.filename}")
                flash(f"Unsupported file type: {file.filename}", "warning")

        except Exception as e:
            logger.error(f"[{session_id}] Error processing {file.filename}: {e}")
            flash(f"Error processing {file.filename}: {str(e)}", "error")
            continue

    if not result:
        flash("No valid files could be processed!", "error")
        return redirect(url_for("home"))
        

    logger.info(f"[{session_id}] Processed {len(result)} files successfully!")

    faiss_index, all_chunks = build_rag_index(result)

    session_storage[session_id] = {
        'documents': result,
        'chunks': all_chunks,
        'faiss_index': faiss_index
    }

    logger.info(f"[{session_id}] Built FAISS index with {len(all_chunks)} chunks!")


    return redirect(url_for("home"))


@app.route("/ask", methods=["POST"])
def ask():
    session_id = session['session_id']
    storage = session_storage.get(session_id, {})

    faiss_index = storage.get('faiss_index')
    all_chunks = storage.get('chunks', [])

    if not faiss_index or not all_chunks:
        flash("Upload documents first!", "error")
        logger.warning(f"[{session_id}] Ask called without documents!")
        return redirect(url_for("home"))

    question = request.form["question"].strip()

    if not question:
        flash("Question cannot be left empty!", "error")
        return redirect(url_for("home"))
    
    logger.info(f"[{session_id}] Question received: {question}")
    session["question"] = question

    retrieved = retrieved_chunks(question, faiss_index, all_chunks)
    
    logger.info(f"[{session_id}] Retrieved {len(retrieved)} relevant chunks!")

    context = "\n\n".join(
        f"File: {chunk['filename']}\n{chunk['text']}"
        for chunk in retrieved
    )

    try:
        chat_response = client.chat.complete(
            model=model,
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Answer only on the given text! "
                        "If the answer isnt in the text, tell that the answer isnt found! "
                    )
                },  
                {
                    "role": "user",
                    "content": f"Documents:\n{context}\n\nQuestion:\n{question}"
                }
            ]
        )
        
        session["answer"] = chat_response.choices[0].message.content
        logger.info(f"[{session_id}] Answer generated successfully!")

    except Exception as e:
        logger.error(f"[{session_id}] Error calling Mistral API: {e}")
        flash(f"Error generating answer: {str(e)}", "error")
        return redirect(url_for("home"))
    
    return redirect(url_for("home"))


def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

def build_rag_index(result):

    all_chunks = []

    for res in result:
        chunks = chunk_text(res["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "filename": res["filename"],
                "chunk_id": i,
                "text": chunk
            })        
        
    if not all_chunks:
        return None, []

    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = embedding_model.encode(texts)

    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, all_chunks

def retrieved_chunks(question, index, all_chunks, k=3):
    question_embedding = embedding_model.encode([question])
    question_embedding = np.array(question_embedding).astype("float32")

    distances, indices = index.search(question_embedding, k)

    return [all_chunks[i] for i in indices[0]]


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)