from flask import Flask, render_template, request, session, redirect, url_for
import pytesseract
import os
import pymupdf
import faiss
import numpy as np

from dotenv import load_dotenv
from mistralai import Mistral
from PIL import Image
from datetime import timedelta
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
app.secret_key = "tajna"
app.permanent_session_lifetime = timedelta(minutes=5)
load_dotenv()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-medium-latest"
client = Mistral(api_key=api_key)

@app.route("/", methods=["GET"])
def home():
    results = session.get("ocr_text", [])
    return render_template("index.html", results=results)

@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")

    result = []

    for file in files:
        if file.filename == "":
            continue

        if file.content_type == "application/pdf":
            doc = pymupdf.open(stream=file.read(), filetype="pdf")

            text = ""
            for page in doc:
                text += page.get_text()

            result.append({
                "filename": file.filename,
                "text": text
            })
        
            print(text)


        elif file.content_type.startswith("image/"):
            text = pytesseract.image_to_string(Image.open(file))

            result.append({
                "filename": file.filename,
                "text": text
            })

            print(text)

    global faiss_index, all_chunks_store

    faiss_index, all_chunks_store = build_rag_index(result)

    session["ocr_text"] = result

    return redirect(url_for("home"))


@app.route("/ask", methods=["POST"])
def ask():
    question = request.form["question"]

    retrieved = retrieved_chunks(question, faiss_index, all_chunks_store)

    context = "\n\n".join(
        f"File: {chunk['filename']}\n{chunk['text']}"
        for chunk in retrieved
    )

    print(context)

    chat_response = client.chat.complete(
        model=model,
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer only ont the given text!"
                    "If the answer isnt in the text, tell that the answer isnt found!"
                )
            },
            {
                "role": "user",
                "content": f"Documents:\n{context}\n\nQuestion:\n{question}"
            }
        ]
    )

    print(chat_response.choices[0].message.content)
    
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
    app.run(debug=True)