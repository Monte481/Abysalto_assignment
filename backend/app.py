from flask import Flask, render_template, request, session, redirect, url_for
import pytesseract
from PIL import Image
from datetime import timedelta

app = Flask(__name__)
app.secret_key = "tajna"
app.permanent_session_lifetime = timedelta(minutes=5)

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

        text = pytesseract.image_to_string(Image.open(file))

        result.append({
            "filename": file.filename,
            "text": text
        })

        print(text)

    session["ocr_text"] = result

    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)