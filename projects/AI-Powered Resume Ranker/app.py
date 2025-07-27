import os
from flask import Flask, render_template, request
import fitz  # PyMuPDF
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")  # assumes installed
pdf_path = "C:/Users/madhu/OneDrive/Documents/ElevateLabs/resume ranker/sample resume/Mohan Varikuti.pdf"

# ---------- Utils ----------
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def clean_text(text):
    doc = nlp(text.lower())
    tokens = [t.lemma_ for t in doc if not t.is_stop and t.is_alpha]
    return " ".join(tokens)

def rank_resumes(jd, resumes):
    corpus = [jd] + resumes
    vect = TfidfVectorizer()
    mat = vect.fit_transform(corpus)
    scores = cosine_similarity(mat[0:1], mat[1:])[0]
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked

# ---------- Flask ----------
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        jd = request.form['job_description']
        files = request.files.getlist('resumes')
        resume_names = []
        resume_texts = []

        for f in files:
            path = os.path.join(UPLOAD_FOLDER, f.filename)
            f.save(path)
            txt = extract_text_from_pdf(path)
            resume_names.append(f.filename)
            resume_texts.append(clean_text(txt))

        jd_clean = clean_text(jd)
        ranked = rank_resumes(jd_clean, resume_texts)
        results = [(resume_names[i], round(score*100, 2)) for i, score in ranked]

        return render_template('result.html', results=results)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
