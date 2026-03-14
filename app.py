from fastapi import FastAPI, UploadFile, File, Form
from transformers import pipeline
from pypdf import PdfReader
import re

app = FastAPI()

qa_pipeline = None
document_text = ""


@app.on_event("startup")
def load_models():
    global qa_pipeline
    print("Loading AI models...")

    qa_pipeline = pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad"
    )

    print("Models loaded successfully")


@app.get("/")
def home():
    return {"message": "AI Research Paper Assistant running"}


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_summary(text: str, max_sentences: int = 5) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    good_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence.split()) > 8:
            good_sentences.append(sentence)

    if not good_sentences:
        return "Could not generate a proper summary from the uploaded PDF."

    return " ".join(good_sentences[:max_sentences])


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global document_text

    pdf = PdfReader(file.file)

    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "

    text = clean_text(text)

    if not text:
        return {"error": "No readable text found in the PDF"}

    document_text = text

    return {
        "message": "Document uploaded successfully",
        "characters_loaded": len(document_text)
    }


@app.post("/summarize")
def summarize():
    global document_text

    if document_text == "":
        return {"error": "Upload a document first"}

    summary = extract_summary(document_text)

    return {"summary": summary}


@app.post("/ask")
def ask(question: str = Form(...)):
    global document_text

    if document_text == "":
        return {"error": "Upload a document first"}

    context = document_text[:1500]

    answer = qa_pipeline(
        question=question,
        context=context
    )

    if answer["score"] < 0.2:
        return {
            "question": question,
            "answer": "Answer not found clearly in the uploaded document.",
            "confidence": answer["score"]
        }

    return {
        "question": question,
        "answer": answer["answer"],
        "confidence": answer["score"]
    }