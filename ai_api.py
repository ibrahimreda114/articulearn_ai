import os
import re
import time
import logging
import tempfile
import unicodedata
from difflib import SequenceMatcher
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from faster_whisper import WhisperModel

# ─────────────────────────────
# Setup
# ─────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("articulearn")

app = FastAPI(
    title="ArticuLearn AI Service (Faster Whisper)",
    version="3.0.0"
)

# ⚡ Faster Whisper Model (NO ffmpeg dependency)
model = WhisperModel("base", device="cpu", compute_type="int8")

# ─────────────────────────────
# Response Schema (UNCHANGED)
# ─────────────────────────────

class ScoreBreakdown(BaseModel):
    accuracy: int
    completeness: int
    final: int

class EvaluationResponse(BaseModel):
    target: str
    predicted: str
    scores: ScoreBreakdown
    feedback: str
    mistakes: list[str]
    phoneme_notes: list[str]
    processing_time_ms: int

# ─────────────────────────────
# Text Utils (same logic)
# ─────────────────────────────

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text: str):
    return text.split()

def _sim(a, b):
    return SequenceMatcher(None, a, b).ratio()

# ─────────────────────────────
# Scoring (same idea)
# ─────────────────────────────

def compute_scores(target, predicted):
    t = tokenize(target)
    p = tokenize(predicted)

    if not t:
        return ScoreBreakdown(0, 0, 0)

    acc_list = []
    for tw in t:
        best = max((_sim(tw, pw) for pw in p), default=0)
        acc_list.append(best)

    accuracy = sum(acc_list) / len(acc_list)

    matched = sum(
        1 for tw in t
        if any(_sim(tw, pw) >= 0.6 for pw in p)
    )

    completeness = matched / len(t)

    final = 0.6 * accuracy + 0.4 * completeness

    return ScoreBreakdown(
        accuracy=round(accuracy * 100),
        completeness=round(completeness * 100),
        final=round(final * 100),
    )

# ─────────────────────────────
# Mistakes (light version)
# ─────────────────────────────

def analyse_mistakes(target, predicted):
    t = tokenize(target)
    p = tokenize(predicted)

    mistakes = []

    for tw in t:
        if not any(_sim(tw, pw) > 0.5 for pw in p):
            mistakes.append(f"word '{tw}' not detected")

    for pw in p:
        if not any(_sim(pw, tw) > 0.5 for tw in t):
            mistakes.append(f"extra word '{pw}' detected")

    return mistakes

# ─────────────────────────────
# Feedback
# ─────────────────────────────

def feedback(score):
    if score.final >= 90:
        return "Perfect pronunciation 🎉"
    if score.final >= 70:
        return "Good job, but needs small improvements"
    if score.final >= 50:
        return "Keep practicing, focus on clarity"
    return "Needs more practice, try slower speech"

# ─────────────────────────────
# 🔥 Faster Whisper Transcription
# ─────────────────────────────

def transcribe(audio_path: str) -> str:
    segments, info = model.transcribe(audio_path)

    text = " ".join([s.text for s in segments]).strip()

    return text

# ─────────────────────────────
# Evaluate Pipeline (UNCHANGED endpoint)
# ─────────────────────────────

def evaluate(audio_path, target):
    raw = transcribe(audio_path)

    target_n = normalize(target)
    pred_n = normalize(raw)

    scores = compute_scores(target_n, pred_n)
    mistakes = analyse_mistakes(target_n, pred_n)
    fb = feedback(scores)

    return {
        "target": target_n,
        "predicted": pred_n,
        "scores": scores,
        "feedback": fb,
        "mistakes": mistakes,
        "phoneme_notes": []
    }

# ─────────────────────────────
# API ENDPOINT (SAME)
# ─────────────────────────────

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_audio(
    target_word: str = Form(...),
    file: UploadFile = File(...)
):
    audio = await file.read()

    if not audio:
        raise HTTPException(400, "Empty file")

    start = time.time()
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio)
            temp_path = tmp.name

        result = evaluate(temp_path, target_word)

        return {
            **result,
            "processing_time_ms": int((time.time() - start) * 1000)
        }

    except Exception as e:
        raise HTTPException(500, f"Evaluation failed: {str(e)}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/")
def root():
    return {"status": "ArticuLearn running 🚀"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "faster-whisper"}