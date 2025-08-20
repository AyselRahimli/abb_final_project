# Streamlit Complaint Analyzer — GenAI Edition (Demo, Gemini version)

import os
import re
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# --- Streamlit ---
import streamlit as st

# --- Gemini SDK ---
import google.generativeai as genai

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.neighbors import NearestNeighbors


# =========================
# Helpers & Demo Seed Data
# =========================

CATEGORIES = [
    "ATM Issues","Mobile App Bugs","Card/Payments","Loan/Interest",
    "Fraud & Security","Fees & Charges","Online Banking/Web",
    "Branch Service","Account Opening/KYC","Other",
]

def redact(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = re.sub(r"\b(?:\d[ -]*?){13,19}\b", "[CARD]", text)
    t = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", t)
    t = re.sub(r"\+?\d[\d\s().-]{7,}\d", "[PHONE]", t)
    t = re.sub(r"\bAZ\d{2}[A-Z]{4}\d{20}\b", "[IBAN]", t)
    return t

def quickfit_dummy_classifier():
    seed_text = [
        "ATM swallowed my card and no cash dispensed",
        "ATM out of service and charged fee",
        "Mobile app crashes on login and transfer fails",
        "App shows error during bill payment",
        "Card declined at POS and payment reversed",
        "Double charge on my card transaction",
        "Loan interest rate miscalculated and schedule wrong",
        "Request to change loan interest and term",
        "Suspicious transaction and possible fraud on account",
        "Why was I charged maintenance fee?",
        "Website down, cannot access online banking",
        "Branch staff were rude and slow",
        "KYC document upload failed in account opening",
        "Other unrelated question",
    ]
    seed_y = [
        "ATM Issues","ATM Issues","Mobile App Bugs","Mobile App Bugs",
        "Card/Payments","Card/Payments","Loan/Interest","Loan/Interest",
        "Fraud & Security","Fees & Charges","Online Banking/Web",
        "Branch Service","Account Opening/KYC","Other",
    ]
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
        ("clf", LinearSVC())
    ])
    pipe.fit(seed_text, seed_y)
    return pipe

def ensure_datetime(col):
    if pd.api.types.is_datetime64_any_dtype(col):
        return col
    try:
        return pd.to_datetime(col, errors="coerce")
    except Exception:
        return pd.to_datetime([datetime.utcnow()] * len(col))

def classify_texts(model, texts):
    preds = model.predict(texts)
    try:
        dec = model.decision_function(texts)
        conf = dec.max(axis=1) if dec.ndim > 1 else np.abs(dec)
        conf = 1 / (1 + np.exp(-conf/2))
    except Exception:
        conf = np.full(len(texts), 0.7)
    return preds, conf

def build_nn_index(emb_texts):
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vec.fit_transform(emb_texts)
    nn = NearestNeighbors(n_neighbors=8, metric="cosine").fit(X)
    return vec, nn, X

def retrieve_similar(query, vec, nn, X, df_source, k=5):
    q = vec.transform([query])
    dist, idx = nn.kneighbors(q, n_neighbors=min(k, X.shape[0]))
    rows = df_source.iloc[idx[0]].copy()
    rows["similarity"] = (1 - dist[0]).round(3)
    return rows


# =================
# GenAI (Gemini)
# =================

LLM_MODEL = "gemini-1.5-flash"

def get_gemini_client(api_key: str):
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai
    except Exception:
        return None

def llm_draft_reply(client, complaint_text, category, similar_examples, language="az"):
    fallback_intro = {
        "az": "Hörmətli müştəri, yaşadığınız narahatlığa görə üzr istəyirik.",
        "en": "Dear customer, we’re sorry for the inconvenience.",
    }
    if client is None:
        sim_txt = "\n\nOxşar hallar:\n" + "\n".join([f"- {row[:120]}…" for row in similar_examples]) if similar_examples else ""
        if language == "az":
            return f"{fallback_intro['az']}\n\nKateqoriya: {category}. Müraciətiniz qeydiyyata alındı."
        return f"{fallback_intro['en']}\n\nCategory: {category}. Your case has been registered.{sim_txt}"

    prompt = f"""
language={language}
category={category}
complaint={complaint_text}

similar_resolved_cases:
{similar_examples}

Draft a polite, concise support reply (120–150 words).
    """
    try:
        model = client.GenerativeModel(LLM_MODEL)
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception:
        return llm_draft_reply(None, complaint_text, category, similar_examples, language)

def llm_weekly_summary(client, top_counts, examples_by_cat, start, end):
    if client is None:
        return f"# Weekly Complaint Summary ({start.date()} → {end.date()})\n" + "\n".join(
            [f"## {cat} — {cnt} cases\n" for cat, cnt in top_counts.items()]
        )
    prompt = f"""
Produce a concise Markdown weekly summary for banking complaints.
Time window: {start.date()} → {end.date()}
Top counts: {top_counts}
Examples: {examples_by_cat}
    """
    try:
        model = client.GenerativeModel(LLM_MODEL)
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception:
        return llm_weekly_summary(None, top_counts, examples_by_cat, start, end)


# =================
# Streamlit App
# =================

st.set_page_config(page_title="Complaint Analyzer — Gemini Demo", layout="wide")

if "data" not in st.session_state:
    st.session_state.data = None
if "clf" not in st.session_state:
    st.session_state.clf = quickfit_dummy_classifier()
if "vec" not in st.session_state:
    st.session_state.vec = None
if "nn" not in st.session_state:
    st.session_state.nn = None
if "Xvec" not in st.session_state:
    st.session_state.Xvec = None

# 🔑 Load API key from Streamlit secrets
st.session_state.api_key = st.secrets.get("GEMINI_API_KEY", "")

# Sidebar
st.sidebar.title("Gemini Demo Controls")
st.session_state.api_key = st.sidebar.text_input(
    "Gemini API key (optional)",
    value=st.session_state.api_key,
    type="password"
)

page = st.sidebar.radio("Navigate", [
    "1) Upload & Preview","2) Overview","3) Triage Queue",
    "4) Similar Cases & Draft Reply (GenAI)","5) Weekly Top-3 Report (GenAI)","6) Settings"
])

# (rest of your UI code is unchanged, just calling llm_draft_reply/llm_weekly_summary with get_gemini_client)
