# Streamlit Complaint Analyzer — GenAI Edition (Demo)

import os
import re
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# --- Safe import: streamlit ---
try:
    import streamlit as st
except Exception as e:
    raise RuntimeError("Streamlit is required to run this app. Install with `pip install streamlit`.\n" + str(e))

# --- Safe import: OpenAI SDK (optional) ---
have_llm = True
try:
    from openai import OpenAI  # >=1.0 style SDK
except Exception:
    have_llm = False

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.neighbors import NearestNeighbors

# =========================
# Helpers & Demo Seed Data
# =========================

CATEGORIES = [
    "ATM Issues",
    "Mobile App Bugs",
    "Card/Payments",
    "Loan/Interest",
    "Fraud & Security",
    "Fees & Charges",
    "Online Banking/Web",
    "Branch Service",
    "Account Opening/KYC",
    "Other",
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
        "ATM Issues","ATM Issues",
        "Mobile App Bugs","Mobile App Bugs",
        "Card/Payments","Card/Payments",
        "Loan/Interest","Loan/Interest",
        "Fraud & Security",
        "Fees & Charges",
        "Online Banking/Web",
        "Branch Service",
        "Account Opening/KYC",
        "Other",
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
        if dec.ndim == 1:
            conf = np.abs(dec)
        else:
            conf = dec.max(axis=1)
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

# ===================
# GenAI Integrations
# ===================

LLM_MODEL = "gpt-4o-mini"  # change as needed

def get_llm_client(api_key: str):
    if not have_llm:
        return None
    if not api_key:
        return None
    try:
        client = OpenAI(api_key=api_key)
        return client
    except Exception:
        return None


def llm_draft_reply(client, complaint_text: str, category: str, similar_examples: list, language: str = "az") -> str:
    fallback_intro = {
        "az": "Hörmətli müştəri, yaşadığınız narahatlığa görə üzr istəyirik.",
        "en": "Dear customer, we’re sorry for the inconvenience.",
    }
    if client is None:
        sim_txt = "\n\nOxşar hallar:\n" + "\n".join([f"- {row[:120]}…" for row in similar_examples]) if similar_examples else ""
        if language == "az":
            return f"{fallback_intro['az']}\n\nKateqoriya: {category}. Müraciətiniz qeydiyyata alındı və məsələ üzrə komanda ilə yoxlanılır. Təhlükəsizlik məqsədilə şəxsiyyət təsdiqini mobil tətbiq və ya 937 nömrəsi ilə tamamlayın.{sim_txt}"
        return f"{fallback_intro['en']}\n\nCategory: {category}. Your case has been registered and is being reviewed by our team. For security, please complete identity verification via the mobile app or our hotline.{sim_txt}"

    system_prompt = (
        "You are a polite, compliant ABB banking support agent. Draft a concise, empathetic reply to the customer. "
        "Constraints: Apologize once; avoid promises; avoid internal details; include a clear next step; keep 120-150 words; "
        "use Azerbaijani if language='az' else English."
    )
    sim_snippets = "\n".join([f"- {s[:200]}" for s in similar_examples]) if similar_examples else "(no examples)"
    user_prompt = (
        f"language={language}\ncategory={category}\ncomplaint=\n{complaint_text}\n\n"
        f"similar_resolved_cases=\n{sim_snippets}\n\nDraft the reply now."
    )
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return llm_draft_reply(None, complaint_text, category, similar_examples, language)


def llm_weekly_summary(client, top_counts: dict, examples_by_cat: dict, start: datetime, end: datetime) -> str:
    # Fallback template if no LLM
    if client is None:
        lines = [f"# Weekly Complaint Summary ({start.date()} → {end.date()})\n"]
        for cat, cnt in top_counts.items():
            lines.append(f"## {cat} — {cnt} cases\n")
            lines.append("**What we're seeing:** Spike in tickets relative to prior week.\n")
            lines.append("**Likely causes:** • Known defect/incident • User flow friction • External dependency issues\n")
            lines.append("**Suggested owner:** Assign to Ops/IT; prepare comms template.\n")
        return "\n".join(lines)

    sys = (
        "You are a banking CX analyst. Produce a concise Markdown weekly summary for management. "
        "For each category: 1) What’s happening (2–3 sentences) 2) Likely root causes (bullets) 3) Suggested owner."
    )
    payload = {
        "window": {"start": str(start.date()), "end": str(end.date())},
        "top_counts": top_counts,
        "examples": {k: [e[:220] for e in v[:20]] for k, v in examples_by_cat.items()},
    }
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return llm_weekly_summary(None, top_counts, examples_by_cat, start, end)

# =================
# Streamlit App UI
# =================

st.set_page_config(page_title="Complaint Analyzer — GenAI Demo", layout="wide")

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
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")

# Sidebar
st.sidebar.title("GenAI Demo Controls")
st.sidebar.caption("No login required. LLM optional — add your API key to enable.")
st.session_state.api_key = st.sidebar.text_input("OpenAI API key (optional)", value=st.session_state.api_key, type="password")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "1) Upload & Preview",
    "2) Overview",
    "3) Triage Queue",
    "4) Similar Cases & Draft Reply (GenAI)",
    "5) Weekly Top-3 Report (GenAI)",
    "6) Settings"
])

# Page 1: Upload & Preview
if page.startswith("1"):
    st.title("Upload & Preview")
    up = st.file_uploader("Upload complaints CSV", type=["csv"]) 
    st.markdown("Required columns: **id**, **text**, **source**, **created_at**.")
    if up:
        df = pd.read_csv(up)
        if "id" not in df.columns:
            df["id"] = np.arange(1, len(df)+1)
        if "text" not in df.columns:
            st.error("Column 'text' is required.")
            st.stop()
        df["created_at"] = ensure_datetime(df.get("created_at", pd.Series([datetime.utcnow()]*len(df))))
        df["source"] = df.get("source", "upload")
        df["language"] = df.get("language", "az")
        df["redacted_text"] = df["text"].astype(str).apply(redact)
        preds, conf = classify_texts(st.session_state.clf, df["redacted_text"].tolist())
        df["pred_category"] = preds
        df["confidence"] = conf
        st.session_state.data = df
        st.subheader("Preview")
        st.dataframe(df.head(20), use_container_width=True)
        st.success("Data loaded and auto-categorized.")

# Page 2: Overview
elif page.startswith("2"):
    st.title("Overview")
    df = st.session_state.data
    if df is None:
        st.info("Upload data first in 'Upload & Preview'.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total complaints", len(df))
        last7 = df[df["created_at"] >= (datetime.utcnow() - timedelta(days=7))]
        col2.metric("Last 7 days", len(last7))
        col3.metric("Unique categories", int(df["pred_category"].nunique()))
        col4.metric("Avg confidence", f"{df['confidence'].mean():.2f}")
        st.subheader("Top categories (last 7 days)")
        st.bar_chart(last7["pred_category"].value_counts().head(10))
        st.subheader("Volume over time (daily)")
        st.line_chart(df.set_index("created_at").resample("D").size())

# Page 3: Triage Queue
elif page.startswith("3"):
    st.title("Triage Queue")
    df = st.session_state.data
    if df is None:
        st.info("Upload data first in 'Upload & Preview'.")
    else:
        cols = st.columns(4)
        cat = cols[0].multiselect("Category", sorted(df["pred_category"].unique()), [])
        src = cols[1].multiselect("Source", sorted(df["source"].astype(str).unique()), [])
        date_from = cols[2].date_input("From", (datetime.utcnow() - timedelta(days=30)).date())
        date_to = cols[3].date_input("To", datetime.utcnow().date())
        mask = (df["created_at"].dt.date >= date_from) & (df["created_at"].dt.date <= date_to)
        if cat:
            mask &= df["pred_category"].isin(cat)
        if src:
            mask &= df["source"].astype(str).isin(src)
        view = df[mask].copy().sort_values("created_at", ascending=False)
        st.dataframe(view[["id","created_at","source","pred_category","confidence","redacted_text"]], use_container_width=True, height=480)
        st.session_state.selected_id = st.number_input("Select ID to open", min_value=int(view["id"].min()) if len(view) else 0, max_value=int(view["id"].max()) if len(view) else 0, step=1)
        if st.button("Open selected") and len(view):
            row = view[view["id"]==st.session_state.selected_id].head(1)
            if row.empty:
                st.warning("ID not found in current filter.")
            else:
                st.subheader(f"Ticket #{int(row['id'].iloc[0])}")
                st.write("**Created:**", row["created_at"].iloc[0])
                st.write("**Source:**", str(row["source"].iloc[0]))
                st.write("**Category:**", row["pred_category"].iloc[0], "| **Confidence:**", f"{row['confidence'].iloc[0]:.2f}")
                st.write("**Text (redacted):**")
                st.info(row["redacted_text"].iloc[0])
                st.session_state.current_row = int(row["id"].iloc[0])
                st.success("Go to 'Similar Cases & Draft Reply (GenAI)' to continue →")

# Page 4: Similar Cases & Draft Reply (GenAI)
elif page.startswith("4"):
    st.title("Similar Cases & Draft Reply (GenAI)")
    df = st.session_state.data
    if df is None:
        st.info("Upload data first in 'Upload & Preview'.")
    else:
        if st.session_state.vec is None:
            vec, nn, X = build_nn_index(df["redacted_text"].tolist())
            st.session_state.vec, st.session_state.nn, st.session_state.Xvec = vec, nn, X
        vec, nn, X = st.session_state.vec, st.session_state.nn, st.session_state.Xvec
        id_list = sorted(df["id"].astype(int).tolist())
        current_id = st.number_input("Ticket ID", min_value=id_list[0] if id_list else 0, max_value=id_list[-1] if id_list else 0, step=1, value=id_list[0] if id_list else 0)
        row = df[df["id"]==current_id]
        if row.empty:
            st.warning("Invalid ID.")
        else:
            text = row["redacted_text"].iloc[0]
            cat = row["pred_category"].iloc[0]
            sim = retrieve_similar(text, vec, nn, X, df, k=6)
            st.subheader("Similar resolved cases (nearest neighbors)")
            st.dataframe(sim[["id","pred_category","similarity","redacted_text"]], use_container_width=True, height=280)
            lang = st.selectbox("Reply language", ["az", "en"], index=0)
            k = st.slider("How many examples to include", 0, 5, 3)
            examples = sim["redacted_text"].head(k).tolist()
            client = get_llm_client(st.session_state.api_key)
            draft = llm_draft_reply(client, text, cat, examples, language=lang)
            st.subheader("Draft reply (editable)")
            edited = st.text_area("", draft, height=240)
            st.download_button("Download reply (.txt)", edited.encode("utf-8"), file_name=f"reply_{current_id}.txt")
            if client is None:
                st.info("Tip: Add an OpenAI API key in the sidebar to enable real GenAI replies. Fallback template used.")

# Page 5: Weekly Top-3 Report (GenAI)
elif page.startswith("5"):
    st.title("Weekly Top-3 Report (GenAI)")
    df = st.session_state.data
    if df is None:
        st.info("Upload data first in 'Upload & Preview'.")
    else:
        end_d = datetime.utcnow().date()
        start_d = end_d - timedelta(days=7)
        d1 = st.date_input("From", start_d)
        d2 = st.date_input("To", end_d)
        mask = (df["created_at"].dt.date >= d1) & (df["created_at"].dt.date <= d2)
        week = df[mask]
        counts = week["pred_category"].value_counts()
        top3 = counts.head(3)
        st.subheader("Top-3 categories")
        st.bar_chart(top3)
        examples_by_cat = {c: week[week["pred_category"]==c]["redacted_text"].head(60).tolist() for c in top3.index}
        client = get_llm_client(st.session_state.api_key)
        report_md = llm_weekly_summary(client, top3.to_dict(), examples_by_cat, datetime.combine(d1, datetime.min.time()), datetime.combine(d2, datetime.min.time()))
        st.subheader("Draft report (Markdown)")
        st.code(report_md, language="markdown")
        st.download_button("Download report (.md)", report_md.encode("utf-8"), file_name="weekly_report.md")
        if client is None:
            st.info("Tip: Add an OpenAI API key in the sidebar to enable real GenAI summaries. Fallback template used.")

# Page 6: Settings
elif page.startswith("6"):
    st.title("Settings (Demo)")
    st.caption("Local-only settings for the demo.")
    st.write("**Categories:**", ", ".join(CATEGORIES))
    st.write("**PII Redaction:** card, email, phone, AZ IBAN")
    st.write("**Classifier:** TF-IDF + LinearSVC (demo)")
    st.write(f"**GenAI Model:** {LLM_MODEL} (optional)")
    c1, c2 = st.columns(2)
    if c1.button("Reset demo classifier"):
        st.session_state.clf = quickfit_dummy_classifier()
        st.success("Classifier reset.")
    if c2.button("Rebuild retrieval index"):
        df = st.session_state.data
        if df is None:
            st.warning("Load data first.")
        else:
            vec, nn, X = build_nn_index(df["redacted_text"].tolist())
            st.session_state.vec, st.session_state.nn, st.session_state.Xvec = vec, nn, X
            st.success("Index rebuilt.")
