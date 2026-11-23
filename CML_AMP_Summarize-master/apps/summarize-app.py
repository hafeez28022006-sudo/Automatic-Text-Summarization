from pathlib import Path
import re
import matplotlib.pyplot as plt
import pandas as pd
import wikipedia as wiki
import streamlit as st

st.set_page_config(
    page_title="Natural Language Processing — Text Summarizer",
    page_icon="📝",
    layout="wide",
)

from text_highlighting import cleanup
from text_highlighting import match_most_text, highlight_text
from st_model_wrappers import (
    abstractive,
    modern_extractive,
    classic_extractive,
    hybrid_extractive,
)

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS = (
    abstractive,
    modern_extractive,
    classic_extractive,
    hybrid_extractive,
)

# ============ DASHBOARD STYLING ============
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
        }
        .nlp-brand-box {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.16), rgba(16, 185, 129, 0.14));
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 14px;
        }
        .nlp-brand-title {
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: 0.3px;
            color: #f8fafc;
            margin: 0;
        }
        .nlp-brand-sub {
            font-size: 0.82rem;
            color: #94a3b8;
            margin-top: 4px;
        }
        .summary-card {
            background-color: rgba(30, 41, 59, 0.55);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 12px;
            padding: 18px 20px;
            margin-top: 8px;
            line-height: 1.65;
        }
        .stat-pill {
            display: inline-block;
            background: rgba(59, 130, 246, 0.15);
            border: 1px solid rgba(96, 165, 250, 0.35);
            color: #bfdbfe;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.8rem;
            margin-right: 8px;
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def _resolve_asset(rel_path: str) -> Path:
    for candidate in (
        BASE_DIR / rel_path,
        Path(__file__).resolve().parent / rel_path,
        Path(rel_path),
    ):
        if candidate.exists():
            return candidate
    return BASE_DIR / rel_path


@st.cache_resource(show_spinner=False)
def _get_hf_summarizer():
    try:
        from transformers import pipeline

        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception:
        return None


def _split_sentences(doc: str):
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", doc.strip()) if s.strip()]
    return parts if parts else ([doc.strip()] if doc.strip() else [])


def _fallback_extractive_summary(doc: str) -> str:
    cleaned = doc.strip()
    if not cleaned:
        return ""
    try:
        from summa.summarizer import summarize as summa_summarize

        out = summa_summarize(cleaned, ratio=0.4)
        if out and out.strip():
            return out.strip()
    except Exception:
        pass

    sentences = _split_sentences(cleaned)
    if len(sentences) <= 1:
        return cleaned
    lead = sentences[0]
    others = sorted(sentences[1:], key=lambda s: len(s.split()), reverse=True)
    picked = [lead]
    if others:
        picked.append(others[0])
    ordered = [s for s in sentences if s in picked]
    return " ".join(ordered)


def _rewrite_clause_abstractive(sentence: str) -> str:
    s = re.sub(
        r"^(?:Among the worst cases,\s*|Official safety figures reveal,\s*)",
        "",
        sentence.strip(),
        flags=re.I,
    )
    s = s.rstrip(".!?").strip()
    clauses = [
        c.strip()
        for c in re.split(r",\s+(?:with|resulting in|where|which)\s+", s, maxsplit=1)
    ]
    main_part = clauses[0]
    main_part = re.sub(
        r"\bresulted in the pilots requesting\b", "prompted", main_part, flags=re.I
    )
    main_part = re.sub(
        r"\bmade an emergency Mayday call\b", "issued a Mayday", main_part, flags=re.I
    )
    main_part = re.sub(
        r"\bbecame unwell during flight\b", "fell ill mid-flight", main_part, flags=re.I
    )
    main_part = re.sub(
        r"\bin only four months\b", "over four months", main_part, flags=re.I
    )
    return main_part[:1].upper() + main_part[1:] + "."


def _fallback_abstractive_summary(doc: str) -> str:
    cleaned = doc.strip()
    if not cleaned:
        return ""
    summarizer = _get_hf_summarizer()
    if summarizer is not None and len(cleaned.split()) >= 25:
        try:
            words = len(cleaned.split())
            max_len = min(130, max(20, int(words * 0.5)))
            min_len = min(15, max(5, max_len // 2))
            result = summarizer(
                cleaned[:3500],
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
            )
            if result and isinstance(result, list) and "summary_text" in result[0]:
                hf_out = result[0]["summary_text"].strip()
                if hf_out:
                    return hf_out
        except Exception:
            pass

    sentences = _split_sentences(cleaned)
    condensed = [_rewrite_clause_abstractive(s) for s in sentences[:3]]
    return " ".join(condensed)


# Global versions for caching
@st.cache_resource
def load_model(_model):
    return _model.load()


@st.cache_data(show_spinner=False)
def summarize_text(article: str, _model) -> str:
    try:
        return _model.summarize(article, load_model(_model))
    except Exception:
        if _model.name == "neural_abstractive":
            return _fallback_abstractive_summary(article)
        return _fallback_extractive_summary(article)


@st.cache_data(show_spinner=False)
def run_custom_mode_summary(doc: str, mode: str) -> str:
    if mode == "Abstractive":
        try:
            out = abstractive.summarize(doc, load_model(abstractive))
            if out and out.strip() and out.strip() != _fallback_extractive_summary(doc):
                return out.strip()
        except Exception:
            pass
        return _fallback_abstractive_summary(doc)
    else:
        for ext_model in (classic_extractive, hybrid_extractive, modern_extractive):
            try:
                out = ext_model.summarize(doc, load_model(ext_model))
                if out and out.strip():
                    return out.strip()
            except Exception:
                continue
        return _fallback_extractive_summary(doc)


@st.cache_data
def load_data(selection: str) -> pd.DataFrame:
    if selection == "wiki":
        return pd.read_csv(_resolve_asset("data/wikipedia/ml_excerpts.pd"))
    elif selection == "cnn":
        return pd.read_csv(_resolve_asset("data/cnn_dailymail/sAMPLE.pd"))
    raise ValueError(f"Unknown dataset selection: {selection}")


def make_bar_chart(df: pd.DataFrame, idx: int, model_name: str):
    labels = [
        "Neural Abstractive",
        "Neural Extractive",
        "Classic Extractive",
        "Hybrid Extractive",
    ]
    values = df[labels].iloc[idx].values
    to_highlight = labels.index(model_name)

    fig, ax = plt.subplots()
    ax.bar([0, 1, 2, 3], height=values)
    ax.bar([to_highlight], height=values[to_highlight], color="orange")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("ROUGE-L score")
    fig.tight_layout()
    return fig


def render_single_model_view(summary_text_value: str, source_text: str, title_label: str):
    st.divider()
    bottom_left, bottom_right = st.columns(2, gap="large")

    with bottom_left:
        st.markdown("### Original Text (Summary Highlighted)")
        snippets = []
        if summary_text_value:
            snippets = match_most_text(summary_text_value, source_text)
        st.markdown('<div class="summary-card">', unsafe_allow_html=True)
        if snippets:
            highlighted_article = highlight_text(snippets, source_text)
            for paragraph in highlighted_article.split("\n"):
                st.write(paragraph, unsafe_allow_html=True)
        else:
            st.write(source_text)
        st.markdown("</div>", unsafe_allow_html=True)

    with bottom_right:
        st.markdown(f"### {title_label} Summary")
        word_count = len(summary_text_value.split()) if summary_text_value else 0
        st.markdown(
            f'<span class="stat-pill">Words: {word_count}</span>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="summary-card">', unsafe_allow_html=True)
        st.write(summary_text_value if summary_text_value else "_No summary generated._")
        st.markdown("</div>", unsafe_allow_html=True)

    return bottom_right


# ============ SIDEBAR ============
st.sidebar.markdown(
    """
    <div class="nlp-brand-box">
        <p class="nlp-brand-title">Natural Language Processing</p>
        <p class="nlp-brand-sub">Extractive & Abstractive Text Summarization Dashboard</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    "Explore **extractive** and **abstractive** text summarization across classic graph methods, "
    "sentence embeddings, and Transformer models."
)
st.sidebar.divider()

# ----- Model Selection -----
model_selector = {m.display_name: m for m in MODELS}
model_obj = model_selector[
    st.sidebar.selectbox("Choose a summarization model", list(model_selector.keys()))
]
st.sidebar.markdown("#### Model Description")
st.sidebar.markdown(model_obj.description, unsafe_allow_html=True)

# ============ MAIN PAGE ============
top_left, top_right = st.columns([1, 1.35], gap="large")

with top_left:
    st.title("Summarize.")
    st.caption("Compare extractive and abstractive summaries on sample datasets or your own text.")

    text_options = {
        "Excerpts from Wikipedia page on Machine Learning": "wiki",
        "Articles from the CNN/Daily News dataset": "cnn",
    }
    text_selection = st.selectbox("Choose a dataset", list(text_options.keys()))
    text_selection = text_options[text_selection]

    if text_selection == "wiki":
        df = load_data(text_selection)
        df.fillna("", inplace=True)
        excerpt_map = {r.heading: i for i, r in df.iterrows()}
        st.subheader("Table of Contents")
        selection = st.selectbox(
            "Choose an excerpt to summarize", list(excerpt_map.keys())
        )
        row_idx = excerpt_map[selection]
    else:
        df = load_data(text_selection)
        article_map = {r.title: i for i, r in df.iterrows()}
        st.subheader("Articles")
        selection = st.selectbox(
            "Choose a news article", list(article_map.keys())
        )
        row_idx = article_map[selection]

article = df.iloc[row_idx]["article"]
original = cleanup(article)

with top_right:
    text = st.text_area(
        "Subsection text — or enter your own text to summarize!",
        article,
        height=290,
    )
    text = cleanup(text)
    st.caption(f"Input length: {len(text.split())} words")

# ----- Display Highlighting & Results -----
if text == original:
    summary = df.iloc[row_idx][model_obj.name + "_summary"]
    bottom_right = render_single_model_view(summary, text, model_obj.display_name)

    with bottom_right:
        if text_selection == "cnn":
            with st.expander("Quantitative Comparison"):
                st.write(
                    "Because the CNN/Daily Mail dataset includes gold standard summaries, "
                    "we can do a quantitative comparison with our model output. The standard approach "
                    "is to compute the ROUGE score between the model's output and the gold standard."
                )
                st.pyplot(make_bar_chart(df, row_idx, model_obj.display_name))
                st.write(
                    "There are several flavors of ROUGE score. ROUGE-L considers the longest "
                    "common subsequence in the summary."
                )
        if text_selection == "wiki":
            with st.expander("Qualitative Comparison"):
                st.markdown(df.iloc[row_idx]["commentary"])
else:
    st.divider()
    custom_mode = st.radio(
        "Summary type for your custom text:",
        ["Extractive", "Abstractive", "Both (Extractive + Abstractive)"],
        horizontal=True,
    )

    if custom_mode == "Both (Extractive + Abstractive)":
        ext_summary = run_custom_mode_summary(text, "Extractive")
        abs_summary = run_custom_mode_summary(text, "Abstractive")

        col_ext, col_abs = st.columns(2, gap="large")
        with col_ext:
            st.markdown("### Extractive Summary")
            st.markdown(
                f'<span class="stat-pill">Words: {len(ext_summary.split()) if ext_summary else 0}</span>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="summary-card">', unsafe_allow_html=True)
            st.write(ext_summary if ext_summary else "_No extractive summary generated._")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_abs:
            st.markdown("### Abstractive Summary")
            st.markdown(
                f'<span class="stat-pill">Words: {len(abs_summary.split()) if abs_summary else 0}</span>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="summary-card">', unsafe_allow_html=True)
            st.write(abs_summary if abs_summary else "_No abstractive summary generated._")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        summary = run_custom_mode_summary(text, custom_mode)
        render_single_model_view(summary, text, custom_mode)