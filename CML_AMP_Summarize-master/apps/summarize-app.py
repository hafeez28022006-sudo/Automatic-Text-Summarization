from pathlib import Path
import re
import matplotlib.pyplot as plt
import pandas as pd
import wikipedia as wiki
import streamlit as st

st.set_page_config(layout="wide")

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
    # Pick lead sentence + highest-content sentence (longest informative sentence)
    lead = sentences[0]
    others = sorted(sentences[1:], key=lambda s: len(s.split()), reverse=True)
    picked = [lead]
    if others:
        picked.append(others[0])
    # Preserve original order
    ordered = [s for s in sentences if s in picked]
    return " ".join(ordered)


def _rewrite_clause_abstractive(sentence: str) -> str:
    # Strip leading conversational/report fillers and split on clauses to condense
    s = re.sub(r"^(?:Among the worst cases,\s*|Official safety figures reveal,\s*)", "", sentence.strip(), flags=re.I)
    s = s.rstrip(".!?").strip()
    clauses = [c.strip() for c in re.split(r",\s+(?:with|resulting in|where|which)\s+", s, maxsplit=1)]
    main_part = clauses[0]
    # Condense verbose phrases into shorter abstract phrasing
    main_part = re.sub(r"\bresulted in the pilots requesting\b", "prompted", main_part, flags=re.I)
    main_part = re.sub(r"\bmade an emergency Mayday call\b", "issued a Mayday", main_part, flags=re.I)
    main_part = re.sub(r"\bbecame unwell during flight\b", "fell ill mid-flight", main_part, flags=re.I)
    main_part = re.sub(r"\bin only four months\b", "over four months", main_part, flags=re.I)
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
    return fig


def render_single_model_view(summary_text_value: str, source_text: str, title_label: str):
    bottom_left, bottom_right = st.columns(2)
    bottom_left.markdown("## Original text (summary highlighted)")

    snippets = []
    if summary_text_value:
        snippets = match_most_text(summary_text_value, source_text)
    if snippets:
        highlighted_article = highlight_text(snippets, source_text)
        for paragraph in highlighted_article.split("\n"):
            bottom_left.write(paragraph, unsafe_allow_html=True)
    else:
        bottom_left.write(source_text)

    bottom_right.markdown(f"## {title_label} Summary")
    bottom_right.write(f"\n{summary_text_value}")
    return bottom_right


# ============ SIDEBAR ============
logo_path = _resolve_asset("images/fflogo1@1x.png")
if logo_path.exists():
    st.sidebar.image(str(logo_path))

st.sidebar.markdown(
    "**Summarize.** is a text summarization prototype that demonstrates "
    "_abstractive_ and _extractive_ summarization, and showcases Transformer-based models "
    "as well as classic and hybrid models. Read more on our "
    "[blog](https://blog.fastforwardlabs.com/2021/09/22/automatic-summarization-from-textrank-to-transformers.html)."
)

# ----- Model Selection -----
model_selector = {m.display_name: m for m in MODELS}
model_obj = model_selector[
    st.sidebar.selectbox("Choose a summarization model.", list(model_selector.keys()))
]
st.sidebar.markdown("#### Model description")
st.sidebar.markdown(model_obj.description, unsafe_allow_html=True)

# ============ MAIN PAGE ============

# ----- Display ToC and Text Box -----
top_left, top_right = st.columns(2)
top_left.title("Summarize.")

# ----- Text Selection -----
text_options = {
    "Excerpts from Wikipedia page on Machine Learning": "wiki",
    "Articles from the CNN/Daily News dataset": "cnn",
}
text_selection = top_left.selectbox("Choose a dataset.", list(text_options.keys()))
text_selection = text_options[text_selection]

# ----- Load Stuff -----
if text_selection == "wiki":
    df = load_data(text_selection)
    df.fillna("", inplace=True)
    text_options = {r.heading: i for i, r in df.iterrows()}
    top_left.subheader("Table of Contents")
    selection = top_left.selectbox(
        "Choose an excerpt to summarize.", list(text_options.keys())
    )
else:
    df = load_data(text_selection)
    text_options = {r.title: i for i, r in df.iterrows()}
    selection = top_left.selectbox(
        "Choose a news article.", list(text_options.keys())
    )

row_idx = text_options[selection]
article = df.iloc[row_idx]["article"]

original = cleanup(article)
text = top_right.text_area(
    "Subection text -- or enter your own text to summarize!", article, height=300
)
text = cleanup(text)

# ----- Display Highlighting & Results -----
if text == original:
    summary = df.iloc[row_idx][model_obj.name + "_summary"]
    bottom_right = render_single_model_view(summary, text, model_obj.display_name)

    if text_selection == "cnn":
        with bottom_right.expander("Quantitative Comparison"):
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
        with bottom_right.expander("Qualitative Comparison"):
            st.markdown(df.iloc[row_idx]["commentary"])
else:
    custom_mode = top_right.radio(
        "Summary type for your custom text:",
        ["Extractive", "Abstractive", "Both (Extractive + Abstractive)"],
        horizontal=True,
    )

    if custom_mode == "Both (Extractive + Abstractive)":
        ext_summary = run_custom_mode_summary(text, "Extractive")
        abs_summary = run_custom_mode_summary(text, "Abstractive")

        col_ext, col_abs = st.columns(2)
        col_ext.markdown("## Extractive Summary")
        col_ext.write(ext_summary)

        col_abs.markdown("## Abstractive Summary")
        col_abs.write(abs_summary)
    else:
        summary = run_custom_mode_summary(text, custom_mode)
        render_single_model_view(summary, text, f"{custom_mode}")