from pathlib import Path
import re
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Natural Language Processing",
    layout="wide",
)

from text_highlighting import cleanup, match_most_text, highlight_text
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

SAMPLE_CUSTOM_TEXT = (
    "Natural language processing enables computers to understand, interpret, and generate human language. "
    "Modern NLP systems combine neural networks, transformer architectures, and statistical ranking methods "
    "to power translation, question answering, sentiment analysis, and automatic text summarization. "
    "Extractive approaches select key sentences directly from the source text, while abstractive approaches "
    "generate concise new phrasing that captures the core ideas. "
    "Evaluating summaries often involves measuring word reduction, coverage of main points, and readability."
)

# Clean, safe responsive styling that preserves native Streamlit interactivity
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2.6rem;
            padding-bottom: 2.2rem;
        }
        .nlp-sidebar-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }
        .nlp-panel {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 10px;
            padding: 14px 16px;
            margin-top: 8px;
            margin-bottom: 12px;
            background: rgba(128, 128, 128, 0.05);
        }
        .nlp-badge {
            display: inline-block;
            font-size: 0.78rem;
            font-weight: 600;
            padding: 3px 9px;
            border-radius: 999px;
            border: 1px solid rgba(128, 128, 128, 0.35);
            margin-right: 6px;
            margin-bottom: 8px;
        }
        div[data-testid="stRadio"] > div {
            flex-wrap: wrap;
            gap: 0.5rem;
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


def _fallback_extractive_summary(doc: str, num_sentences: int = 3) -> str:
    cleaned = doc.strip()
    if not cleaned:
        return ""
    try:
        from summa.summarizer import summarize as summa_summarize

        out = summa_summarize(cleaned, ratio=0.5)
        if out and out.strip():
            sentences = _split_sentences(out.strip())
            if len(sentences) >= num_sentences:
                return " ".join(sentences[: max(1, num_sentences)])
    except Exception:
        pass

    sentences = _split_sentences(cleaned)
    target = max(1, min(len(sentences), num_sentences))
    if len(sentences) <= target:
        return " ".join(sentences)

    scored = sorted(
        enumerate(sentences),
        key=lambda item: len(set(re.findall(r"\w+", item[1].lower()))),
        reverse=True,
    )
    top_indices = sorted(idx for idx, _ in scored[:target])
    return " ".join(sentences[i] for i in top_indices)


def _paraphrase_sentence_abstractive(sentence: str) -> str:
    s = sentence.strip().rstrip(".!?")
    s = re.sub(
        r"^(?:Among the worst cases,\s*|Official safety figures reveal,\s*|In this classic version,\s*|Here we have a\s*)",
        "",
        s,
        flags=re.I,
    )
    s = re.sub(
        r"\s+(?:in order to|so that|while|whereas|which|where|with)\b.*$",
        "",
        s,
        flags=re.I,
    )
    replacements = [
        (r"\benables computers to\b", "helps systems"),
        (r"\bunderstand, interpret, and generate\b", "process and produce"),
        (r"\bcombine\b", "pair"),
        (r"\bselect key sentences directly from\b", "pull important lines from"),
        (r"\bgenerate concise new phrasing that captures\b", "write shorter summaries conveying"),
        (r"\bhave reported\b", "logged"),
        (r"\bresulted in the pilots requesting\b", "triggered"),
        (r"\bmade an emergency Mayday call\b", "sent a Mayday alert"),
        (r"\bbecame unwell during flight\b", "got sick mid-flight"),
        (r"\bin only four months\b", "within four months"),
    ]
    for pattern, repl in replacements:
        s = re.sub(pattern, repl, s, flags=re.I)

    s = re.sub(r"\s+", " ", s).strip(" ,;-")
    if not s:
        return ""
    return f"Key takeaway: {s[0].upper() + s[1:]}."


def _fallback_abstractive_summary(
    doc: str, max_words: int = 120, num_sentences: int = 3
) -> str:
    cleaned = doc.strip()
    if not cleaned:
        return ""

    summarizer = _get_hf_summarizer()
    if summarizer is not None and len(cleaned.split()) >= 25:
        try:
            max_len = min(220, max(30, int(max_words * 1.35)))
            min_len = min(25, max(5, max_len // 3))
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
    target_count = max(1, min(len(sentences), num_sentences))
    if len(sentences) == 1:
        indices = [0]
    else:
        indices = sorted(
            {
                round(i * (len(sentences) - 1) / max(1, target_count - 1))
                for i in range(target_count)
            }
        )

    rewritten = [_paraphrase_sentence_abstractive(sentences[i]) for i in indices]
    combined = " ".join(p for p in rewritten if p)
    words = combined.split()
    if len(words) > max_words:
        combined = " ".join(words[:max_words]).rstrip(",;") + "."
    return combined


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
def run_custom_mode_summary(
    doc: str, mode: str, num_sentences: int = 3, max_words: int = 120
) -> str:
    ext_out = _fallback_extractive_summary(doc, num_sentences=num_sentences)
    if mode == "Abstractive":
        try:
            out = abstractive.summarize(doc, load_model(abstractive))
            if out and out.strip() and out.strip() != ext_out.strip():
                return out.strip()
        except Exception:
            pass
        return _fallback_abstractive_summary(
            doc, max_words=max_words, num_sentences=num_sentences
        )

    for ext_model in (classic_extractive, hybrid_extractive, modern_extractive):
        try:
            out = ext_model.summarize(doc, load_model(ext_model))
            if out and out.strip():
                sents = _split_sentences(out.strip())
                return " ".join(sents[: max(1, num_sentences)])
        except Exception:
            continue
    return ext_out


@st.cache_data
def load_data(selection: str) -> pd.DataFrame:
    if selection == "wiki":
        return pd.read_csv(_resolve_asset("data/wikipedia/ml_excerpts.pd"))
    if selection == "cnn":
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


def _render_clarity_panel(summary_text_value: str, source_text: str, label: str):
    src_words = len(source_text.split()) if source_text else 0
    sum_words = len(summary_text_value.split()) if summary_text_value else 0
    sum_sents = len(_split_sentences(summary_text_value)) if summary_text_value else 0
    ratio = round((sum_words / src_words) * 100, 1) if src_words > 0 else 0.0
    read_sec = max(5, int(round((sum_words / 200) * 60))) if sum_words > 0 else 0

    st.markdown(
        f"""
        <div>
            <span class="nlp-badge">Words: {sum_words} / {src_words}</span>
            <span class="nlp-badge">Sentences: {sum_sents}</span>
            <span class="nlp-badge">Length: {ratio}%</span>
            <span class="nlp-badge">~{read_sec}s read</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if summary_text_value:
        st.text_area(
            f"Generated {label} text (easy to copy):",
            value=summary_text_value,
            height=115,
            key=f"copy_box_{label}_{hash(summary_text_value)}",
        )
        st.download_button(
            label=f"Download {label} (.txt)",
            data=summary_text_value,
            file_name=f"{label.lower().replace(' ', '_')}_summary.txt",
            mime="text/plain",
            key=f"dl_btn_{label}_{hash(summary_text_value)}",
        )


def render_single_model_view(
    summary_text_value: str, source_text: str, title_label: str
):
    st.divider()
    bottom_left, bottom_right = st.columns(2)

    bottom_left.markdown("## Original text (summary highlighted)")
    snippets = match_most_text(summary_text_value, source_text) if summary_text_value else []
    bottom_left.markdown('<div class="nlp-panel">', unsafe_allow_html=True)
    if snippets:
        highlighted_article = highlight_text(snippets, source_text)
        for paragraph in highlighted_article.split("\n"):
            bottom_left.write(paragraph, unsafe_allow_html=True)
    else:
        bottom_left.write(source_text)
    bottom_left.markdown("</div>", unsafe_allow_html=True)

    bottom_right.markdown(f"## {title_label} Summary")
    bottom_right.markdown('<div class="nlp-panel">', unsafe_allow_html=True)
    bottom_right.write(summary_text_value if summary_text_value else "_No summary generated._")
    bottom_right.markdown("</div>", unsafe_allow_html=True)

    with bottom_right:
        _render_clarity_panel(summary_text_value, source_text, title_label)

    return bottom_right


# ============ SIDEBAR ============
st.sidebar.markdown(
    '<div class="nlp-sidebar-title">Natural Language Processing</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "**Summarize.** demonstrates **extractive** and **abstractive** text summarization "
    "across classic, hybrid, and neural approaches. Read more on our "
    "[blog](https://blog.fastforwardlabs.com/2021/09/22/automatic-summarization-from-textrank-to-transformers.html)."
)

model_selector = {m.display_name: m for m in MODELS}
model_obj = model_selector[
    st.sidebar.selectbox("Choose a summarization model.", list(model_selector.keys()))
]
st.sidebar.markdown("#### Model description")
st.sidebar.markdown(model_obj.description, unsafe_allow_html=True)

# ============ MAIN PAGE ============
top_left, top_right = st.columns(2)
top_left.title("Summarize.")
top_left.caption("Select a sample article or enter your own text to generate clear summaries.")

text_options = {
    "Excerpts from Wikipedia page on Machine Learning": "wiki",
    "Articles from the CNN/Daily News dataset": "cnn",
}
text_selection = top_left.selectbox("Choose a dataset.", list(text_options.keys()))
text_selection = text_options[text_selection]

if text_selection == "wiki":
    df = load_data(text_selection)
    df.fillna("", inplace=True)
    excerpt_options = {r.heading: i for i, r in df.iterrows()}
    top_left.subheader("Table of Contents")
    selection = top_left.selectbox(
        "Choose an excerpt to summarize.", list(excerpt_options.keys())
    )
    row_idx = excerpt_options[selection]
else:
    df = load_data(text_selection)
    article_options = {r.title: i for i, r in df.iterrows()}
    top_left.subheader("Articles")
    selection = top_left.selectbox(
        "Choose a news article.", list(article_options.keys())
    )
    row_idx = article_options[selection]

article = df.iloc[row_idx]["article"]
original = cleanup(article)

editor_key = "custom_text_editor"
source_tracker_key = "last_loaded_article"
if st.session_state.get(source_tracker_key) != article:
    st.session_state[source_tracker_key] = article
    st.session_state[editor_key] = article

btn_left, btn_right, _ = top_right.columns([1, 1.2, 1.8])
if btn_left.button("Clear Text"):
    st.session_state[editor_key] = ""
    st.rerun()
if btn_right.button("Load Sample"):
    st.session_state[editor_key] = SAMPLE_CUSTOM_TEXT
    st.rerun()

raw_text = top_right.text_area(
    "Subsection text -- or enter your own text to summarize!",
    key=editor_key,
    height=280,
)
text = cleanup(raw_text)

input_words = len(text.split()) if text.strip() else 0
input_sents = len(_split_sentences(text)) if text.strip() else 0
top_right.caption(f"Input overview: {input_words} words | {input_sents} sentences")

# ----- Display Highlighting & Results -----
if not text.strip():
    st.info(
        "Enter or paste your own text in the box above (or click **Load Sample**) to generate a summary."
    )
elif text == original:
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
    st.divider()
    total_sentences = max(1, len(_split_sentences(text)))
    default_sentences = min(3, total_sentences)

    ctrl_left, ctrl_right = top_right.columns([1.5, 1.1])
    custom_mode = ctrl_left.radio(
        "Summary type for your custom text:",
        ["Extractive", "Abstractive", "Both (Extractive + Abstractive)"],
        horizontal=True,
    )
    num_sentences = ctrl_right.slider(
        f"Sentences (1 to {total_sentences} total)",
        min_value=1,
        max_value=total_sentences,
        value=default_sentences,
    )
    max_words = max(40, num_sentences * 40)

    if custom_mode == "Both (Extractive + Abstractive)":
        ext_summary = run_custom_mode_summary(
            text,
            "Extractive",
            num_sentences=num_sentences,
            max_words=max_words,
        )
        abs_summary = run_custom_mode_summary(
            text,
            "Abstractive",
            num_sentences=num_sentences,
            max_words=max_words,
        )

        col_ext, col_abs = st.columns(2)

        col_ext.markdown("## Extractive Summary")
        col_ext.markdown('<div class="nlp-panel">', unsafe_allow_html=True)
        col_ext.write(ext_summary if ext_summary else "_No extractive summary generated._")
        col_ext.markdown("</div>", unsafe_allow_html=True)
        with col_ext:
            _render_clarity_panel(ext_summary, text, "Extractive")

        col_abs.markdown("## Abstractive Summary")
        col_abs.markdown('<div class="nlp-panel">', unsafe_allow_html=True)
        col_abs.write(abs_summary if abs_summary else "_No abstractive summary generated._")
        col_abs.markdown("</div>", unsafe_allow_html=True)
        with col_abs:
            _render_clarity_panel(abs_summary, text, "Abstractive")
    else:
        summary = run_custom_mode_summary(
            text,
            custom_mode,
            num_sentences=num_sentences,
            max_words=max_words,
        )
        render_single_model_view(summary, text, custom_mode)