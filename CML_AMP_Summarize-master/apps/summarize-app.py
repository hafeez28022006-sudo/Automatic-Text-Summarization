from pathlib import Path
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


# Global versions for caching
@st.cache_resource
def load_model(_model):
    return _model.load()


@st.cache_data(show_spinner=False)
def summarize_text(article: str, _model) -> str:
    return _model.summarize(article, load_model(_model))


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


def render_single_model_view(summary_text_value: str, source_text: str, selected_model):
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

    bottom_right.markdown(f"## {selected_model.display_name} Summary")
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
    bottom_right = render_single_model_view(summary, text, model_obj)

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
    run_all_four = top_right.checkbox(
        "Apply all 4 models (Neural Abstractive + all 3 Extractive) to custom text",
        value=True,
    )

    if run_all_four:
        st.markdown("## All Four Models — Summary Comparison")
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)
        grid_cells = [row1_col1, row1_col2, row2_col1, row2_col2]

        for m, cell in zip(MODELS, grid_cells):
            cell.markdown(f"### {m.display_name}")
            try:
                model_summary = summarize_text(text, m)
                cell.write(
                    model_summary if model_summary else "_No summary generated._"
                )
            except Exception as exc:
                cell.warning(f"Could not run {m.display_name}: {exc}")
    else:
        summary = summarize_text(text, model_obj)
        render_single_model_view(summary, text, model_obj)