import attr
import streamlit as st

try:
    from summa.models import neural_extractive as ne
except ImportError:
    ne = None

try:
    from summa.models import neural_abstractive as na
except ImportError:
    na = None

try:
    from summa.models import classic_extractive as ce
except ImportError:
    ce = None


def _unavailable_load(model_label):
    def _load():
        raise ModuleNotFoundError(
            f"'{model_label}' module is not available in this environment."
        )
    return _load


def _unavailable_summarize(model_label):
    def _summarize(document, model):
        st.error(
            f"The '{model_label}' backend (`summa.models`) is unavailable in this deployment."
        )
        return ""
    return _summarize


@attr.s()
class SummarizationModel(object):
    """
    Wrapper class for a summarization model for use in the Summarize. Streamlit app.
    """
    name: str = attr.ib()
    load = attr.ib()
    summarize = attr.ib()
    display_name = attr.ib()
    description: str = attr.ib()

    def __hash__(self):
        return self.name


abstractive = SummarizationModel(
    name="neural_abstractive",
    load=na.load_abstractive_model if na else _unavailable_load("neural_abstractive"),
    summarize=na.abstractive_summary if na else _unavailable_summarize("neural_abstractive"),
    display_name="Neural Abstractive",
    description=(
        "HuggingFace Summarization Pipeline.\n\n HuggingFace provide Transformer models "
        "that perform _abstractive_ summarization. These models ingest a document and "
        "generate text word by word (or token by token) until a summary of a desired "
        "length is achieved."
    ),
)

modern_extractive = SummarizationModel(
    name="neural_extractive",
    load=ne.load_neural_extractive_model if ne else _unavailable_load("neural_extractive"),
    summarize=ne.summarize if ne else _unavailable_summarize("neural_extractive"),
    display_name="Neural Extractive",
    description=(
        "Fine-tuning SentenceBERT.\n\n Here we have a Transformer that performs "
        "_extractive_ summarization. Specifically, a SentenceBERT model is fine-tuned to "
        "identify which sentences in the article _should_ be included in the summary from "
        "those that _should not._ At inference time, the model assigns each sentence a "
        "score and we select the top three as the article summary.\n\n Details on the "
        "training process can be found in our blog post, [Extractive Summarization with "
        "SentenceBERT](https://blog.fastforwardlabs.com/2021/09/21/extractive-summarization-with-sentence-bert.html)."
    ),
)

classic_extractive = SummarizationModel(
    name="classic_extractive",
    load=ce.build_classic_nlp_pipeline if ce else _unavailable_load("classic_extractive"),
    summarize=ce.classic_summary if ce else _unavailable_summarize("classic_extractive"),
    display_name="Classic Extractive",
    description=(
        "TextRank is a classic graph-based ranking algorithm that computes the importance "
        "of each vertex given global information about the entire graph. The basic idea is "
        "that of \"voting\": when two vertices are linked, they \"vote\" for each other, and "
        "votes from important vertices (those with many votes) count for even more.\n\n "
        "In this classic version, each vertex represents a word from the document. The edges "
        "between the vertices (the \"votes\") are the co-occurrence of those words within a "
        "given context window size. After the graph is constructed, the PageRank algorithm "
        "(of search engine fame) computes the recursive scoring. These scores are used to "
        "determine the most important words and phrases in the document; and sentences "
        "containing the top phrases are extracted as a summary."
    ),
)

hybrid_extractive = SummarizationModel(
    name="hybrid_extractive",
    load=ce.build_trf_nlp_pipeline if ce else _unavailable_load("hybrid_extractive"),
    summarize=ce.sentence_summary_trf if ce else _unavailable_summarize("hybrid_extractive"),
    display_name="Hybrid Extractive",
    description=(
        "TextRank + SentenceBERT.\n\n We still use TextRank to build a graph, but now "
        "each vertex represents a sentence from the document, rather than a single word. "
        "We obtain an embedding of each sentence with a SentenceBERT Transformer model. "
        "The edges of the graph are the cosine similarity between each pair of sentence "
        "representations. The PageRank algorithm computes the final importance scores for "
        "each sentence. Sentences with the highest scores are selected as the document summary."
    ),
)