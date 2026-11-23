import re
import html
from difflib import SequenceMatcher


def cleanup(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def match_most_text(summary, article):
    summary = cleanup(summary)
    article = cleanup(article)

    words = list(re.finditer(r"\b[\w'-]+\b", article))
    article_words = [w.group().lower() for w in words]
    summary_words = re.findall(r"\b[\w'-]+\b", summary.lower())

    if not words or not summary_words:
        return []

    matcher = SequenceMatcher(
        None, article_words, summary_words, autojunk=False
    )

    snippets = []

    for block in matcher.get_matching_blocks():
        if block.size >= 2:
            start = words[block.a].start()
            end = words[block.a + block.size - 1].end()
            snippets.append(article[start:end])

    return snippets


def highlight_text(snippets, text):
    text = str(text or "")

    for snippet in sorted(set(snippets or []), key=len, reverse=True):
        if snippet:
            text = text.replace(
                snippet,
                "<mark>" + html.escape(snippet) + "</mark>"
            )

    return text