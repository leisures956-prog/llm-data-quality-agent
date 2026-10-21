import re
from collections import Counter

from bs4 import BeautifulSoup


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9]+")


def _tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _paragraphs(text: str) -> list[str]:
    paragraphs = [
        part.strip()
        for part in re.split(r"\n\s*\n", text)
        if part.strip()
    ]

    if not paragraphs and text.strip():
        return [text.strip()]

    return paragraphs


def _repeated_line_ratio(text: str) -> float:
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return 0.0

    counts = Counter(lines)
    repeated_count = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    return repeated_count / len(lines)


def _special_char_ratio(text: str) -> float:
    characters = [char for char in text if not char.isspace()]

    if not characters:
        return 0.0

    special_count = sum(
        not (
            char.isalnum()
            or "\u4e00" <= char <= "\u9fff"
        )
        for char in characters
    )

    return special_count / len(characters)


def compute_features(
    html: str,
    title: str,
    clean_text: str,
) -> dict[str, int | float]:
    soup = BeautifulSoup(html, "lxml")

    raw_text = soup.get_text(" ", strip=True)
    raw_text_chars = len(raw_text)
    clean_text_chars = len(clean_text)

    links = soup.find_all("a")
    link_text = " ".join(
        link.get_text(" ", strip=True)
        for link in links
    )

    link_ratio = len(link_text) / max(raw_text_chars, 1)

    title_tokens = set(_tokens(title))
    body_tokens = set(_tokens(clean_text))

    if title_tokens:
        title_body_overlap = len(
            title_tokens & body_tokens
        ) / len(title_tokens)
    else:
        title_body_overlap = 0.0

    paragraphs = _paragraphs(clean_text)

    html_paragraph_count = len(
        soup.find_all("p")
    )

    paragraph_count = (
        html_paragraph_count
        or len(paragraphs)
    )
    lines = [
        line.strip()
        for line in clean_text.splitlines()
        if line.strip()
    ]

    sentence_count = len(
        re.findall(r"[.!?。！？]+", clean_text)
    )

    if clean_text and sentence_count == 0:
        sentence_count = 1

    if raw_text_chars:
        removed_text_ratio = max(
            0.0,
            1 - clean_text_chars / raw_text_chars,
        )
    else:
        removed_text_ratio = 0.0

    return {
        "raw_text_chars": raw_text_chars,
        "clean_text_chars": clean_text_chars,
        "paragraph_count":  paragraph_count,
        "line_count": len(lines),
        "sentence_count": sentence_count,
        "token_count": len(_tokens(clean_text)),
        "link_count": len(links),
        "link_ratio": round(link_ratio, 4),
        "repeated_line_ratio": round(
            _repeated_line_ratio(clean_text),
            4,
        ),
        "special_char_ratio": round(
            _special_char_ratio(clean_text),
            4,
        ),
        "title_body_overlap": round(
            title_body_overlap,
            4,
        ),
        "removed_text_ratio": round(
            removed_text_ratio,
            4,
        ),
    }