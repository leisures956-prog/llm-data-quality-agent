import re

import trafilatura
from bs4 import BeautifulSoup


BOILERPLATE_TAGS = (
    "script",
    "style",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "noscript",
)


def extract_document(html: str) -> dict[str, str | int]:
    soup = BeautifulSoup(html, "lxml")

    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    # 删除常见的导航、页脚、广告等模板内容
    for tag in soup(BOILERPLATE_TAGS):
        tag.decompose()

    clean_html = str(soup)

    clean_text = trafilatura.extract(
        clean_html,
        include_comments=False,
        include_tables=False,
    ) or ""

    if not clean_text:
        clean_text = soup.get_text("\n")

    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text).strip()

    return {
        "title": title,
        "clean_text": clean_text,
        "char_count": len(clean_text),
    }