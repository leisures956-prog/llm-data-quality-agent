import re


BOILERPLATE_TERMS = {
    "广告",
    "首页",
    "登录",
    "注册",
    "导航",
    "搜索",
    "帮助",
    "联系我们",
    "版权",
    "版权所有",
}


def _normalise_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip().casefold()


def _is_boilerplate_line(line: str) -> bool:
    tokens = re.findall(
        r"[\u4e00-\u9fff]+|[A-Za-z0-9]+",
        line.casefold(),
    )

    return bool(tokens) and all(
        token in BOILERPLATE_TERMS
        for token in tokens
    )


def repair_text(text: str) -> tuple[str, list[str]]:
    """Remove obvious template lines and repeated lines.

    This is intentionally conservative: it only removes a line when the
    entire line looks like boilerplate or when the same normalized line has
    already occurred.  It does not rewrite meaningful article content.
    """

    seen_lines: set[str] = set()
    repaired_lines: list[str] = []
    actions: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        normalized = _normalise_line(line)

        if not normalized:
            continue

        if _is_boilerplate_line(line):
            actions.append("remove_boilerplate_line")
            continue

        if normalized in seen_lines:
            actions.append("remove_repeated_line")
            continue

        seen_lines.add(normalized)
        repaired_lines.append(line)

    return "\n".join(repaired_lines), sorted(set(actions))


def repair_document(document: dict) -> dict:
    repaired_text, actions = repair_text(
        document.get("clean_text", "")
    )

    return {
        **document,
        "clean_text": repaired_text,
        "repair_applied": bool(actions),
        "repair_actions": actions,
    }
