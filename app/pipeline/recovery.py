from collections import Counter


VALID_DECISIONS = {"keep", "reject"}


def apply_manual_labels(
    documents: list[dict],
    labels: list[dict],
) -> list[dict]:
    """Merge human audit labels and produce the final document decision.

    The machine score remains in ``decision``.  The human-audited result is
    stored separately in ``human_decision`` and ``final_decision`` so that
    the pipeline keeps a complete audit trail.
    """

    documents_by_id = {
        document["id"]: document
        for document in documents
    }

    labels_by_id: dict[str, dict] = {}
    for label in labels:
        document_id = label["id"]
        if document_id in labels_by_id:
            raise ValueError(
                f"重复的人工标签 id：{document_id}"
            )
        if label.get("human_decision") not in VALID_DECISIONS:
            raise ValueError(
                f"无效的人工决策：{document_id}"
            )
        labels_by_id[document_id] = label

    missing_ids = [
        document_id
        for document_id in documents_by_id
        if document_id not in labels_by_id
    ]
    if missing_ids:
        raise ValueError(
            "以下文档缺少人工标签："
            + ", ".join(missing_ids)
        )

    unknown_ids = [
        document_id
        for document_id in labels_by_id
        if document_id not in documents_by_id
    ]
    if unknown_ids:
        raise ValueError(
            "人工标签中存在未知文档："
            + ", ".join(unknown_ids)
        )

    results = []
    for document in documents:
        label = labels_by_id[document["id"]]
        machine_decision = document.get("decision", "review")
        human_decision = label["human_decision"]
        is_false_rejection = bool(
            label.get("is_false_rejection", False)
        )

        if (
            is_false_rejection
            and machine_decision == "reject"
            and human_decision == "keep"
        ):
            recovery_status = "recovered"
        elif machine_decision != human_decision:
            recovery_status = "human_reviewed"
        else:
            recovery_status = "confirmed"

        results.append(
            {
                **document,
                "human_decision": human_decision,
                "is_false_rejection": is_false_rejection,
                "manual_action": label.get("action", ""),
                "manual_reason": label.get("reason", ""),
                "final_decision": human_decision,
                "recovery_status": recovery_status,
            }
        )

    return results


def summarize_recovery(results: list[dict]) -> dict:
    false_rejection_count = sum(
        item["is_false_rejection"]
        for item in results
    )
    recovered_count = sum(
        item["recovery_status"] == "recovered"
        for item in results
    )

    recovery_rate = (
        recovered_count / false_rejection_count * 100
        if false_rejection_count
        else 0.0
    )

    return {
        "total": len(results),
        "machine_decision_counts": dict(
            Counter(item["decision"] for item in results)
        ),
        "final_decision_counts": dict(
            Counter(item["final_decision"] for item in results)
        ),
        "recovery_status_counts": dict(
            Counter(item["recovery_status"] for item in results)
        ),
        "false_rejection_count": false_rejection_count,
        "recovered_count": recovered_count,
        "recovery_rate": round(recovery_rate, 2),
    }
