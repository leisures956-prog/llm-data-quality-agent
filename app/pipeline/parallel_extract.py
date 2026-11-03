"""Experimental serial/process extraction comparison utilities."""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from app.pipeline.extract import extract_document
from app.pipeline.features import compute_features


def extract_document_with_features(
    document: dict[str, Any],
) -> dict[str, Any]:
    html = str(document.get("html") or "")
    extracted = extract_document(html)
    features = compute_features(
        html=html,
        title=extracted["title"],
        clean_text=extracted["clean_text"],
    )
    return {
        **document,
        "title": extracted["title"],
        "clean_text": extracted["clean_text"],
        **features,
    }


def extract_documents_serial(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [extract_document_with_features(document) for document in documents]


def extract_documents_parallel(
    documents: list[dict[str, Any]],
    workers: int,
) -> list[dict[str, Any]]:
    if workers <= 0:
        raise ValueError("workers 必须是正整数")
    if not documents:
        return []

    chunksize = max(1, len(documents) // (workers * 4))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(
            executor.map(
                extract_document_with_features,
                documents,
                chunksize=chunksize,
            )
        )


def _signature(document: dict[str, Any]) -> tuple[Any, ...]:
    fields = (
        "id",
        "title",
        "clean_text",
        "raw_text_chars",
        "clean_text_chars",
        "paragraph_count",
        "line_count",
        "sentence_count",
        "token_count",
        "link_count",
        "link_ratio",
        "repeated_line_ratio",
        "special_char_ratio",
        "title_body_overlap",
        "removed_text_ratio",
    )
    return tuple(document.get(field) for field in fields)


def compare_extraction_results(
    serial_rows: list[dict[str, Any]],
    parallel_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    serial_by_id = {str(row.get("id")): row for row in serial_rows}
    parallel_by_id = {str(row.get("id")): row for row in parallel_rows}
    all_ids = sorted(set(serial_by_id) | set(parallel_by_id))
    mismatches = [
        document_id
        for document_id in all_ids
        if document_id not in serial_by_id
        or document_id not in parallel_by_id
        or _signature(serial_by_id[document_id])
        != _signature(parallel_by_id[document_id])
    ]
    return {
        "equal": not mismatches,
        "serial_count": len(serial_rows),
        "parallel_count": len(parallel_rows),
        "mismatch_count": len(mismatches),
        "mismatch_ids": mismatches[:20],
    }


def _timed(
    operation,
    input_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    rows = operation()
    elapsed = time.perf_counter() - started
    return rows, {
        "elapsed_seconds": round(elapsed, 4),
        "throughput_docs_per_second": round(
            input_count / elapsed if elapsed > 0 else 0.0,
            4,
        ),
        "output_count": len(rows),
    }


def benchmark_extraction(
    documents: list[dict[str, Any]],
    worker_counts: tuple[int, ...] = (2, 4),
) -> dict[str, Any]:
    serial_rows, serial_metrics = _timed(
        lambda: extract_documents_serial(documents),
        len(documents),
    )
    parallel_results = []
    for workers in worker_counts:
        parallel_rows, metrics = _timed(
            lambda workers=workers: extract_documents_parallel(
                documents,
                workers,
            ),
            len(documents),
        )
        comparison = compare_extraction_results(serial_rows, parallel_rows)
        speedup = (
            serial_metrics["elapsed_seconds"]
            / metrics["elapsed_seconds"]
            if metrics["elapsed_seconds"] > 0
            else 0.0
        )
        parallel_results.append(
            {
                "workers": workers,
                **metrics,
                "speedup_vs_serial": round(speedup, 4),
                "correctness": comparison,
            }
        )

    return {
        "input_count": len(documents),
        "serial": {"workers": 1, **serial_metrics},
        "parallel": parallel_results,
        "serial_rows": serial_rows,
    }
