from app.pipeline.near_deduplicate import annotate_near_duplicates
from app.pipeline.near_deduplicate_indexed import (
    annotate_near_duplicates_indexed,
)
from app.pipeline.parallel_extract import (
    benchmark_extraction,
    compare_extraction_results,
    extract_documents_parallel,
    extract_documents_serial,
)


def _documents():
    return [
        {
            "id": "doc_a",
            "html": (
                "<html><head><title>A</title></head><body>"
                "<p>Python is a programming language for data processing.</p>"
                "</body></html>"
            ),
            "clean_text": (
                "Python is a programming language for data processing."
            ),
        },
        {
            "id": "doc_b",
            "html": (
                "<html><head><title>B</title></head><body>"
                "<p>Python is a programming language for data processing.</p>"
                "</body></html>"
            ),
            "clean_text": (
                "Python is a programming language for data processing."
            ),
        },
        {
            "id": "doc_c",
            "html": (
                "<html><head><title>C</title></head><body>"
                "<p>Completely unrelated astronomy content about stars.</p>"
                "</body></html>"
            ),
            "clean_text": (
                "Completely unrelated astronomy content about stars."
            ),
        },
    ]


def test_indexed_near_dedup_matches_baseline_pairs():
    baseline_rows, baseline_pairs = annotate_near_duplicates(
        _documents(),
        threshold=0.75,
    )
    indexed_rows, indexed_pairs, stats = annotate_near_duplicates_indexed(
        _documents(),
        threshold=0.75,
    )

    baseline_set = {
        (pair["id_a"], pair["id_b"])
        for pair in baseline_pairs
    }
    indexed_set = {
        (pair["id_a"], pair["id_b"])
        for pair in indexed_pairs
    }
    assert baseline_set == indexed_set
    assert [row["is_near_duplicate"] for row in baseline_rows] == [
        row["is_near_duplicate"] for row in indexed_rows
    ]
    assert stats["candidate_pair_count"] <= (
        stats["theoretical_pair_count"]
    )


def test_indexed_near_dedup_validates_threshold():
    try:
        annotate_near_duplicates_indexed(_documents(), threshold=1.1)
    except ValueError as exc:
        assert "threshold" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("应拒绝超出范围的 threshold")


def test_parallel_extraction_matches_serial():
    serial_rows = extract_documents_serial(_documents())
    parallel_rows = extract_documents_parallel(_documents(), workers=2)

    comparison = compare_extraction_results(
        serial_rows,
        parallel_rows,
    )
    assert comparison["equal"] is True
    assert comparison["mismatch_count"] == 0


def test_parallel_extraction_benchmark_reports_speed_metrics():
    report = benchmark_extraction(
        _documents(),
        worker_counts=(2,),
    )

    assert report["input_count"] == 3
    assert report["serial"]["output_count"] == 3
    assert len(report["parallel"]) == 1
    assert report["parallel"][0]["correctness"]["equal"] is True
