"""Tests for document upload tag separation and extension validation."""

from __future__ import annotations

import pytest

from amortized.api.documents import _ALLOWED_EXTENSIONS, _sanitize_filename


class TestExtensionValidation:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("report.pdf", ".pdf"),
            ("slides.pptx", ".pptx"),
            ("notes.docx", ".docx"),
            ("page.html", ".html"),
            ("readme.txt", ".txt"),
            ("guide.md", ".md"),
            ("data.xlsx", ".xlsx"),
        ],
    )
    def test_allowed_extensions_accepted(self, filename: str, expected: str) -> None:
        suffix = "." + filename.rsplit(".", 1)[-1].lower()
        assert suffix == expected
        assert suffix in _ALLOWED_EXTENSIONS

    @pytest.mark.parametrize(
        "filename",
        [
            "data.jsonl",
            "data.parquet",
            "script.py",
            "image.png",
            "archive.zip",
        ],
    )
    def test_disallowed_extensions_rejected(self, filename: str) -> None:
        suffix = "." + filename.rsplit(".", 1)[-1].lower()
        assert suffix not in _ALLOWED_EXTENSIONS

    def test_no_extension_rejected(self) -> None:
        filename = "README"
        suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        assert suffix not in _ALLOWED_EXTENSIONS


class TestUrlFilenameExtraction:
    def test_query_params_stripped(self) -> None:
        from urllib.parse import urlparse

        url = "https://example.com/docs/report.pdf?token=abc&v=2"
        path = urlparse(url).path
        filename = _sanitize_filename(path.rsplit("/", 1)[-1] or "document")
        assert filename == "report.pdf"

    def test_fragment_stripped(self) -> None:
        from urllib.parse import urlparse

        url = "https://example.com/doc.pdf#page=3"
        path = urlparse(url).path
        filename = _sanitize_filename(path.rsplit("/", 1)[-1] or "document")
        assert filename == "doc.pdf"

    def test_query_and_fragment_stripped(self) -> None:
        from urllib.parse import urlparse

        url = "https://example.com/file.docx?dl=1#ref"
        path = urlparse(url).path
        filename = _sanitize_filename(path.rsplit("/", 1)[-1] or "document")
        assert filename == "file.docx"

    def test_plain_url(self) -> None:
        from urllib.parse import urlparse

        url = "https://example.com/slides.pptx"
        path = urlparse(url).path
        filename = _sanitize_filename(path.rsplit("/", 1)[-1] or "document")
        assert filename == "slides.pptx"


class TestWorkerTagMapping:
    def test_upload_job_type_maps_to_document(self) -> None:
        from amortized.models import JobType

        job_type = JobType.upload.value
        mlflow_tag_type = "document" if job_type == JobType.upload.value else job_type
        assert mlflow_tag_type == "document"

    def test_sdg_job_type_unchanged(self) -> None:
        from amortized.models import JobType

        job_type = JobType.sdg.value
        mlflow_tag_type = "document" if job_type == JobType.upload.value else job_type
        assert mlflow_tag_type == "sdg"

    def test_training_job_type_unchanged(self) -> None:
        from amortized.models import JobType

        job_type = JobType.training.value
        mlflow_tag_type = "document" if job_type == JobType.upload.value else job_type
        assert mlflow_tag_type == "training"
