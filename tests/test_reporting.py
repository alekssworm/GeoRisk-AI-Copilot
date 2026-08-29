from io import BytesIO

from docx import Document
from pypdf import PdfReader

from app.reporting import build_risk_report_docx, build_risk_report_pdf


def _sample_report() -> dict:
    return {
        "prediction": {
            "dose_rate_usv_h": 0.321,
            "risk_level": "Moderate",
            "advisory": "Increase monitoring frequency.",
            "model_version": "test-model-v1",
            "uncertainty": {
                "lower_usv_h": 0.27,
                "upper_usv_h": 0.39,
                "confidence_label": "medium",
            },
            "distribution_check": {"warning": "One input is outside the training distribution."},
        },
        "scenario_comparison": [
            {
                "name": "Baseline",
                "dose_rate_usv_h": 0.321,
                "risk_level": "Moderate",
                "delta_vs_baseline_usv_h": 0.0,
            },
            {
                "name": "Remediation",
                "dose_rate_usv_h": 0.181,
                "risk_level": "Low",
                "delta_vs_baseline_usv_h": -0.14,
            },
        ],
        "explanation": {
            "top_features": [
                {"feature": "contamination_bq_m2", "direction": "increases", "value": 2.5}
            ]
        },
        "rag_answer": None,
    }


def test_pdf_report_is_readable():
    content = build_risk_report_pdf(_sample_report())

    reader = PdfReader(BytesIO(content))
    assert content.startswith(b"%PDF")
    assert len(reader.pages) >= 1
    assert "RISK ANALYSIS" in reader.pages[0].extract_text()


def test_docx_report_is_readable():
    content = build_risk_report_docx(_sample_report())

    document = Document(BytesIO(content))
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    assert content.startswith(b"PK")
    assert "RISK ANALYSIS" in text
