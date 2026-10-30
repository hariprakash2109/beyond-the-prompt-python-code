import json

import pytest

from ex05_job_analyzer import AnalysisError, JobAnalysis, analyze_job, extract_json_object

POSTING = """Junior Python Developer at Acme Labs.
Requirements: 1 year of Python experience, familiarity with SQL, and good communication.
Salary: competitive."""


def good_json():
    return json.dumps({
        "job_title": "Junior Python Developer", "company": "Acme Labs",
        "experience_level": "entry",
        "required_skills": [
            {"skill": "Python", "evidence": "1 year of Python experience"},
            {"skill": "SQL", "evidence": "familiarity with SQL"},
        ],
        "salary_mentioned": True, "notes": [],
    })


def test_happy_path():
    result = analyze_job(POSTING, lambda prompt: good_json())
    assert isinstance(result, JobAnalysis)
    assert [s.skill for s in result.required_skills] == ["Python", "SQL"]


def test_code_fences_are_tolerated():
    assert extract_json_object("```json\n{\"a\": 1}\n```") == {"a": 1}


def test_invented_evidence_is_removed_and_noted():
    data = json.loads(good_json())
    data["required_skills"].append({"skill": "Kubernetes", "evidence": "five years of Kubernetes"})
    result = analyze_job(POSTING, lambda p: json.dumps(data))
    assert "Kubernetes" not in [s.skill for s in result.required_skills]
    assert any("Kubernetes" in n for n in result.notes)


def test_invalid_json_raises_analysis_error():
    with pytest.raises(AnalysisError):
        analyze_job(POSTING, lambda p: "Sure! Here is the JSON: nope")


def test_schema_violation_raises():
    bad = json.dumps({"experience_level": "wizard"})
    with pytest.raises(AnalysisError):
        analyze_job(POSTING, lambda p: bad)


def test_empty_posting_rejected_without_calling_model():
    def explode(prompt):
        raise AssertionError("model should not be called")
    with pytest.raises(AnalysisError):
        analyze_job("   ", explode)


def test_posting_is_wrapped_in_data_markers():
    seen = {}
    analyze_job(POSTING, lambda p: seen.setdefault("p", p) and good_json())
    assert "<<<POSTING" in seen["p"] and "POSTING>>>" in seen["p"]
