import json
from pathlib import Path

from ex05_job_analyzer import analyze_job

POSTING = (Path(__file__).resolve().parents[1] / "examples" / "data" / "job_posting.txt").read_text()

GOOD_REPLY = json.dumps({
    "job_title": "Junior Data Analyst",
    "company": "Northwind Learning",
    "experience_level": "entry",
    "required_skills": [
        {"skill": "SQL", "evidence": "Strong SQL skills"},
        {"skill": "Python", "evidence": "Experience with Python (pandas)"},
        {"skill": "Kubernetes", "evidence": "Hands-on Kubernetes experience"},  # invented by the "model"
    ],
    "salary_mentioned": False,
    "notes": ["Title says entry-level but posting asks for 5+ years of experience"],
})


def test_sample_posting_exists_and_contains_the_contradiction():
    assert "entry-level" in POSTING and "5+ years" in POSTING


def test_real_quotes_kept_invented_quote_removed():
    result = analyze_job(POSTING, lambda prompt: GOOD_REPLY)
    kept = {s.skill for s in result.required_skills}
    assert kept == {"SQL", "Python"}
    assert any("Kubernetes" in n for n in result.notes)
    assert any("5+ years" in n for n in result.notes)
