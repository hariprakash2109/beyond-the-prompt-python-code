# Beyond the Prompt: A Hands-On Guide to Generative AI, LLMs, and Agentic AI with Python

*From Your First AI Application to Intelligent Workflows*

Manuscript, companion code, and build tools for a 15-chapter Kindle eBook for students, fresh graduates, and beginner developers.

## Folder overview

| Path | Contents |
|---|---|
| `chapters/` | Source Markdown, one file per section of the book (edit these) |
| `manuscript.md` | Consolidated manuscript, **generated**; do not edit by hand |
| `examples/` | Runnable chapter examples (`ex02` to `ex10`) and shared `llm_helpers.py` |
| `capstone/` | The Chapter 14 project: `knowledge_assistant/` package, `sample_data/`, tests |
| `tests/` | Tests for the chapter examples |
| `build/` | `build_manuscript.py`, `make_back_matter.py`, `build_formats.sh` |
| `references.md`, `kindle_description.md`, `kindle_keywords.md`, `publication_checklist.md` | Publishing support files |
| `PROGRESS.md` | Status checklist |

## Run the code

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # fill in only if you want live model calls

python -m pytest                     # chapter examples: runs offline
cd capstone && python -m pytest      # capstone: runs offline
```

Tests use scripted fake models, so they need no API key and no network. Live-model code paths (the OpenAI adapters) are **not** exercised by the tests; run them yourself with your own key and a model name available to your account.

## Build the book

```bash
python build/make_back_matter.py     # regenerates back matter and the two Kindle text files
python build/build_manuscript.py     # chapters/*.md -> manuscript.md (expands {{include: ...}} from real source files, builds the TOC)
bash build/build_formats.sh          # DOCX and EPUB into build/out/ (requires pandoc)
```

Code listings in the book come from the real files through `{{include: path}}` directives, so edit the source file, not the manuscript, and rebuild.

Edit placeholders before publishing: author name, year, biography, and the copyright page. Then work through `publication_checklist.md`.

## Status and honesty notes

- See `PROGRESS.md` for what is complete.
- The author did not run any code against a live model provider, nor the optional Sentence Transformers embedder. These are labeled in the text.
- All sample documents, people, and data are fictional.
- `epubcheck` was not available in the authoring environment; validate the EPUB yourself (see the checklist).
## Copyright and Usage

Copyright (c) 2026 Hari Prakash S. All rights reserved.

This repository is provided for personal educational and
reference purposes only.

You may not copy, modify, redistribute, publish, sublicense,
or commercially exploit the source code without prior written
permission from the copyright holder.

See the LICENSE file for the full terms.