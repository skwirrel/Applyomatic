
# Auto‑apply (LLM)

Automates drafting a tailored covering letter and CV for a given job role using OpenAI's Responses API with structured outputs. 

It uses local config files to hold personal information and guides the user through improving and approving suggested edits. Designed for human-in-the-loop workflows where final submission decisions remain manual.

---

## Features

- Uses OpenAI's **Responses API** with **structured outputs** (JSON Schema)
- Automatically assesses relevance of your CV skills and past roles to a job description
- Composes a clean Markdown CV
- Generates a persuasive covering letter tailored to the job
- Suggests improvements to your CV and lets you **review, edit, delete, and apply** them interactively
- Supports single-shot and scheduled retry modes (waits 30–90 days)

---

## Getting started

```bash
npm install
cp .env.example .env
cp config/cv.base.json.example config/cv.base.json
cp config/covering-letter-notes.example.md config/covering-letter-notes.md
cp config/job.config.json config/job.config.json
```

Edit the three config files with your own details — your CV, the job you're targeting, and any free-form notes to guide the covering letter.

Then:

```bash
npm run once    # generate one CV + letter
npm run daemon  # run periodically (random 30–90 day intervals)
```

> ⚠️ Do not commit `.env`, `cv.base.json`, or your personal notes to version control.

---

## Configuration overview

### `.env`
```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.3
```

### `config/job.config.json`
```json
{
  "job": "applications to the No10 Innovation Fellowship",
  "minReapplyDays": 30,
  "maxReapplyDays": 90,
  "maxAttempts": 3,
  "daemon": false
}
```

### `config/cv.base.json`
Contains your personal details, skills, job history, and qualifications. Keep private.

### `config/covering-letter-notes.md`
Free-form notes to guide tone and emphasis in the covering letter. Markdown accepted.

---

## Interactive CV polishing

After your CV is generated, the tool will:

1. Use the LLM to suggest edits (spelling, formatting, clarity, impact, etc.)
2. Show each suggestion with its category and location
3. Let you delete, edit, or approve any suggestions via command line
4. Apply **all** approved changes in one go using a structured prompt

Commands:
```
d <n>[,n]       delete suggestions (e.g. d 2,4)
d a            delete all
e <n>          edit a suggestion interactively
c              continue and apply approved changes
q              quit without changes
```

---

## Output and review

CVs and covering letters are printed to standard output. Redirect them to files if you wish:

```bash
npm run once > output_$(date +%F).txt
```

No files are written by default.

---

## Technical notes

- Uses `text.format: json_schema` for structured outputs in OpenAI's Responses API
- Uses `gpt-4o-mini` by default; compatible with `gpt-4o`, `gpt-4.1`, and newer
- Node 18.17+ required (due to API and syntax support)
- Built-in scheduler waits between reapplication attempts (randomised)

---


## Disclaimer

This tool does **not** submit applications for you. It prepares drafts for human review.

The quality of suggestions and compositions depends on the input data and job description retrieved. Always check before submitting.

