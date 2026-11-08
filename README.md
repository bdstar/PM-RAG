# PM-RAG

**An Evidence-Grounded AI Framework for Corrective-Action Decision Support in Industrial Capital Project Execution.**

PM-RAG is a research-oriented web platform that ingests real NYC capital-project
reporting data, detects execution problems (schedule slippage, budget variance,
repeated forecast revisions, …), and produces **corrective-action recommendations
that are grounded only in retrieved evidence**. It is built to study retrieval,
grounding, abstention, and actionability — not to act as an autonomous chatbot.

Every recommendation cites the source records it relies on, is validated for
factual grounding (NLI + lexical fallback), **abstains when evidence is
insufficient**, and is flagged for **human review** before any consequential
action. No fabricated numbers, project IDs, dates, or experimental results are
ever presented as genuine.

---

## Key properties

- **Evidence-or-abstain.** The generator may only use retrieved evidence; claims
  are checked for grounding and the system abstains rather than guess.
- **Human-in-the-loop.** Corrective actions are never auto-executed;
  consequential categories require authorized human review.
- **Untrusted documents.** Retrieved content is sanitized against prompt
  injection and never treated as instructions.
- **No fabricated results.** Experiment metrics are only shown once a run has
  actually executed and stored them (with bootstrap confidence intervals).
- **Runs out-of-the-box, offline.** SQLite + deterministic stub LLM by default;
  no paid API, DB server, or GPU required. Real models load automatically when
  available, with graceful fallbacks (hashing embedder, pure-Python BM25,
  lexical NLI/reranker).
- **No hard-coded credentials.** All DB/model settings come from the environment.

---

## Architecture

```
dataset discovery ─▶ CSV inspection ─▶ content-based classification ─▶ import
      │                                                                   │
      ▼                                                                   ▼
project-state analysis (schedule / budget / financial snapshots)   normalization
      │                                                                   │
      ▼                                                                   ▼
scenario detection (rule-generated execution problems)  ──▶  evidence indexing
                                                                          │
                                           hybrid retrieval (BM25 + dense + RRF)
                                                                          │
                                                    cross-encoder reranking
                                                                          │
                                      evidence-only / LLM action generation
                                                                          │
                              citation validation ─▶ NLI grounding ─▶ abstention
                                                                          │
                                 experiment benchmarking ─▶ research-result visuals
```

### Project layout

```
app/
├── core/            config (env-only), logging, security (injection defence, ACLs)
├── database/        SQLAlchemy 2.0 models (39 tables) + connection/init
├── ingestion/       csv_inspector, schema_mapper (classify), normalizer, importer
├── project_analysis/ schedule_analysis, budget_analysis, scenario_detection
├── rag/             embeddings, vector_store, bm25, hybrid_retriever, reranker,
│                    generator, grounding, llm_provider
├── recommendations/ service (orchestration), validation, schemas
├── experiments/     runner, baselines, metrics, benchmark
├── services/        background jobs, stats
├── api/routes/      system, datasets, projects, scenarios, retrieval,
│                    recommendations, experiments, research_results, pages
├── templates/       Jinja2 + Bootstrap 5 pages (10 UI pages)
└── static/          app.js (API client, polling, charts), css
scripts/run_pipeline.py   one-shot bootstrap (ingest → analyze → detect → index)
tests/                    pytest unit + integration suite
```

---

## Quickstart (Windows / PowerShell)

> Requires Python 3.10+. GPU optional.

```powershell
# 1. (Recommended) virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure (defaults work out-of-the-box: SQLite + stub LLM)
Copy-Item .env.example .env

# 4. Bootstrap the pipeline from the NYC CSVs
#    (dataset dir defaults to ..\dataset relative to the repo)
python scripts/run_pipeline.py

# 5. Run the API + research dashboard
python -m uvicorn app.main:app --reload --port 8000
```

Open **http://127.0.0.1:8000**. Interactive API docs are at **/docs**.

### Bootstrap options

```powershell
python scripts/run_pipeline.py --limit 5000        # cap rows per CSV (fast demo)
python scripts/run_pipeline.py --index-limit 2000  # cap evidence chunks per type
python scripts/run_pipeline.py --skip-index        # ingest + analyze only
```

---

## Configuration (`.env`)

Nothing is hard-coded; every value below has a safe default.

| Variable | Default | Notes |
|----------|---------|-------|
| `DB_BACKEND` | `sqlite` | `sqlite` \| `mysql` \| `postgresql`. SQLite needs no server. |
| `DB_HOST` / `DB_PORT` / `DB_NAME` / `DB_USER` / `DB_PASSWORD` | `localhost` / `3306` / `pm_rag` / `root` / *(empty)* | Used to build the URL for MySQL/PostgreSQL. |
| `DATABASE_URL` | *(unset)* | Explicit SQLAlchemy URL; overrides the above. |
| `DATASET_DIR` | `..\dataset` | Where the source CSVs live (never modified). |
| `LLM_BACKEND` | `huggingface` | `huggingface` \| `openai` \| `ollama` \| `stub`. Falls back to `stub` if a model can't load. |
| `LLM_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Any HF causal LM. |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Falls back to a hashing embedder if unavailable. |
| `RERANKER_ENABLED` / `NLI_ENABLED` | `true` | Cross-encoder rerank / NLI grounding (lexical fallback if models missing). |
| `TOP_K` / `BM25_WEIGHT` / `RRF_K` | `8` / `0.5` / `60` | Hybrid retrieval tuning. |
| `ABSTENTION_GROUNDING_THRESHOLD` | `0.5` | Per-claim entailment probability to count as supported. |
| `ABSTENTION_MIN_SUPPORT_RATE` | `0.01` | Fraction of factual claims that must be supported; raise (e.g. `0.5`) for the stricter H4 ablation. |

### Using MySQL

```powershell
# in .env
DB_BACKEND=mysql
DB_NAME=pm_rag
DB_USER=root
DB_PASSWORD=
```

Create the database first (`CREATE DATABASE pm_rag CHARACTER SET utf8mb4;`) and
ensure `PyMySQL` is installed (it is in `requirements.txt`). Tables are created
automatically on startup.

---

## Dataset characteristics

PM-RAG ingests three public **NYC Capital Projects Dashboard** datasets
(data.gov). Datasets are classified by **column signature, not filename**
(`app/ingestion/schema_mapper.py`) — so a mislabeled or swapped filename still
lands in the correct type. Source IDs and raw cell values are always preserved;
the importer is idempotent (dedup via `source_record_hash`) and never overwrites
the CSVs.

Value conventions across all three files: money is `"$66,808,000"` (and `"($x)"`
for negatives), percentages are `"12.5%"`, reporting periods are `YYYYMM`
(e.g. `202305`), and dates are `MM/DD/YYYY`. Blank cells are common and treated
as missing.

### 1. Citywide Budget and Schedule → `citywide_budget_schedule`
- **Source:** [capital-projects-dashboard-citywide-budget-and-schedule](https://catalog.data.gov/dataset/capital-projects-dashboard-citywide-budget-and-schedule)
- **Grain:** one row per project **per reporting period** (a point-in-time snapshot).
- **Rows ingested:** ~56,525 → table `project_budget_schedule`.
- **Key columns:** `Reporting Period`, `Managing Agency`, `Sponsor Agency`,
  `PID`, `FMS ID`, `Total Budget`, `Spend to Date`, `Spend to Date (%)`,
  `FMS/Agency Project Name`, `Current Phase`, `Forecast Current Phase End`,
  `Forecast Completion`, actual design/construction milestone dates, `Borough`,
  `Budget Line`, `Ten Year Plan Category`, data dates.
- **Role:** the primary "current state" of each project (budget, phase, forecast
  completion) over time.

### 2. Citywide Schedule History and Variance → `schedule_history`
- **Source:** [capital-projects-dashboard-citywide-schedule-history-and-variance](https://catalog.data.gov/dataset/capital-projects-dashboard-citywide-schedule-history-and-variance)
- **Grain:** one row per project per reporting period per completion-date record.
- **Rows ingested:** ~22,464 → table `project_schedule_history`.
- **Key columns:** `Reporting Period`, `Managing Agency`, `PID`,
  `Agency Project Name`, `Current Phase`, `Completion Date`,
  `Completion Date Type` (Forecast/Actual), `Variance (day)`,
  `Reason for Forecast Completion Change`, `Data Date`.
- **Role:** schedule movement over time — the basis for detecting slippage and
  forecast deterioration.

### 3. Citywide Budget Spend History and Variance → `budget_spend_history`
- **Source:** [capital-projects-dashboard-citywide-budget-spend-history-and-variance](https://catalog.data.gov/dataset/capital-projects-dashboard-citywide-budget-spend-history-and-variance)
- **Grain:** one row per (`Managing Agency`, `FMS ID`) per month reported.
- **Rows ingested:** ~53,495 → table `project_budget_history`.
- **Key columns:** `Managing Agency`, `FMS ID`, `Year-Month Reported`,
  `Total Budget`, `Spend to Date`, `Spend to Date %`, `Budget Variance`,
  `Budget Variance %`.
- **Role:** budget/spend movement over time — the basis for detecting budget
  variance and unusual budget changes.

> **Identity note:** projects are resolved across the three files by `FMS ID`,
> `PID`, or project name (in that order). A 1:1 FMS↔project relationship is **not**
> assumed; mappings are recorded in `project_fms_mappings`.

---

## User manual — operating PM-RAG

PM-RAG is operated through the web dashboard at **http://127.0.0.1:8000** (every
page is also backed by a REST endpoint, so the same operations can be scripted).
The intended flow is **load data → analyze → detect problems → retrieve evidence
→ get a corrective-action recommendation → human review → benchmark**.

### The end-to-end workflow (what goes in, what comes out)

| # | Step | You provide (input) | PM-RAG returns (output) |
|---|------|---------------------|-------------------------|
| 1 | **Discover & classify** | Datasets folder (`DATASET_DIR`) | Detected files, auto-assigned `dataset_type`, classification confidence + warnings |
| 2 | **Ingest** | Confirm/override the type per file | Rows imported into the typed tables, normalized values, quality report, dedup stats |
| 3 | **Analyze** | (none — runs over ingested data) | Execution events + financial snapshots; schedule/budget variance derived |
| 4 | **Detect scenarios** | (none — rule thresholds) | `execution_scenarios` (problem_type + current_state), labeled `rule_generated` |
| 5 | **Index evidence** | (none) | Each source record → evidence chunk, embedded (dense) + BM25; index version recorded |
| 6 | **Retrieve** | A query or scenario, mode, `top_k`, as-of period | Ranked evidence with BM25/dense/fused/rerank scores + source IDs |
| 7 | **Recommend** | A scenario (or ad-hoc problem) | Problem/evidence summary, recommended action(s), citations, grounding, **or an abstention** |
| 8 | **Review** | accept / edit / reject / request-evidence | Human-review record (audit trail; nothing is auto-executed) |
| 9 | **Benchmark** | Baselines/ablations to run | Retrieval, grounding, abstention, efficiency metrics with bootstrap CIs |

### Detected execution problem types

Scenario detection (`app/project_analysis/scenario_detection.py`) is rule-based
and transparent; default thresholds:

| `problem_type` | Triggered when | Threshold |
|----------------|----------------|-----------|
| `schedule_delay` | Schedule variance (days) | ≥ 30 days |
| `forecast_deterioration` | Forecast completion date moves later | ≥ 30 days |
| `budget_variance` | Budget change | ≥ 10% |
| *(unusual budget)* | Large budget change flagged as unusual | ≥ 25% |
| `repeated_forecast_revision` | Multiple forecast revisions over time | recurring |
| `insufficient_evidence` | Signal present but records too sparse to classify | — |

Thresholds are constants in that module; all scenarios are labeled
`origin='rule_generated'` and are **not** expert-validated.

### Page-by-page guide

- **Dashboard (`/`)** — system health, record/scenario counts, model/backend
  status. *Output only.*
- **Datasets (`/datasets`)** — run **Discover** (input: none; output: file list +
  proposed types), **Inspect** a file (output: profile — encoding, delimiter,
  columns, inferred types, sample), then **Ingest** (input: confirmed type;
  output: import run with rows succeeded/failed/skipped). You can also upload a
  CSV. Afterwards run **Analysis** and **Rebuild index** here.
- **Projects (`/projects`, `/projects/{id}`)** — browse resolved projects;
  the detail page shows the budget/schedule timeline, schedule history, and
  budget history for one project. *Input:* a project selection. *Output:*
  per-project history + derived events.
- **Scenarios (`/scenarios`)** — list/filter detected execution problems; open
  one to see its state and evidence. You can also trigger **detect** or **build a
  benchmark split**. *Input:* filters / detect request. *Output:* scenarios.
- **PM-RAG Workbench (`/rag-workbench`)** — the retrieval sandbox. *Input:* a
  free-text query, retrieval mode (`current_project`, `same_project_historical`,
  `cross_project`, `hybrid_temporal`), `top_k`, optional as-of period and
  BM25/dense weights, which retrievers to enable. *Output:* ranked evidence with
  all component scores. **Note:** scores are retrieval scores, *not* calibrated
  factual-confidence probabilities.
- **Recommendations (`/recommendations`, `/recommendations/{id}`)** — generate
  and review corrective actions (see input/output below). Review decisions are
  recorded here (human-in-the-loop).
- **Experiments (`/experiments`)** — configure and run benchmark comparisons of
  baselines and ablations; poll the run to completion. *Output:* stored metrics.
- **Research Results (`/research-results`)** — charts for retrieval, grounding,
  actionability, efficiency, and abstention, built **only** from actually-executed
  runs. Export is available.
- **Evidence Audit (`/evidence-audit`)** — trace any recommendation's citations
  back to the exact source records (provenance + temporal validity).
- **Settings (`/settings`)** — shows the active configuration (backends, models,
  thresholds, paths). Credentials are never displayed or stored here.

### Generating a recommendation — input and output in detail

**Input** (Recommendations page, or `POST /api/recommendations/generate`):

| Field | Meaning |
|-------|---------|
| `scenario_id` *or* (`project_id` + `problem_type` + `problem_description`) | What problem to address |
| `as_of_period` | `YYYYMM` decision cutoff — evidence after this date is excluded (no temporal leakage) |
| `retrieval_mode`, `top_k` | How evidence is gathered |
| `grounding_enabled`, `citation_enabled`, `abstention_enabled` | Toggle the safety stages (also used for ablations) |
| `reranker_enabled`, `enabled_retrievers`, `force_rule_based` | Retrieval/generation controls |

**Output:**

- `problem_summary`, `evidence_summary` — grounded in cited records only
- `recommended_actions[]` — each with `action`, `category`, `priority`,
  `responsible_role`, `rationale`, `supporting_evidence_ids`, `expected_impact`,
  `assumptions`, and `requires_human_approval`
- `retrieved_evidence[]` — the evidence with scores and source IDs
- `validation.citations` — every cited ID checked for existence, in-retrieved-set,
  and temporal validity (`valid X / total Y`)
- `validation.grounding` — per-claim NLI/lexical support (`supported / total`)
- `abstained` + `abstention_reason` — **when evidence is insufficient or no
  factual claim is supported, PM-RAG returns no action and explains why** instead
  of guessing

### How to read the results (important)

- **Abstention is a correct answer.** An empty action list with an
  `abstention_reason` means the evidence did not support a defensible
  recommendation — not a bug. Stricter settings (real NLI, higher
  `ABSTENTION_MIN_SUPPORT_RATE`) abstain more often.
- **Citations are verifiable.** Every factual claim points to a real source
  record you can open in Evidence Audit.
- **Nothing is executed.** All consequential actions set
  `requires_human_approval=true`; PM-RAG supports a decision, it does not take it.
- **Scores ≠ confidence.** Retrieval/rerank scores rank evidence; they are not
  probabilities that a claim is true.
- **Observed vs. derived vs. expert.** Ingested rows are `observed`; events,
  scenarios, and system-generated gold evidence are `derived` / `rule_generated`
  and explicitly **not** expert-validated.

### Minimal script-driven run

```powershell
# load + analyze + detect + index (one shot)
python scripts/run_pipeline.py

# then query retrieval or generate a recommendation via the API, e.g.:
$body = @{ scenario_id = 1; retrieval_mode = "hybrid_temporal"; top_k = 6 } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/recommendations/generate -Method POST -Body $body -ContentType "application/json"
```

---

## REST API (selected)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/health`, `/api/system/status` | Liveness + models/paths/stats. |
| POST | `/api/datasets/discover`, `/inspect`, `/ingest` | Discover, profile, import CSVs. |
| POST | `/api/analysis/run`, `/api/index/rebuild` | Project analysis / evidence index build. |
| GET | `/api/projects`, `/api/projects/{id}/timeline` | Project explorer + history. |
| GET/POST | `/api/scenarios`, `/api/scenarios/detect` | Execution scenarios. |
| GET/POST | `/api/retrieval/modes`, `/api/retrieval/search` | Hybrid retrieval workbench. |
| POST | `/api/recommendations/generate` | Evidence-grounded corrective action. |
| POST | `/api/recommendations/{id}/review` | Record human review (HITL). |
| POST | `/api/experiments`, `/api/experiments/{id}/run` | Benchmark baselines/ablations. |
| GET | `/api/research-results/*` | Retrieval, grounding, abstention, efficiency metrics. |

Long-running work (ingest, analysis, indexing, experiments) runs as background
jobs — poll `/api/jobs/{job_id}`.

---

## Testing

Fully offline (SQLite + stub LLM; NLI/reranker disabled for determinism):

```powershell
python -m pytest -q
```

The suite covers normalization, content-based classification, security /
prompt-injection defence, claim extraction + grounding, the evidence-only
generator (grounding + abstention), credential-free config URL building, and an
integration pass that boots the app and renders every page and key endpoint.

---

## Research design notes

- **Temporal validity.** Retrieval can filter evidence to an as-of date
  (`reporting_period`) to prevent leakage of future records.
- **Scenario provenance.** Rule-detected scenarios are labeled
  `origin='rule_generated'`; system-derived gold evidence is explicitly marked
  *not expert-validated*. Observed vs. derived vs. expert content is distinguished.
- **Benchmarks.** Splits are project-disjoint; metrics (recall/precision/nDCG@k,
  MRR, attribution P/R/F1, supported/unsupported/contradiction + abstention
  rates, efficiency) are reported with percentile-bootstrap confidence intervals.
- **Reproducibility.** Each run records its configuration and environment.

---

## Safety & responsible-use

PM-RAG supports human decision-making; it does **not** execute corrective
actions. Recommendations are evidence-grounded estimates derived from reporting
snapshots and do not prove causation. Always have qualified project staff review
outputs before acting.
