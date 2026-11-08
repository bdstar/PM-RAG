# Design Summary

This document explains the key design choices behind the AquaTech RAG system: the
**embedding model**, the **LLM**, the **chunking strategy**, and the **anti-hallucination
guardrail**.

---

## 1. Embedding model — `sentence-transformers/all-MiniLM-L6-v2`

**Why this model:**

- **Purpose-built for retrieval.** It is a bi-encoder trained specifically to map sentences
  into a semantic space where cosine similarity reflects meaning — exactly what dense retrieval
  needs. A generative LLM is not an embedding model, so the embedding step uses a dedicated
  sentence-transformer, which is the correct, standard choice.
- **Lightweight & fast.** 384-dimensional vectors, ~22M parameters. It embeds the whole policy
  corpus in well under a second and runs comfortably on CPU, keeping the demo responsive.
- **Strong quality-for-size.** It remains one of the most popular, well-benchmarked models on the
  MTEB retrieval tasks for its size, which is ideal for a small, factual knowledge base.

**Implementation:** wired into LlamaIndex via `HuggingFaceEmbedding`. The ChromaDB collection is
created with cosine space (`hnsw:space = cosine`) so similarity scores are directly comparable
against the retrieval threshold.

**Upgrade path:** for larger/multilingual corpora, swap `EMBED_MODEL` for `BAAI/bge-small-en-v1.5`
or `bge-base-en-v1.5` — no other code changes are required.

---

## 2. LLM — `Qwen/Qwen2.5-0.5B-Instruct`

**Why this model:**

- **Small and open.** At ~0.5B parameters (~1 GB) it loads quickly, fits a modest 4 GB GPU, and
  even runs on CPU — so the system is runnable out-of-the-box with no license gating or access
  token required.
- **Instruction-tuned.** The guardrail relies on the model returning a small JSON object and
  honestly judging whether the answer is in the context. Instruction-tuned (`-Instruct`) models
  follow this contract far more reliably than base models, so the code applies the model's
  **chat template** automatically when one is available.
- **Good enough for grounded extraction.** For a small, factual knowledge base the task is mostly
  locating and restating facts from the retrieved context, which this model handles well — all
  four evaluation cases pass.

**Configurability.** The LLM is selected via `.env` (`LLM_MODEL`, `LLM_BACKEND`). Any larger
Hugging Face instruct model (e.g. `Qwen/Qwen2.5-1.5B-Instruct`) can be dropped in to improve
recall at the cost of memory/latency; `LOAD_IN_4BIT=true` helps on low-VRAM GPUs, and `HF_TOKEN`
enables gated models. A **`stub` backend** also provides an extractive, GPU-free fallback that can
only echo retrieved context — useful for CI and low-resource machines — and it activates
automatically if the chosen model fails to load.

**Generation settings:** low temperature (`0.1`) for deterministic, faithful answers;
`max_new_tokens = 320`; the prompt forbids outside knowledge and requires grounding.

---

## 3. Chunking strategy — sentence-aware, 512 tokens / 64 overlap

**Splitter:** LlamaIndex `SentenceSplitter` (`chunk_size=512`, `chunk_overlap=64`).

**Rationale:**

- **Sentence-aware splitting** avoids cutting mid-sentence, so each chunk is a self-contained,
  semantically coherent unit — which produces cleaner embeddings and more precise retrieval.
- **512 tokens** is large enough to keep a full policy statement (e.g. the entire remote-work or
  healthcare paragraph) together, but small enough that an embedding vector stays topically focused
  rather than averaging several unrelated facts.
- **64-token overlap (~12.5%)** preserves context across boundaries so a fact mentioned near the
  edge of one chunk is not lost — important for short, fact-dense policy documents where a single
  sentence can contain the entire answer.

For this small corpus most policy documents become one or a few chunks, which keeps retrieval crisp
and the cited sources easy to verify.

---

## 4. Anti-hallucination guardrail (Pydantic) — defense in depth

Grounding is enforced in **three independent layers**:

1. **Retrieval threshold.** If the top chunk's cosine similarity is below `SIMILARITY_THRESHOLD`
   (0.30), the system refuses *before* calling the LLM — there is simply nothing relevant to answer
   from.
2. **Pydantic-validated structured output.** The LLM must return JSON matching the `GroundedAnswer`
   schema (`status`, `answer`, `confidence`, `supporting_quotes`). The parser is hardened (strips
   code fences, repairs trailing commas, and falls back to regex field extraction) so that messy
   model output never crashes the pipeline; anything unparseable or marked `not_found` is converted
   into a safe **"I don't know. Information not provided"** refusal via `GroundedAnswer.enforce()`.
3. **Faithfulness check.** Even when the model claims an answer is grounded, the answer's content
   words — and crucially any **numbers** — must actually appear in the retrieved context, otherwise
   it is refused. This catches fabricated figures, the most common RAG failure mode.

**Evaluation result:** all four evaluation cases pass — three in-scope questions are answered
correctly with citations, and the out-of-scope "cryptocurrency bonuses" question is correctly
refused.

---

## 5. Notable detail: distractor document

The provided `knowledge/` folder includes an unrelated `Intelligence Summary.pdf`. It is retained
on purpose: it acts as a **distractor** that demonstrates the retriever surfaces only the relevant
policy chunks for policy questions, and that the guardrail does not let off-topic content leak into
answers.
