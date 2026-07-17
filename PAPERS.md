# Research Foundation

This project is a working implementation of four ideas from research, combined to address one problem: **LLMs give confident, fluent answers even when they are wrong, and this is unacceptable in a safety-relevant domain like vehicle maintenance advice.** RAG and ReAct solve the "don't hallucinate facts or math" half of that problem (see below). Two further additions — a real trained computer-vision model, and a multi-agent supervisor architecture — extend the system from "answers text questions accurately" to "handles a photo, triages urgency, and scales past a single agent juggling too many responsibilities," which is what makes it closer to something deployable rather than a demo.

## 1. Retrieval-Augmented Generation (Lewis et al., 2020)

**Paper:** "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis, Perez, Piktus, et al., NeurIPS 2020). https://arxiv.org/abs/2005.11401

**Core idea:** instead of relying purely on parameters memorized during training, the model retrieves relevant passages from an external, inspectable knowledge source at query time and conditions its generation on them. The paper shows this reduces hallucination and lets you update the system's knowledge by editing the document store, not retraining the model.

**Where it lives in this codebase:**

| Paper concept | Code |
|---|---|
| Document index built from a knowledge corpus | `app/knowledge_base/*.md` — source documents |
| Dense retriever (paper uses DPR; we use a modern equivalent) | `app/ingest.py` — `sentence-transformers/all-MiniLM-L6-v2` embeddings, a distilled BERT-family encoder, indexed with FAISS (Facebook AI Similarity Search) instead of the paper's MIPS index |
| Retrieve top-k passages, condition generation on them | `app/tools.py::_search_knowledge_base` — retrieves top-3 chunks; `app/agent.py` — system prompt instructs the LLM to ground factual claims in retrieved chunks |
| Non-parametric memory that can be updated without retraining | Add/edit a `.md` file in `knowledge_base/` and re-run `python -m app.ingest` — no fine-tuning required |

**Deviation from the paper, and why:** the original RAG paper fine-tunes the generator jointly with the retriever end-to-end. This project uses RAG in the now-standard "retrieve-then-prompt" inference-time form (as popularized by LangChain and used industrially), because it needs zero training data and zero GPU budget while keeping the same core mechanism: ground generation in retrieved, auditable text rather than parametric memory.

## 2. ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)

**Paper:** "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao, Zhao, Yu, et al., ICLR 2023). https://arxiv.org/abs/2210.03629

**Core idea:** instead of a single-shot answer, or reasoning without grounding, interleave *thoughts* ("I need to check the mileage") with *actions* (call a tool) and *observations* (the tool's result), letting the model decide dynamically which tool to invoke and when it has enough information to answer. This is what makes a system "agentic" rather than a plain chatbot.

**Where it lives in this codebase:**

| Paper concept | Code |
|---|---|
| A set of discrete, callable actions | `app/tools.py` — three `StructuredTool`s: `search_maintenance_docs`, `check_service_interval`, `lookup_warning_light` |
| Thought → Action → Observation loop | `app/agent.py::build_agent_executor` — `create_tool_calling_agent` + `AgentExecutor` implement this loop (modern LangChain uses the LLM's native function/tool-calling instead of ReAct's original text-parsing format, but the reasoning-act-observe cycle is the same) |
| Model decides *which* action fits the sub-question, possibly chaining several | System prompt in `app/agent.py` explicitly instructs: "If a question needs more than one tool, call them in sequence and combine the results." |

## 3. Computer vision: a real trained CNN (LeCun et al., 1998 lineage)

**Paper:** "Gradient-Based Learning Applied to Document Recognition" (LeCun, Bottou, Bengio, Haffner, 1998) — the original LeNet paper, still the conceptual ancestor of every conv-pool-conv-pool-FC image classifier, including the tiny one used here. https://ieeexplore.ieee.org/document/726791

**Why it's here:** the job posting explicitly asks for depth in at least two of {Generative AI, Computer Vision, NLP, MLOps}. A RAG+agent system alone is GenAI/NLP only. Real vehicle-maintenance interactions often involve a photo ("what's this light on my dash?") rather than the driver correctly naming the icon, so a CV path is also a genuine product improvement, not just a resume checkbox.

**What's actually implemented (and what isn't):** `app/vision/dataset.py` procedurally generates labeled 32x32 icon images (5 classes matching the warning-light table) with randomized jitter and pixel noise, so the classification task is non-trivial and requires generalization, not memorization. `app/vision/model.py` defines a small real `nn.Module` CNN. `app/vision/train.py` runs an actual training loop (forward pass, cross-entropy loss, backprop, Adam optimizer, train/validation split, held-out accuracy reporting) and saves real learned weights to `app/vision/model.pt`. This is a genuine — if intentionally small and synthetic — training pipeline, not a pretrained model wrapper. The honest limitation: it's trained on procedurally generated icons, not real dashboard photos; swapping in a labeled real-photo dataset would need no code changes beyond `dataset.py`.

## 4. Multi-agent orchestration (Wu et al., 2023 — AutoGen)

**Paper:** "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (Wu, Bansal, Zhang, et al., Microsoft Research, 2023). https://arxiv.org/abs/2308.08155

**Core idea:** rather than one agent with an ever-growing list of tools and responsibilities, decompose the system into specialist agents with narrow scopes, coordinated by a supervisor/orchestrator that routes requests (and can combine multiple specialists' outputs into one answer).

**Where it lives in this codebase:** `app/agents/supervisor.py`. `get_diagnostic_tools()` and `get_scheduling_tools()` (in `app/tools.py`) partition the five tools into two responsibilities: *diagnosis* (what's wrong — RAG, image classification, warning-light lookup) and *scheduling/triage* (what to do about it — service-interval math, risk scoring). Each becomes its own `AgentExecutor`, then each executor is wrapped as a `StructuredTool` (the "agents-as-tools" pattern) so a top-level Supervisor agent can call either — or both, for compound questions like "here's a photo of my warning light and I'm at 42,000 km, what should I do?" — and merge their answers.

**Why this over the flat version:** the original single-agent design (`app/agent.py`, kept for comparison and reachable at `/chat/simple`) works fine at 3-5 tools, but it doesn't scale: a real production system would have dozens of tools across many domains, and one prompt trying to reason about all of them at once gets both slower and less reliable. Splitting by responsibility is exactly the architectural move BMW's own job posting asks for under "design and develop end-to-end, scalable AI systems."

## The real-world problem this solves

Vehicle owners and service advisors ask questions that mix three different needs: **factual lookup** ("how often should brake fluid be flushed?"), **deterministic calculation** ("is a service due at 42,000 km?"), and **structured diagnosis** ("what does a flashing amber engine light mean?"). A single LLM call, unconstrained, will answer all three fluently — and will occasionally fabricate a mileage interval or downplay a genuinely urgent red warning light, because nothing forces it to check a ground-truth source.

This is a real, well-documented failure mode in production LLM systems (see e.g. Ji et al., "Survey of Hallucination in Natural Language Generation", ACM Computing Surveys 2023, https://arxiv.org/abs/2202.03629) and it is exactly why the industry moved from "just prompt a chatbot" to retrieval-grounded, tool-using agents for any domain where a wrong answer has real consequences — automotive safety being a direct example. RAG addresses the factual-grounding half of the problem; ReAct-style tool use addresses the calculation/structured-lookup half by forcing the model to delegate arithmetic and lookups to deterministic code instead of doing them "in its head." Combining both is what lets this system say, correctly, "your service is 5,000 km overdue" (a calculator did the math) while also being able to explain *why* that interval matters (retrieved from source documentation) — and to flag a red warning light as urgent rather than glossing over it.

## Evaluation

`eval/evaluate.py` runs a small retrieval-quality benchmark inspired by the RAG paper's own evaluation methodology: for a labeled set of questions (`eval/qa_eval.jsonl`), each with a known relevant document, it measures **retrieval hit-rate@k** — whether the correct source document appears in the top-k chunks returned by the retriever. This is the standard sanity check that the retrieval component (the "R" in RAG) is doing its job before ever involving the LLM. See `eval/evaluate.py` and the "Evaluation" section in `README.md` for how to run it.
