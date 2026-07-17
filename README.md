# AutoCare Agentic Assistant

A multi-agent AI system for vehicle maintenance support and safety triage. It implements four research ideas end to end — **Retrieval-Augmented Generation** (Lewis et al., 2020), **ReAct** tool-calling (Yao et al., 2022), a real trained **CNN image classifier** (LeCun et al., 1998 lineage), and **multi-agent orchestration** (Wu et al., 2023, AutoGen) — applied to a concrete real-world problem: LLMs confidently hallucinating maintenance facts, mileage calculations, or the urgency of a dashboard warning light, in a domain where a wrong answer is a safety issue, not just an inconvenience. See **`PAPERS.md`** for the full paper-to-code mapping and the argument for why this problem matters.

Built with **LangChain** (multi-agent orchestration, tool use, RAG), **PyTorch** (sentence embeddings for retrieval, and a genuinely trained CNN for warning-light image classification), and **FAISS**. Exposed as a **FastAPI** service, covered by unit/integration tests, and includes a retrieval-quality evaluation harness (`eval/`).

## The problem it solves

Ask a bare LLM "is my service due at 42,000 km?" or "what does a flashing engine light mean?" and it will answer fluently — and sometimes wrong, because nothing forces it to check a real source or do real arithmetic. In a safety-relevant domain like vehicle maintenance, a wrong answer ("that flashing light is probably fine") is a real-world harm, not just an inconvenience. This is a documented, general failure mode of LLMs (Ji et al., 2023, "Survey of Hallucination in Natural Language Generation"). The fix used here, and increasingly standard in industry, is to stop asking the model to *recall* facts and instead make it *retrieve* facts (RAG) and *delegate* calculations/lookups to deterministic tools (ReAct-style agent) — so every factual or numeric claim in the final answer is traceable to a document chunk or a function's return value, not to the model's parameters.

## Why this project (for the BMW Agentic AI Engineer role)

It mirrors a realistic industrial use case: a customer-facing assistant that must (1) answer questions grounded in internal documentation instead of hallucinating, (2) understand a photo as well as text, (3) perform deterministic calculations and turn them into a prioritized recommendation, and (4) do all of that through specialist agents coordinated by a supervisor — rather than one prompt trying to do everything. It deliberately demonstrates depth in two of the job posting's methodological areas (Generative AI/NLP via RAG + agents, and Computer Vision via the trained CNN), plus the platform/architecture thinking ("design and develop end-to-end, scalable AI systems") the role asks for.

## Architecture

```
User query (text, and/or an image path)
    │
    ▼
Supervisor Agent  ── routes to one or both specialists, merges their answers
    │
    ├── Diagnostic Agent            "what's wrong / what does this mean?"
    │     ├── search_maintenance_docs   → FAISS + PyTorch sentence-transformer RAG
    │     ├── classify_warning_light_image → trained PyTorch CNN (app/vision/)
    │     └── lookup_warning_light      → structured meaning/urgency table
    │
    └── Scheduling Agent             "what should I do, how urgent is it?"
          ├── check_service_interval    → deterministic mileage/date calculator
          └── assess_maintenance_risk   → combines both signals into a LOW…CRITICAL priority
    │
    ▼
FastAPI  POST /chat  (multi-agent, default)   |   POST /chat/simple  (original single flat agent, kept for comparison)
```

- `app/ingest.py` — loads markdown docs from `app/knowledge_base/`, chunks them, embeds them with a local PyTorch sentence-transformer model, and builds a FAISS index.
- `app/tools.py` — five LangChain `Tool` implementations (RAG retrieval, image classification, warning-light lookup, service-interval math, risk triage), grouped into `get_diagnostic_tools()` / `get_scheduling_tools()` for the multi-agent split.
- `app/vision/` — the CV component: `dataset.py` (synthetic labeled warning-light images), `model.py` (a small real CNN), `train.py` (a genuine training loop you run once), `classify.py` (inference).
- `app/agents/supervisor.py` — the multi-agent system: two specialist `AgentExecutor`s wrapped as tools for a top-level Supervisor agent.
- `app/agent.py` — the original single flat agent (all 5 tools, one prompt) — kept so you can directly compare the two architectures.
- `app/api.py` — FastAPI app exposing `POST /chat` (multi-agent) and `POST /chat/simple` (flat agent).
- `tests/` — unit tests per tool, RAG retrieval tests, CV dataset/model/inference tests, and agent-assembly tests for both architectures — all using a `FakeListLLM` where an LLM is needed, so no API key is required to run the suite.
- `eval/` — a small labeled QA set (`qa_eval.jsonl`) and `evaluate.py`, which measures retrieval hit-rate@k, the sanity check used in RAG-style papers before trusting the generator.
- `PAPERS.md` — line-by-line mapping from the four ideas this project implements (RAG, ReAct, CNN image classification, multi-agent orchestration) to the code, and the argument for why this problem is worth solving this way.

## How to run it — step by step

You need Python 3.10 or 3.11 and about 2 GB of free disk space (PyTorch + the embedding model account for most of that). An OpenAI API key is only needed for step 5 (talking to the agent) — everything before that runs fully offline.

**1. Get the code and enter the folder**

```bash
cd autocare-agent
```

**2. Create an isolated virtual environment** (keeps these dependencies from clashing with anything else on your machine)

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

If `torch` is slow to download on your connection, install the smaller CPU-only build explicitly first, then the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**4. Build the vector index** — this reads every `.md` file in `app/knowledge_base/`, splits it into overlapping chunks, embeds each chunk with a local PyTorch sentence-transformer model, and saves a FAISS index to `app/vector_store/`. It only needs to be re-run when you change the knowledge base.

```bash
python -m app.ingest
```

You should see `Vector store built at .../app/vector_store`. The first run also downloads the ~90 MB embedding model from Hugging Face (cached afterward, so this only happens once).

**5. Train the vision model** — this generates the synthetic warning-light image dataset in memory and trains the small CNN for ~15 epochs (a few seconds on CPU), saving weights to `app/vision/model.pt`.

```bash
python -m app.vision.train
```

You'll see per-epoch training loss and validation accuracy printed; it should reach well above 90% validation accuracy (the synthetic task is intentionally solvable). This step is what makes `classify_warning_light_image` work — skip it and that one tool will raise a clear error telling you to run this command.

**6. Add an LLM key** — the agents' *reasoning* (deciding which tool/specialist to call, and composing the final answer) is delegated to an LLM; retrieval and vision inference do not need one.

```bash
cp .env.example .env
# open .env and set OPENAI_API_KEY=sk-...
```

Don't want to use OpenAI? Swap `get_default_llm()` in `app/agent.py` for any other LangChain chat model (Anthropic, a local model via Ollama, etc.) — both `build_agent_executor()` and `build_supervisor_executor()` accept any `BaseLanguageModel`.

**7. Start the API**

```bash
uvicorn app.api:app --reload
```

This serves the agent at `http://localhost:8000`. Interactive API docs are auto-generated at `http://localhost:8000/docs`.

**8. Talk to it**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "My car has done 42000 km and the last service was at 30000 km, is a service due? Also, what does a flashing orange engine light mean, and how urgent is my situation overall?"}'
```

Behind the scenes: the Supervisor reads the question and decides it needs both specialists. It calls the **diagnostic_agent**, which uses `lookup_warning_light("engine")` to explain what a flashing amber engine light means. It calls the **scheduling_agent**, which uses `check_service_interval(42000, 30000)` (deterministic arithmetic — "OVERDUE by 3000 km," not a guess) and `assess_maintenance_risk(...)` to produce an overall priority (in this case, likely HIGH). The Supervisor then merges both specialists' answers into one reply. Try it with an image instead of naming the light: `{"message": "Here's a photo of my warning light at /path/to/photo.png, what is it and what should I do? Current mileage 42000, last service 30000."}` — this routes through `classify_warning_light_image` (the trained CNN) first.

Prefer the simpler, original single-agent version? `POST /chat/simple` uses the flat `app/agent.py` agent with all five tools directly (no sub-agent routing) — useful for comparing the two architectures side by side.

## Running the tests

```bash
pytest -v
```

- `tests/test_tools.py` — pure-Python unit tests for the calculator and warning-light tools, no model calls.
- `tests/test_risk.py` — unit tests for the risk-triage logic (all four priority levels).
- `tests/test_rag.py` — builds/loads the real FAISS index and checks retrieval returns the right document for a couple of questions.
- `tests/test_vision.py` — checks the synthetic dataset shapes and the CNN's forward pass; the end-to-end classification test skips gracefully if you haven't run `python -m app.vision.train` yet.
- `tests/test_agent.py` / `tests/test_supervisor.py` — verify both the flat agent and the multi-agent supervisor assemble correctly with all their tools, using `langchain_community.llms.fake.FakeListLLM` so no API key or network call is needed.

None of the tests require `OPENAI_API_KEY`.

## Running the research-style evaluation

```bash
python -m eval.evaluate
```

This runs the 10 labeled questions in `eval/qa_eval.jsonl` against the retriever and reports **hit-rate@k** — the fraction of questions for which the correct source document was actually retrieved (see `PAPERS.md` for why this specific metric, and what paper it's borrowed from). Use `--k 1` for a stricter top-1-only check.

## Extending

- Swap `sentence-transformers/all-MiniLM-L6-v2` for a larger PyTorch embedding model in `app/ingest.py`.
- Replace the synthetic dataset in `app/vision/dataset.py` with real labeled dashboard photos — no other CV code needs to change.
- Add a third specialist (e.g. a "parts/cost" agent) to the supervisor in `app/agents/supervisor.py` — the agents-as-tools pattern scales by just wrapping and registering another `AgentExecutor`.
- Add MLOps hooks (e.g. log every tool/sub-agent call + latency to a metrics backend, or track model versions with MLflow) around the `AgentExecutor.invoke()` calls.
- Containerize with the included `Dockerfile` and deploy behind an API gateway.

## Tech stack

Python, LangChain, PyTorch (sentence-transformer embeddings + a trained CNN), FAISS, FastAPI, pytest, Docker.
