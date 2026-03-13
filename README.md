# Infra — Learning Loop Tests

Two tests ship in this container:

| Test | What it does |
|---|---|
| **Test 1** | Runs `learning_loop.py` end-to-end — generates mock data, runs the query expansion loop, and prints CTR before/after |
| **Test 2** | Runs the `pytest` suite — 19 unit/integration tests covering data generation, CTR calculation, loop correctness, and the overfitting sanity check |

---

## Requirements

- [Docker](https://docs.docker.com/get-docker/) installed and running

---

## Build

From the repo root:

```bash
docker build -t oa-infra-tests ./infra
```

Expected output (last few lines):

```
#8 Successfully installed iniconfig-2.3.0 pytest-8.0.0 ...
#9 [5/5] COPY code/ ./code/
#10 exporting to image ... done
```

---

## Run Both Tests

```bash
docker run --rm oa-infra-tests
```

Expected output:

```
=== Test 1: Learning Loop End-to-End ===
========================================================================
  Query Expansion Learning Loop — Demonstration
========================================================================

[1] Generating mock dataset...
    Posts:              57
    Alerts:             57
    Feedbacks:          57
    Positive feedback:  39 / 57  (68%)

[2] Initializing baseline query terms from brand guidelines...
    zephyr_energy: 4 core terms — "energy drink", "focus", "productivity", "morning routine"
    ...

[3] CTR BEFORE learning loop (unfiltered — all alerts surfaced):
    Overall:  68.4%

[4] Running Query Expansion Learning Loop...
    zephyr_energy: +5 new terms  |  top: "anyone else" (0.95), ...

[5] CTR AFTER learning loop (filtering by learned term scores):
    Overall:              88.9%
    CTR improvement:  68.4%  →  88.9%
    ✓ Overfitting confirmed: learning loop improves CTR on training data.

=== Test 2: Pytest Suite ===
collected 19 items
...
19 passed in 0.05s
```

---

## Run Tests Individually

**Test 1 only — end-to-end script:**

```bash
docker run --rm oa-infra-tests python3 code/src/learning_loop.py
```

**Test 2 only — pytest suite:**

```bash
docker run --rm oa-infra-tests python3 -m pytest code/tests/ -v
```

**Single pytest test by name:**

```bash
docker run --rm oa-infra-tests python3 -m pytest code/tests/ -v -k "test_ctr_improves_on_training_data"
```

---

## Project Structure

```
infra/
├── Dockerfile                  # builds the test image
└── code/
    ├── requirements.txt        # full production deps (sentence-transformers, torch, etc.)
    ├── requirements-test.txt   # test-only deps (pytest only — no torch/CUDA)
    ├── src/
    │   ├── learning_loop.py    # Query Expansion Learning Loop (Test 1)
    │   ├── mock_api.py         # mock Social Listening API data generator
    │   └── embedder.py         # post embedder (sentence-transformers, not used in tests)
    └── tests/
        └── test_learning_loop.py   # 19 pytest tests (Test 2)
```

> **Note:** The Docker image installs only `requirements-test.txt` (pytest, ~5MB).
> `requirements.txt` is for the full production stack and pulls PyTorch + CUDA (~3GB).
> The two tests only use Python stdlib + pytest.
