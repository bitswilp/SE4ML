# Research Code vs Production Code — Objective 1.2

**AIMLCZG546 · Assignment II · Group 40**

Report-ready extract of section 6 of `notebooks/research_prototype.ipynb`.
The notebook is the research artifact; `src/loan_default/` is the production
artifact. This document is the comparison between them.

---

---

# 6. Research code vs production code

Everything above **works**. It loads real data, finds real problems, trains real models,
and produces a real artifact. So what is missing?

The gap is not correctness. It is that this notebook answers *"does this idea work?"*,
while the `src/loan_default/` package has to answer *"will this keep working, unattended,
on data nobody has looked at, when the person who wrote it has moved on?"*

## 6.1 Trait by trait

| # | Research code (this notebook) | Production code (`src/loan_default/`) | Why the change matters |
|---|---|---|---|
| 1 | Column types inferred by `select_dtypes` at runtime | `NUMERIC_FEATURES` / `CATEGORICAL_FEATURES` pinned in `data.py` | A serving payload of one row can infer types differently from a 148k-row file. Same code, different features, no error. |
| 2 | `print()` to stdout | `logging_config.get_logger()` with INFO / WARNING / ERROR | `print` vanishes under a process manager. Levels let you page on ERROR and ignore INFO. |
| 3 | Failures surface as raw tracebacks | `SchemaValidationError`, `DataQualityError`, `FileNotFoundError`, HTTP 503 | A typed error tells a caller *what* went wrong and whether retrying helps. |
| 4 | Constants edited in the source | `config/config.yaml` | Change a threshold without editing (or redeploying) code. |
| 5 | No tests | 88 pytest tests | Nothing here would notice if a refactor silently swapped two columns. |
| 6 | Leakage found by hand, once | `DataQualityChecker` — 8 automated checks | A one-off human check does not protect next month's data. |
| 7 | Runs top to bottom, in order, or not at all | Importable modules with single responsibilities | You cannot call cell 12 of a notebook from an API request handler. |
| 8 | Artifact written to the working directory | `CONFIG["model"]["artifact_path"]`, resolved absolutely | A relative path means the code only works from one directory. |
| 9 | Model reachable only by re-running the notebook | FastAPI `/predict` with a Pydantic schema | Other systems need an interface, not a copy of your environment. |
| 10 | No input validation — any DataFrame is accepted | Pydantic request model, 422 on malformed input | Production input is hostile or careless, not curated. |
| 11 | Fairness/leakage exclusions are comments | `test_invariance_to_excluded_columns` asserts them | A comment cannot fail the build. A test can. |
| 12 | State lives in the kernel; cells can run out of order | Functions with explicit arguments and return values | Out-of-order execution is a bug class that simply cannot exist in the package. |

## 6.2 The point that generalises

Traits 6 and 11 are the ones specific to *ML* systems rather than software generally.

Ordinary software fails loudly — a null pointer, a 500, a crash. ML systems fail
**silently**. Give this model a scrambled feature matrix and it returns a confident
probability. Feed it next quarter's data after an upstream schema change and it returns
confident probabilities. Nothing crashes. The output stays the same *shape*, and only the
*meaning* is wrong.

That is why the production version invests where a normal refactor would not bother: an
automated data-quality gate, invariance tests, directional tests, drift monitoring. They
exist to make silent failures loud.

## 6.3 What the production version does *not* improve

Being fair to the notebook:

- **The model is identical.** Same three candidates, same seed, same PR-AUC. Production
  engineering did not make it more accurate.
- **The exploration had to happen this way.** Sections 2.2 and 2.3 are what a notebook is
  *for*. Discovering the missingness leak took several pivots — running that as a test
  suite would have been slower and worse.
- **A notebook is better for communicating a finding.** The missingness table above is
  more persuasive read top-to-bottom than the same logic as `check_missingness_leakage()`.

The lesson is not "notebooks are bad". It is that the notebook is the **right tool for
the first question and the wrong tool for the second**, and that the transition between
them is real engineering work rather than a tidy-up.