# notebooks/

## `research_prototype.ipynb`

The **research code** half of Objective 1.2, shown against the modular
`src/loan_default/` package as the production half.

It is the exploratory workflow behind Assignment I's `loan_default_pipeline.py`:
load the data, investigate it, find the two target-leakage problems, train three
candidates, pick one on PR-AUC. Written the way exploratory work actually gets
written — top-to-bottom, `print()` for output, constants edited in place,
`select_dtypes` for column typing, no tests.

Section 6 is the comparison itself: twelve traits contrasted against their
production counterparts, plus an honest account of what the production version
does *not* improve.

**Provenance.** The modelling cells are taken from `loan_default_pipeline.py`
essentially verbatim (same constants, same functions, same seed). The
exploration cells in section 2 are a reconstruction — a finished script does not
preserve the investigation that produced its `LEAKAGE_COLS` list — and they
re-derive both findings from the actual dataset rather than quoting them.

### Running it

```bash
cd notebooks && jupyter lab research_prototype.ipynb   # or: jupyter notebook
```

It reads `../data/raw/loan_default_dataset.csv` and takes roughly two minutes,
most of it the Random Forest. Outputs are already saved in the file, so it can
be read without being run.