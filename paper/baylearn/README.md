# BayLearn 2026 extended abstract

Anonymized 2-page abstract (NeurIPS 2023 format) summarizing the pre-registered
LCT-layer evaluation in `paper/report.md`.

- **Deadline:** Thu July 30, 2026, 11:59pm PDT
- **Submit via CMT:** https://cmt3.research.microsoft.com/BAYLEARN2026
- **Rules:** 2 pages max in NeurIPS 2023 style (`neurips_2023.sty`, vendored
  here), one extra page for references/acknowledgements only, fully anonymized.
- **Tracking:** Linear ALOK-808.

## Build

```bash
cd paper/baylearn
tectonic baylearn2026.tex
```

(or `latexmk -pdf baylearn2026.tex` with TeXLive).

`std_study.png` is copied from `paper/figures/` — regenerate upstream with
`MPLBACKEND=Agg uv run --with matplotlib python scripts/plot_mps_study.py`
(which writes both `mps_study.png` and `std_study.png`) if the study data
changes, then re-copy.

The PDF must stay anonymous: no author names, no repository URL.
