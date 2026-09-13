# ICLR manuscript

The current manuscript is built from `main.tex` with the local ICLR 2027 style files.

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The editable and rendered figures are under `pics/`. The compact experiment bundle used by the paper is under `experiment_data/`.

