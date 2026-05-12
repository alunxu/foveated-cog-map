TEX = main
# Use xelatex: pdflatex segfaults on TeXLive 2026/Homebrew during 3rd pass
# (hyperref + figure-float \pdfendlink nesting issue).
ENGINE = xelatex

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex literature.bib
	$(ENGINE) -interaction=nonstopmode $(TEX).tex
	bibtex $(TEX)
	$(ENGINE) -interaction=nonstopmode $(TEX).tex
	$(ENGINE) -interaction=nonstopmode $(TEX).tex

quick:
	$(ENGINE) -interaction=nonstopmode $(TEX).tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out

distclean: clean
	rm -f $(TEX).pdf

.PHONY: all quick clean distclean
