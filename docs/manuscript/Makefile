TEX = main

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex literature.bib
	pdflatex -interaction=nonstopmode $(TEX).tex
	bibtex $(TEX)
	pdflatex -interaction=nonstopmode $(TEX).tex
	pdflatex -interaction=nonstopmode $(TEX).tex

quick:
	pdflatex -interaction=nonstopmode $(TEX).tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out

distclean: clean
	rm -f $(TEX).pdf

.PHONY: all quick clean distclean
