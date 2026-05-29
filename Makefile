fast: 
	pdflatex main.tex

all: 
	pdflatex main.tex
	bibtex main.aux
	pdflatex main.tex
	pdflatex main.tex

work:
	evince main.pdf 2>/dev/null &
	vim main.tex

homeworks:
	$(MAKE) -C homeworks

tutorials:
	$(MAKE) -C tutorials

slides:
	$(MAKE) -C slides

.PHONY: fast all work homeworks tutorials slides
