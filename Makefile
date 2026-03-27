all: 
	pdflatex main.tex
	bibtex main.aux
	pdflatex main.tex
	pdflatex main.tex

work:
	evince main.pdf 2>/dev/null &
	vim main.tex

