PROJECT_NAME := $(shell basename $(CURDIR))

all: compile clean

compile:
	pdflatex -interaction=nonstopmode -halt-on-error -jobname=$(PROJECT_NAME) main.tex
	pdflatex -interaction=nonstopmode -halt-on-error -jobname=$(PROJECT_NAME) main.tex

clean:
	rm -f *.aux *.log *.out *.toc *.nav *.snm *.vrb
