MAIN = main
LATEXMK = latexmk
LATEXMKFLAGS = -pdf -interaction=nonstopmode -halt-on-error

.DEFAULT_GOAL := all

.PHONY: all main force watch clean distclean

# Always delegate to latexmk: it tracks every dependency (figures, .bib,
# .sty) itself and is a no-op when everything is current. The figures under
# figures/ are generated (../REPRODUCE.md); nothing here regenerates them.
all: main

main:
	$(LATEXMK) $(LATEXMKFLAGS) $(MAIN)

# Full rebuild that also clears the aux files a killed run can corrupt.
force:
	rm -f $(MAIN).aux $(MAIN).toc $(MAIN).out $(MAIN).fdb_latexmk $(MAIN).blg $(MAIN).bbl
	$(LATEXMK) $(LATEXMKFLAGS) $(MAIN)

watch:
	$(LATEXMK) $(LATEXMKFLAGS) -pvc $(MAIN)

clean:
	$(LATEXMK) -c $(MAIN)
	rm -f $(MAIN).bbl $(MAIN).run.xml $(MAIN).synctex.gz

distclean: clean
	$(LATEXMK) -C $(MAIN)
