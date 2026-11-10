MAIN = main
LATEXMK = latexmk
LATEXMKFLAGS = -pdf -interaction=nonstopmode -halt-on-error

.DEFAULT_GOAL := all

.PHONY: all main force watch clean distclean arxiv

# The arXiv bundle (arxiv/main.tar.gz, preview arxiv/main.pdf): the camera-ready
# without the NeurIPS checklist, comments stripped (arxiv_latex_cleaner, via
# uvx), the .bbl shipped. Built in the TeX Live 2026 docker image: the system
# TeX Live 2023 lays the main text out one page longer than the submitted PDF.
# Upload arxiv/main.tar.gz; arXiv's own compile (TeX Live 2025) matches it.
ARXIV_DIR = arxiv
TEXLIVE_IMAGE = texlive/texlive:latest

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
	rm -rf $(ARXIV_DIR)
	$(LATEXMK) -C $(MAIN)

arxiv: main
	rm -rf $(ARXIV_DIR) && mkdir -p $(ARXIV_DIR)/src
	cp -r $(MAIN).tex $(MAIN).bbl *.sty appendix_*.tex figures $(ARXIV_DIR)/src/
	sed -i '/\\input{checklist\.tex}/d' $(ARXIV_DIR)/src/$(MAIN).tex
	cd $(ARXIV_DIR) && uvx arxiv_latex_cleaner src
	mv $(ARXIV_DIR)/src_arXiv $(ARXIV_DIR)/bundle && rm -rf $(ARXIV_DIR)/src
	cd $(ARXIV_DIR)/bundle && tar czf ../$(MAIN).tar.gz *
	mkdir -p $(ARXIV_DIR)/build && tar xzf $(ARXIV_DIR)/$(MAIN).tar.gz -C $(ARXIV_DIR)/build
	docker run --rm -u $$(id -u):$$(id -g) -v $(CURDIR)/$(ARXIV_DIR)/build:/w -w /w $(TEXLIVE_IMAGE) \
		sh -c 'for i in 1 2 3; do pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex >/dev/null || exit 1; done; ! grep -E "LaTeX Warning|Token not allowed" $(MAIN).log'
	cp $(ARXIV_DIR)/build/$(MAIN).pdf $(ARXIV_DIR)/$(MAIN).pdf
	@pdfinfo $(ARXIV_DIR)/$(MAIN).pdf | grep Pages
