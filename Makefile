# You can set these variables from the command line.
SPHINXOPTS    = -c ./doc
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = MBFF
SOURCEDIR     = doc
BUILDDIR      = doc/_build

PYTEST_WORKERS=--workers auto

.PHONY: help Makefile doc doc-clean doc-rebuild test

doc: Makefile
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	notify-send "Make" "Sphinx documentation built."

doc-clean: Makefile
	rm -rf "$(BUILDDIR)/doctrees"
	rm -rf "$(BUILDDIR)/html"

doc-rebuild: doc-clean doc
	@notify-send "Make" "Sphinx documentation rebuilt."

test-all: Makefile
	pytest $(PYTEST_WORKERS) --capture=tee-sys
	notify-send "Make" "Testing complete."

test: Makefile
	pytest $(PYTEST_WORKERS) --capture=tee-sys -m "not slow"
	notify-send "Make" "Testing complete."

test-clean:
	rm -rf tests/testfiles/tmp/*

clean:
	find -name __pycache__ | xargs rm -rf
