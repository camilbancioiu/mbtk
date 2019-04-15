# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    = -c ./doc
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = MBFF
SOURCEDIR     = doc
BUILDDIR      = doc/_build

.PHONY: help Makefile doc doc-clean doc-rebuild test

doc: Makefile
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	notify-send "Make" "Sphinx documentation built."

doc-clean: Makefile
	rm -rf "$(BUILDDIR)/doctrees"
	rm -rf "$(BUILDDIR)/html"

doc-rebuild: doc-clean doc
	notify-send "Make" "Sphinx documentation rebuilt."

test: Makefile
	python3 test.py
	notify-send "Make" "Testing complete."
