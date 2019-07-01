# You can set these variables from the command line.
SPHINXOPTS    = -c ./doc
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = MBFF
SOURCEDIR     = doc
BUILDDIR      = doc/_build

.PHONY: help Makefile doc doc-clean doc-rebuild test

# Enable Make to receive extra CLI arguments, instead of interpreting them as
# targets to build. See https://stackoverflow.com/a/47008498/583574
#
# Catch-all target that prevents failing when specifying multiple arguments to
# Make, because Make normally thinks all its arguments are real targets and will
# normally complain when one argument isn't an actual target.
%:
	@:

# Macro that retrieves the targets passed to Make, excluding the current target. 
args = `arg="$(filter-out $@,$(MAKECMDGOALS))" && echo $${arg:-${1}}`

doc: Makefile
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	notify-send "Make" "Sphinx documentation built."

doc-clean: Makefile
	rm -rf "$(BUILDDIR)/doctrees"
	rm -rf "$(BUILDDIR)/html"

doc-rebuild: doc-clean doc
	@notify-send "Make" "Sphinx documentation rebuilt."

test: Makefile
	@python3 test.py $(call args,all)
	notify-send "Make" "Testing complete."

test-clean:
	rm -rf testfiles/tmp/*
