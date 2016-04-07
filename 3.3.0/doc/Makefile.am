manual_ug = user-guide

TEX_FILES = \
 $(manual_ug).tex \
 API-section.tex \
 common-macros.tex \
 configure-installation.tex \
 environmentvariables-section.tex \
 examples-section.tex \
 FAQ-section.tex \
 instrumented-routines-section.tex \
 merge-section.tex \
 online.tex \
 overhead.tex \
 quick-guide-section.tex \
 pnmpi-section.tex \
 regression-tests.tex \
 submit-bug.tex \
 user-guide.tex \
 wholeXML-section.tex \
 XML-section.tex \
 XML


EXTRA_DIST = \
 user-guide.ps \
 user-guide.pdf \
 user-guide \
 $(TEX_FILES)

#
# Aquesta combinacio de params pot ser interessant per a generar un single HTML
# -split 0 -show_section_numbers -nonumbered_footnotes -mkdir -dir single

all:
	@echo Nothing to be done by default. If you want to build the documentation issue make build-documentation.

build-documentation: $(TEX_FILES)
	@echo "\newcommand{\TRACEVERSION}{$(PACKAGE_VERSION)}" > version.tex
	./copy_tex_files_from_to.sh $(top_srcdir)/doc .
if BUILD_DOCS_PS
	$(latex) $(manual_ug)
	$(latex) $(manual_ug)
	$(dvips) $(manual_ug).dvi -o $(manual_ug).ps
if BUILD_DOCS_PDF
	$(dvipdf) $(manual_ug)
endif
endif
if BUILD_HTML_DOCS
	$(latex2html) -split 0 -show_section_numbers -nonumbered_footnotes $(manual_ug)
	$(SED) s/HREF=\"user-guide.html/HREF=\"/ < user-guide/index.html > user-guide/index.2.html
	$(SED) s/SRC=\"img/SRC=\"http:\\/\\/www.bsc.es\\/sites\\/default\\/files\\/public\\/computer_science\\/performance_tools\\/img/ < user-guide/index.2.html > user-guide/index.bsc.html
	rm -fr user-guide/index.2.html
	cd ..
endif

install-exec-hook: $(manual_ug).ps $(manual_ug).pdf
	$(MKDIR_P) $(DESTDIR)$(datadir)/doc
	cp $(top_srcdir)/doc/$(manual_ug).ps $(DESTDIR)$(datadir)/doc
	cp $(top_srcdir)/doc/$(manual_ug).pdf $(DESTDIR)$(datadir)/doc
	$(MKDIR_P) $(DESTDIR)$(datadir)/doc/html
	cp $(top_srcdir)/doc/user-guide/user-guide.html $(top_srcdir)/doc/user-guide/user-guide.css $(top_srcdir)/doc/user-guide/*.png $(DESTDIR)$(datadir)/doc/html
	ln -f $(DESTDIR)$(datadir)/doc/html/user-guide.html $(DESTDIR)$(datadir)/doc/html/index.html

clean:
	@echo Nothing to be done by default. If you want to clean the documentation issue make clean-documentation.

clean-documentation:
	rm -fr $(manual_ug).aux $(manual_ug).dvi $(manual_ug).lof $(manual_ug).log $(manual_ug).lot $(manual_ug).ps $(manual_ug).pdf $(manual_ug).toc $(manual_ug)
