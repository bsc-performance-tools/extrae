
EXTRA_DIST = \
 user-guide.ps \
 user-guide.pdf \
 user-guide

manual_ug = user-guide

#
# Aquesta combinacio de params pot ser interessant per a generar un single HTML
# -split 0 -show_section_numbers -nonumbered_footnotes -mkdir -dir single

all:
	@echo Nothing to be done by default. If you want to build the documentation
	@echo issue make build-documentation.

build-documentation: $(manual_ug).dvi
if BUILD_DOCS_PS
	$(dvips) $(manual_ug).dvi -o $(manual_ug).ps
if BUILD_DOCS_PDF
	$(dvipdf) $(manual_ug)
endif
endif
if BUILD_HTML_DOCS
	$(latex2html) -split 0 -show_section_numbers -nonumbered_footnotes $(manual_ug)
endif

version.tex:
	@echo "\newcommand{\TRACEVERSION}{$(PACKAGE_VERSION)}" > version.tex

$(manual_ug).dvi: $(manual_ug).tex version.tex
if BUILD_DOCS_PS
	$(latex) $(manual_ug)
	$(latex) $(manual_ug)
endif

install-exec-hook: $(manual_ug).ps $(manual_ug).pdf
	$(mkdir_p) $(DESTDIR)$(datadir)/doc
if BUILD_DOCS_PS
	cp $(manual_ug).ps $(DESTDIR)$(datadir)/doc
if BUILD_DOCS_PDF
	cp $(manual_ug).pdf $(DESTDIR)$(datadir)/doc
endif
endif
if BUILD_HTML_DOCS
	$(mkdir_p) $(DESTDIR)$(datadir)/doc/html
	cp user-guide/user-guide.html user-guide/user-guide.css user-guide/*.png $(DESTDIR)$(datadir)/doc/html
	ln -f $(DESTDIR)$(datadir)/doc/html/user-guide.html $(DESTDIR)$(datadir)/doc/html/index.html
endif

clean:
	@echo Nothing to be done by default. If you want to clean the documentation
	@echo issue make clean-documentation.

clean-documentation:
	rm -fr $(manual_ug).aux $(manual_ug).dvi $(manual_ug).lof $(manual_ug).log $(manual_ug).lot $(manual_ug).ps $(manual_ug).pdf $(manual_ug).toc $(manual_ug)
