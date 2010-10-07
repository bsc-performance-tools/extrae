

EXTRA_DIST = API-section.tex \
 common-macros.tex \
 environmentvariables-section.tex \
 FAQ-section.tex \
 instrumented-routines-section.tex \
 merge-section.tex quick-guide-section.tex \
 user-guide.tex \
 wholeXML-section.tex \
 XML-section.tex \
 XML/buffer.xml.tex \
 XML/bursts.xml.tex \
 XML/callers.xml.tex \
 XML/cell.xml.tex \
 XML/config.xml.tex \
 XML/counters.xml.tex \
 XML/mpitrace.xml.tex \
 XML/mpi.xml.tex \
 XML/openmp.xml.tex \
 XML/pacx.xml.tex \
 XML/storage.xml.tex

manual_ug = user-guide

#
# Aquesta combinacio de params pot ser interessant per a generar un single HTML
# -split 0 -show_section_numbers -nonumbered_footnotes -mkdir -dir single

all: $(manual_ug).tex
if BUILD_DOCS_PS
	$(latex) $(manual_ug)
	$(latex) $(manual_ug)
	$(dvips) $(manual_ug).dvi -o $(manual_ug).ps
if BUILD_DOCS_PDF
	$(dvipdf) $(manual_ug)
endif
endif
if BUILD_HTML_DOCS
	$(latex2html) -split 0 -show_section_numbers -nonumbered_footnotes  $(manual_ug)
endif

install-exec-hook: all
	$(mkdir_p) $(datadir)/doc
if BUILD_DOCS_PS
	cp $(manual_ug).ps $(datadir)/doc
if BUILD_DOCS_PDF
	cp $(manual_ug).pdf $(datadir)/doc
endif
endif
if BUILD_HTML_DOCS
	$(mkdir_p) $(datadir)/doc/html
	cp user-guide/user-guide.html user-guide/user-guide.css user-guide/*.png $(datadir)/doc/html
	ln -f $(datadir)/doc/html/user-guide.html $(datadir)/doc/html/index.html
endif

clean:
	rm -fr $(manual_ug).aux $(manual_ug).dvi $(manual_ug).lof $(manual_ug).log $(manual_ug).lot $(manual_ug).ps $(manual_ug).pdf $(manual_ug).toc $(manual_ug)
