# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         = a4
BUILDDIR      = builds

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source
# the i18n builder cannot share the environment and doctrees with the others
I18NSPHINXOPTS  = $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source

.NOTPARALLEL: all

clean:
	rm -rf $(BUILDDIR)/*

all: pdf html man

html:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

pdf:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo "Running LaTeX files through pdflatex..."
	$(MAKE) -C $(BUILDDIR)/latex all-pdf
	@echo "pdflatex finished; the PDF files are in $(BUILDDIR)/latex."

man:
	$(SPHINXBUILD) -b man $(ALLSPHINXOPTS) $(BUILDDIR)/man
	$(MKDIR_P) $(BUILDDIR)/man/man1
	$(MKDIR_P) $(BUILDDIR)/man/man3
	mv $(BUILDDIR)/man/extrae.1 $(BUILDDIR)/man/man1/
	mv $(BUILDDIR)/man/mpi2prv.1 $(BUILDDIR)/man/man1/
	mv $(BUILDDIR)/man/UFlist.sh.1 $(BUILDDIR)/man/man1/
	mv $(BUILDDIR)/man/extrae_event.3 $(BUILDDIR)/man/man3/
	cd $(BUILDDIR)/man/man1 && ln -sn mpi2prv.1 mpimpi2prv.1
	cd $(BUILDDIR)/man/man1 && ln -sn mpi2prv.1 mpi2dim.1
	cd $(BUILDDIR)/man/man1 && ln -sn mpi2prv.1 mpimpi2dim.1
	@echo
	@echo "Build finished. The manual pages are in $(BUILDDIR)/man."

install-exec-hook:
	$(MKDIR_P) $(datadir)/docs
	cp -r $(BUILDDIR)/html $(datadir)/docs/
	cp -r $(BUILDDIR)/man $(datadir)/
	cp $(BUILDDIR)/latex/user-guide.pdf $(datadir)/docs/

uninstall-local:
	rm -r $(datadir)/docs/html
	rm -r $(datadir)/man
	rm $(datadir)/docs/user-guide.pdf
	rmdir --ignore-fail-on-non-empty $(datadir)/docs
