
scripts_FILES = UFlist.sh mpitrace_wizard
if HAVE_MRNET
scripts_FILES += mrnrun cpistack_stats.pl compare_clusters.pl animate.pl
endif

bin_SCRIPTS = $(scripts_FILES)

install-data-hook:
if HAVE_MRNET
	$(SED) "s|@sed_MPITRACE_HOME@|${prefix}|g" < mrnrun | $(SED) "s|@sed_SPECTRAL_HOME@|${SPECTRAL_HOME}|g" > $(bindir)/mrnrun
	chmod u+x $(bindir)/mrnrun
endif

