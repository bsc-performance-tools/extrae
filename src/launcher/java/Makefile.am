
bin_SCRIPTS = extraej.bash
CLEANFILES = extraej.bash
EXTRA_DIST = extraej.bash.skeleton

extraej.bash: extraej.bash.skeleton
	cp $(top_srcdir)/src/launcher/java/extraej.bash.skeleton $(top_srcdir)/src/launcher/java/extraej.bash
	$(top_srcdir)/substitute $(SED) "@sub_JAVA@" "${JAVA}" $(top_srcdir)/src/launcher/java/extraej.bash
	$(top_srcdir)/substitute $(SED) "@sub_AJC@" "${AJC}" $(top_srcdir)/src/launcher/java/extraej.bash
	$(top_srcdir)/substitute $(SED) "@sub_ASPECTWEAVER_JAR@" "${ASPECT_WEAVER_JAR}" $(top_srcdir)/src/launcher/java/extraej.bash
	$(top_srcdir)/substitute $(SED) "@sub_PREFIXDIR@" "${prefix}" $(top_srcdir)/src/launcher/java/extraej.bash


install-data-hook: extraej.bash
	cp -v $(top_srcdir)/src/launcher/java/extraej.bash $(prefix)/bin/extraej.bash
	cd $(prefix)/bin; $(LN_S) -f extraej.bash extraej; cd -
