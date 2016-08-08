
EXTRA_DIST = extraej.bash.skeleton

install-data-hook: extraej.bash.skeleton
	cp -v $(top_srcdir)/src/launcher/java/extraej.bash.skeleton $(top_srcdir)/src/launcher/java/extraej.bash
	$(top_srcdir)/substitute $(SED) "@sub_JAVA@" "${JAVA}" $(top_srcdir)/src/launcher/java/extraej.bash
	$(top_srcdir)/substitute $(SED) "@sub_AJC@" "${AJC}" $(top_srcdir)/src/launcher/java/extraej.bash
	$(top_srcdir)/substitute $(SED) "@sub_ASPECTWEAVER_JAR@" "${ASPECT_WEAVER_JAR}" $(top_srcdir)/src/launcher/java/extraej.bash
	$(top_srcdir)/substitute $(SED) "@sub_PREFIXDIR@" "${prefix}" $(top_srcdir)/src/launcher/java/extraej.bash
	cp -v $(top_srcdir)/src/launcher/java/extraej.bash $(prefix)/bin/extraej.bash
	cd $(prefix)/bin; $(LN_S) -f extraej.bash extraej; cd -

uninstall-local:
	rm $(prefix)/bin/extraej
	rm $(prefix)/bin/extraej.bash
