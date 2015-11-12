
bin_PROGRAMS = extraej.bash

extraej.bash$(EXEEXT):
	cp $(top_srcdir)/src/launcher/java/extraej.bash.skeleton $(top_srcdir)/src/launcher/java/extraej.bash$(EXEEXT)
	$(top_srcdir)/substitute $(SED) "@sub_JAVA@" "${JAVA}" $(top_srcdir)/src/launcher/java/extraej.bash
	$(top_srcdir)/substitute $(SED) "@sub_AJC@" "${AJC}" $(top_srcdir)/src/launcher/java/extraej.bash
	$(top_srcdir)/substitute $(SED) "@sub_ASPECTWEAVER_JAR@" "${ASPECT_WEAVER_JAR}" $(top_srcdir)/src/launcher/java/extraej.bash
	$(top_srcdir)/substitute $(SED) "@sub_PREFIXDIR@" "${prefix}" $(top_srcdir)/src/launcher/java/extraej.bash


install-data-hook: extraej.bash$(EXEEXT)
	cp -v $(top_srcdir)/src/launcher/java/extraej.bash$(EXEEXT) $(prefix)/bin/extraej.bash$(EXEEXT)
	cd $(prefix)/bin; $(LN_S) -f extraej.bash$(EXEEXT) extraej; cd -
