
EXTRA_DIST = extraej.bash

install-data-hook:
	cp -v extraej.bash $(prefix)/bin
	$(top_srcdir)/substitute $(SED) "@sub_JAVA@" "${JAVA}" $(prefix)/bin/extraej.bash
	$(top_srcdir)/substitute $(SED) "@sub_AJC@" "${AJC}" $(prefix)/bin/extraej.bash
	$(top_srcdir)/substitute $(SED) "@sub_ASPECTWEAVER_JAR@" "${ASPECT_WEAVER_JAR}" $(prefix)/bin/extraej.bash
	$(top_srcdir)/substitute $(SED) "@sub_PREFIXDIR@" "${prefix}" $(prefix)/bin/extraej.bash
	cd $(prefix)/bin; $(LN_S) -f extraej.bash extraej; cd -
