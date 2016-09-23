
EXTRA_DIST = extraej.bash.skeleton

noinst_SCRIPTS = extraej.bash

extraej.bash: extraej.bash.skeleton
	cp -v $^ $@
	$(top_srcdir)/substitute $(SED) "@sub_JAVA@" "${JAVA}" $@
	$(top_srcdir)/substitute $(SED) "@sub_AJC@" "${AJC}" $@
	$(top_srcdir)/substitute $(SED) "@sub_ASPECTWEAVER_JAR@" $@
	$(top_srcdir)/substitute $(SED) "@sub_PREFIXDIR@" "${prefix}" $@

install-data-hook: extraej.bash
	cp -v extraej.bash $(prefix)/bin/extraej.bash
	cd $(prefix)/bin; $(LN_S) -f extraej.bash extraej; cd -

uninstall-local:
	rm $(prefix)/bin/extraej
	rm $(prefix)/bin/extraej.bash
