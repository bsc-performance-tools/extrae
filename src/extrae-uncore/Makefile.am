SUBDIRS = uncore-service

dist_bin_SCRIPTS = extrae-uncore.py

install-data-hook:
	$(top_srcdir)/substitute $(SED) "@sub_PREFIXDIR@" "${prefix}"    $(DESTDIR)${prefix}/bin/extrae-uncore.py
	$(top_srcdir)/substitute $(SED) "@sub_LIBDIR@"    "${libdir}"    $(DESTDIR)${prefix}/bin/extrae-uncore.py
	$(top_srcdir)/substitute $(SED) "@sub_PAPI_HOME@" "${PAPI_HOME}" $(DESTDIR)${prefix}/bin/extrae-uncore.py
	chmod u+x $(DESTDIR)${prefix}/bin/extrae-uncore.py
	ln -fs ${prefix}/bin/extrae-uncore.py $(DESTDIR)${prefix}/bin/extrae-uncore
	$(MKDIR_P) $(DESTDIR)$(sysconfdir)
	cp extrae_uncore_template.xml $(DESTDIR)$(sysconfdir)/extrae_uncore_template.xml
	$(top_srcdir)/substitute $(SED) "@sub_PREFIXDIR@" "${prefix}" $(DESTDIR)${prefix}/etc/extrae_uncore_template.xml

