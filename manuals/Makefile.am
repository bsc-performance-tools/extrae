
man_MANS = UFlist.1 mpi2prv.1 extrae.1 extrae_event.3 

EXTRA_DIST = UFlist.1 mpi2prv.1 extrae.1 extrae_event.3 

man_dir=$(top_srcdir)/manuals

install-exec-hook:
	$(MKDIR_P) $(DESTDIR)$(mandir)/man1
	cp -f $(man_dir)/UFlist.1 $(man_dir)/mpi2prv.1 $(man_dir)/extrae.1 $(DESTDIR)$(mandir)/man1
	cd  $(DESTDIR)$(mandir)/man1 ; $(LN_S) -f mpi2prv.1 mpi2dim.1
	cd  $(DESTDIR)$(mandir)/man1 ; $(LN_S) -f mpi2prv.1 mpimpi2prv.1
	cd  $(DESTDIR)$(mandir)/man1 ; $(LN_S) -f mpi2prv.1 mpimpi2dim.1
	$(MKDIR_P) $(DESTDIR)$(mandir)/man3
	cp -f $(man_dir)/extrae_event.3 $(DESTDIR)$(mandir)/man3
	cd  $(DESTDIR)$(mandir)/man3 ; $(LN_S) -f extrae_event.3 extrae_counters.3
	cd  $(DESTDIR)$(mandir)/man3 ; $(LN_S) -f extrae_event.3 extrae_eventandcounters.3
	cd  $(DESTDIR)$(mandir)/man3 ; $(LN_S) -f extrae_event.3 extrae_restart.3
	cd  $(DESTDIR)$(mandir)/man3 ; $(LN_S) -f extrae_event.3 extrae_shutdown.3
	cd  $(DESTDIR)$(mandir)/man3 ; $(LN_S) -f extrae_event.3 extrae_set_tracing_tasks.3
	cd  $(DESTDIR)$(mandir)/man3 ; $(LN_S) -f extrae_event.3 extrae_next_hwc_set.3
	cd  $(DESTDIR)$(mandir)/man3 ; $(LN_S) -f extrae_event.3 extrae_previous_hwc_set.3
# These are maintained for backwards compatibility
	cd  $(DESTDIR)$(mandir)/man3 ; $(LN_S) -f extrae_event.3 mpitrace_event.3
	cd  $(DESTDIR)$(mandir)/man3 ; $(LN_S) -f extrae_event.3 mpitrace_counters.3
	cd  $(DESTDIR)$(mandir)/man3 ; $(LN_S) -f extrae_event.3 mpitrace_eventandcounters.3
	cd  $(DESTDIR)$(mandir)/man3 ; $(LN_S) -f extrae_event.3 mpitrace_restart.3
	cd  $(DESTDIR)$(mandir)/man3 ; $(LN_S) -f extrae_event.3 mpitrace_shutdown.3
	cd  $(DESTDIR)$(mandir)/man3 ; $(LN_S) -f extrae_event.3 mpitrace_set_tracing_tasks.3
	cd  $(DESTDIR)$(mandir)/man3 ; $(LN_S) -f extrae_event.3 mpitrace_next_hwc_set.3
	cd  $(DESTDIR)$(mandir)/man3 ; $(LN_S) -f extrae_event.3 mpitrace_previous_hwc_set.3

uninstall-local:
	rm -f $(DESTDIR)$(mandir)/man1/mpimpi2prv.1 \
        $(DESTDIR)$(mandir)/man1/mpimpi2dim.1 \
        $(DESTDIR)$(mandir)/man1/mpi2dim.1 \
	$(DESTDIR)$(mandir)/man1/UFlist.1 \
	$(DESTDIR)$(mandir)/man1/extrae.1 \
	$(DESTDIR)$(mandir)/man1/mpi2prv.1
	rm -f $(DESTDIR)$(mandir)/man3/extrae_event.3 \
        $(DESTDIR)$(mandir)/man3/extrae_counters.3 \
        $(DESTDIR)$(mandir)/man3/extrae_eventandcounters.3 \
        $(DESTDIR)$(mandir)/man3/extrae_restart.3 \
        $(DESTDIR)$(mandir)/man3/extrae_shutdown.3 \
        $(DESTDIR)$(mandir)/man3/extrae_set_tracing_tasks.3 \
        $(DESTDIR)$(mandir)/man3/extrae_next_hwc_set.3 \
        $(DESTDIR)$(mandir)/man3/extrae_previous_hwc_set.3 \
        $(DESTDIR)$(mandir)/man3/mpitrace_event.3 \
	$(DESTDIR)$(mandir)/man3/mpitrace_counters.3 \
        $(DESTDIR)$(mandir)/man3/mpitrace_eventandcounters.3 \
        $(DESTDIR)$(mandir)/man3/mpitrace_restart.3 \
        $(DESTDIR)$(mandir)/man3/mpitrace_shutdown.3 \
        $(DESTDIR)$(mandir)/man3/mpitrace_set_tracing_tasks.3 \
        $(DESTDIR)$(mandir)/man3/mpitrace_next_hwc_set.3 \
        $(DESTDIR)$(mandir)/man3/mpitrace_previous_hwc_set.3
	rmdir --ignore-fail-on-non-empty $(DESTDIR)$(mandir)/man1 \
					 $(DESTDIR)$(mandir)/man3 \
					 $(DESTDIR)$(mandir)
