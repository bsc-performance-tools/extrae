

man_MANS = UFlist.1 mpi2prv.1 mpitrace.1 mpitrace_event.3

EXTRA_DIST = UFlist.1 mpi2prv.1 mpitrace.1 mpitrace_event.3 

install-exec-hook:
	$(mkdir_p) $(mandir)/man1
	cp -f UFlist.1 mpi2prv.1 mpitrace.1 $(mandir)/man1
	cd  $(mandir)/man1 ; $(LN_S) -f mpi2prv.1 mpi2dim.1
	cd  $(mandir)/man1 ; $(LN_S) -f mpi2prv.1 mpimpi2prv.1
	cd  $(mandir)/man1 ; $(LN_S) -f mpi2prv.1 mpimpi2dim.1
	$(mkdir_p) $(mandir)/man3
	cp -f mpitrace_event.3 $(mandir)/man3
	cd  $(mandir)/man3 ; $(LN_S) -f mpitrace_event.3 mpitrace_counters.3
	cd  $(mandir)/man3 ; $(LN_S) -f mpitrace_event.3 mpitrace_eventandcounters.3
	cd  $(mandir)/man3 ; $(LN_S) -f mpitrace_event.3 mpitrace_restart.3
	cd  $(mandir)/man3 ; $(LN_S) -f mpitrace_event.3 mpitrace_shutdown.3
	cd  $(mandir)/man3 ; $(LN_S) -f mpitrace_event.3 mpitrace_set_tracing_tasks.3
	cd  $(mandir)/man3 ; $(LN_S) -f mpitrace_event.3 mpitrace_next_hwc_set.3
	cd  $(mandir)/man3 ; $(LN_S) -f mpitrace_event.3 mpitrace_previous_hwc_set.3

uninstall-local:
	rm -f $(mandir)/man1/mpimpi2prv.1 \
        $(mandir)/man1/mpimpi2dim.1 \
        $(mandir)/man1/mpi2dim.1
	rm -f $(mandir)/man3/mpitrace_counters.3 \
        $(mandir)/man3/mpitrace_eventandcounters.3 \
        $(mandir)/man3/mpitrace_restart.3 \
        $(mandir)/man3/mpitrace_shutdown.3 \
        $(mandir)/man3/mpitrace_set_tracing_tasks.3 \
        $(mandir)/man3/mpitrace_next_hwc_set.3 \
        $(mandir)/man3/mpitrace_previous_hwc_set.3
