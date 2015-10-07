include Makefile.inc

CFLAGS = -O -g -I $(EXTRAE_HOME)/include
LFLAGS = -L$(EXTRAE_HOME)/lib -lseqtrace $(UNWIND_LIBS) $(XML2_LIBS) $(POSIX_CLOCK_LIBS) $(LIBEXECINFO_LIBS)

targets: pi pi_instrumented

pi:	pi.c
	$(CC) $(CFLAGS) pi.c -o pi

pi_instrumented:	pi_instrumented.c
	$(CC) $(CFLAGS) pi_instrumented.c -o pi_instrumented $(LFLAGS)

clean:
	rm -fr TRACE*.mpit* TRACE.sym pi pi_instrumented set-0

