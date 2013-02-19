EXTRAE_HOME=@sub_PREFIXDIR@
UNWIND_HOME=@sub_UNWIND_HOME@

CC = gcc
CFLAGS = -O -g -I $(EXTRAE_HOME)/include
LFLAGS = -L$(EXTRAE_HOME)/lib -lseqtrace -lxml2 -L$(UNWIND_HOME)/lib @sub_UNWIND_LIBS@ @sub_POSIX_CLOCK@

targets: pi pi_instrumented

pi:	pi.c
	$(CC) $(CFLAGS) pi.c -o pi

pi_instrumented:	pi_instrumented.c
	$(CC) $(CFLAGS) pi_instrumented.c -o pi_instrumented $(LFLAGS)

clean:
	rm -fr TRACE*.mpit* TRACE.sym pi pi_instrumented set-0

