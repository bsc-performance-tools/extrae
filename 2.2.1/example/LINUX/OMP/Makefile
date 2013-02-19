PAPI_HOME = @sub_PAPI_HOME@
EXTRAE_HOME = @sub_PREFIXDIR@
UNWIND_HOME = @sub_UNWIND_HOME@

CC = gcc
CFLAGS = -O -fopenmp -g -I$(EXTRAE_HOME)/include
CLIBS = -L$(EXTRAE_HOME)/lib -lomptrace -lxml2 -L$(PAPI_HOME)/lib @sub_PAPI_LIBS@ @sub_BFD_LIBS@ @sub_LIBERTY_LIBS@ -L$(UNWIND_HOME)/lib @sub_UNWIND_LIBS@ @sub_POSIX_CLOCK@

all: pi pi_instrumented

pi:	pi.c
	$(CC) $(CFLAGS) pi.c -o pi 

pi_instrumented:	pi_instrumented.c
	$(CC) $(CFLAGS) pi_instrumented.c -o pi_instrumented $(CLIBS)

clean:
	rm -fr TRACE.sym TRACE.mpits set-0 pi pi_instrumented

