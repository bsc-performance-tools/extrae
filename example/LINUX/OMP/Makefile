include Makefile.inc

CFLAGS = -O -fopenmp -g -I$(EXTRAE_HOME)/include
CLIBS = -L$(EXTRAE_HOME)/lib -lomptrace

all: pi pi_instrumented

pi:	pi.c
	$(CC) $(CFLAGS) pi.c -o pi 

pi_instrumented:	pi_instrumented.c
	$(CC) $(CFLAGS) pi_instrumented.c -o pi_instrumented $(CLIBS)

clean:
	rm -fr TRACE.sym TRACE.mpits set-0 pi pi_instrumented

