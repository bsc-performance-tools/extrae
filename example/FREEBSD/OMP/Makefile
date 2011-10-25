PAPI_HOME = @sub_PAPI_HOME@
EXTRAE_HOME = @sub_PREFIXDIR@
CC = gcc
FC = ifort
CFLAGS = -O -fopenmp -g -I $(EXTRAE_HOME)/include
CLIBS = -L$(EXTRAE_HOME)/lib -lomptrace `xml2-config --libs` -L $(PAPI_HOME)/lib @sub_PAPI_LIBS@ @sub_BFD_LIBS@ @sub_LIBERTY_LIBS@ @sub_libexecinfo_libs@ @sub_POSIX_CLOCK@
FFLAGS = -O -openmp -g3

all: pi pi_instrumented

pi:	pi.c
	$(CC) $(CFLAGS) pi.c -o pi 

pi_instrumented:	pi_instrumented.c
	$(CC) $(CFLAGS) pi_instrumented.c -o pi_instrumented $(CLIBS)

clean:
	rm -fr TRACE.sym TRACE.mpits set-0 pi pi_instrumented
