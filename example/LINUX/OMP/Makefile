PAPI_HOME = @sub_PAPI_HOME@
EXTRAE_HOME = @sub_PREFIXDIR@

CC = gcc
FC = ifort
CFLAGS = -O -fopenmp -g -I$(EXTRAE_HOME)/include
CLIBS = -L$(EXTRAE_HOME)/lib -lomptrace -lxml2 -L $(PAPI_HOME)/lib @sub_PAPI_LIBS@ @sub_BFD_LIBS@ @sub_LIBERTY_LIBS@  @sub_POSIX_CLOCK@
FFLAGS = -O -openmp -g3

all: pi_f pi_c

pi_c:	pi.c
	$(CC) $(CFLAGS) pi.c -o pi_c $(CLIBS)

pi_f:	pi.f
	$(FC) $(CFLAGS) pi.f -o pi_f $(LFLAGS)

clean:
	rm -f TRACE*.mpit pi_c pi_f

