
CC = gcc
FC = ifort
CFLAGS = -O -fopenmp -g
FFLAGS = -O -openmp -g3
PAPI_HOME = @sub_PAPI_HOME@
EXTRAE_HOME = @sub_PREFIXDIR@

all: pi_f pi_c

pi_c:	pi.c
	$(CC) $(CFLAGS) pi.c -o pi_c $(LFLAGS) $(EXTRAE_HOME)/lib/lib_dyn_omptrace.a -lxml2 -L $(PAPI_HOME)/lib @sub_PAPI_LIBS@

pi_f:	pi.f
	$(FC) $(CFLAGS) pi.f -o pi_f $(LFLAGS)

clean:
	rm -f TRACE*.mpit pi_c pi_f

