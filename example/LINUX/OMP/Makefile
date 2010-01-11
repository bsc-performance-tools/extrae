
CC = gcc
FC = ifort
CFLAGS = -O -fopenmp -g
FFLAGS = -O -openmp -g3
MPITRACE_HOME = ../../..

all: pi_f pi_c

pi_c:	pi.c
	$(CC) $(CFLAGS) pi.c -o pi_c $(LFLAGS) $(MPITRACE_HOME)/lib/lib_dyn_omptrace.a -lxml2 -L /apps/PAPI/3.6.2/lib -lpapi

pi_f:	pi.f
	$(FC) $(CFLAGS) pi.f -o pi_f $(LFLAGS)

clean:
	rm -f TRACE*.mpit pi_c pi_f

