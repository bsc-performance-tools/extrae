include ../Makefile.inc

CC = gcc
FC = gfortran
CFLAGS = -O -g -I $(EXTRAE_HOME)/include
FFLAGS = -O -g -I $(EXTRAE_HOME)/include
LFLAGS = -L$(EXTRAE_HOME)/lib -lseqtrace

TARGETS = pi pi_instrumented pi_instrumentedf

targets: $(TARGETS)

pi:	pi.c
	$(CC) $(CFLAGS) pi.c -o pi

pi_instrumented:	pi_instrumented.c
	$(CC) $(CFLAGS) pi_instrumented.c -o pi_instrumented $(LFLAGS)

pi_instrumentedf:   pi_instrumented.f90
	$(FC) -c $(EXTRAE_HOME)/include/extrae_module.f90
	$(FC) $(FFLAGS) pi_instrumented.f90 -o pi_instrumentedf $(LFLAGS)

clean:
	rm -fr TRACE*.mpit* TRACE.sym set-0 $(TARGETS) *.o *.mod

