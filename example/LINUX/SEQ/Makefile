include Makefile.inc

CFLAGS = -O -g -I $(EXTRAE_HOME)/include
FFLAGS = -O -g -I $(EXTRAE_HOME)/include
LFLAGS = -Wl,-rpath -Wl,$(EXTRAE_HOME)/lib -L$(EXTRAE_HOME)/lib -lseqtrace

TARGETS = pi pi_instrumented pi_instrumentedf pi_instrumented_functions

targets: $(TARGETS)

pi:	pi.c
	$(CC) $(CFLAGS) pi.c -o pi

pi_instrumented_functions:	pi.c
	$(CC) $(CFLAGS) -finstrument-functions pi.c -o pi_instrumented_functions
	nm pi_instrumented_functions | grep pi_kernel | sed s/T/#/ > list-instrumented-functions

pi_instrumented:	pi_instrumented.c
	$(CC) $(CFLAGS) pi_instrumented.c -o pi_instrumented $(LFLAGS)

pi_instrumentedf:   pi_instrumented.f90
	# To use the Extrae API in Fortran you need to compile extrae_module with
	# the same compiler and include it with `use extrae_module` in your program
	make -f $(EXTRAE_HOME)/include/Makefile.extrae_module
	$(FC) $(FFLAGS) pi_instrumented.f90 -o pi_instrumentedf $(LFLAGS)

clean:
	rm -fr TRACE*.mpit* TRACE.sym set-0 $(TARGETS) *.o *.mod

