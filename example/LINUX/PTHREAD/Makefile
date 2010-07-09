PAPI_HOME = @sub_PAPI_HOME@
EXTRAE_HOME = @sub_PREFIXDIR@

LIBS = -L$(EXTRAE_HOME)/lib -lpttrace @sub_PAPI_LIBS@
CFLAGS = -O -g -I$(EXTRAE_HOME)/include

targets: user-comms pthread-example

user-comms: user-comms.c
	gcc $< $(CFLAGS) $(LIBS) -o $@ 

pthread-example: pthread-example.c
	gcc $< $(CFLAGS) $(LIBS) -o $@ 

run-user-comms:
	LD_LIBRARY_PATH=$(EXTRAE_HOME)/lib EXTRAE_ON=1 ./user-comms

merge-user-comms:
	$(EXTRAE_HOME)/bin/mpi2prv set-0/*.mpit -e ./user-comms

run-pthread-example:
	LD_LIBRARY_PATH=$(EXTRAE_HOME)/lib EXTRAE_ON=1 ./pthread-example

merge-pthread-example:
	$(EXTRAE_HOME)/bin/mpi2prv set-0/*.mpit -e ./pthread-example

clean:
	rm -fr user-comms pthread-example set-0 *.prv *.pcf *.row TRACE.mpits
