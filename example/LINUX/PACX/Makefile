PACX_HOME=$(HOME)/aplic/pacx/07.2009-openmpi
MPITRACE_HOME=../../..

example:
	$(PACX_HOME)/bin/pacxcc -O example.c -L $(MPITRACE_HOME)/lib -lpacxtrace -lppacx -o example

clean:
	rm pacx_example

