include ../Makefile.inc

LIBS=-L$(EXTRAE_HOME)/lib -lpacxtrace -lppacx

example:
	$(PACX_HOME)/bin/pacxcc -O -g example.c -o example $(LIBS)

clean:
	rm pacx_example

