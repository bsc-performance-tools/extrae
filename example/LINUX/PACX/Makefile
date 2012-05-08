include ../Makefile.inc

LIBS=-L$(EXTRAE_HOME)/lib -lpacxtrace -lppacx $(BFD_LIBS) $(POSIX_CLOCK_LIBS) $(UNWIND_LIBS)

example:
	$(PACX_HOME)/bin/pacxcc -O -g example.c -o example $(LIBS)

clean:
	rm pacx_example

