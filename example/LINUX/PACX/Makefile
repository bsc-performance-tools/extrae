PACX_HOME=@sub_PACX_HOME@
EXTRAE_HOME=@sub_PREFIXDIR@
UNWIND_HOME = @sub_UNWIND_HOME@

LIBS=-L$(EXTRAE_HOME)/lib -lpacxtrace -lppacx @sub_BFD_LIBS@ @sub_LIBERTY_LIBS@ @sub_POSIX_CLOCK@ -L$(UNWIND_HOME)/lib @sub_UNWIND_LIBS@

example:
	$(PACX_HOME)/bin/pacxcc -O -g example.c -o example $(LIBS)

clean:
	rm pacx_example

