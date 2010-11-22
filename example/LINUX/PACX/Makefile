PACX_HOME=@sub_PACX_HOME@
EXTRAE_HOME=@sub_PREFIXDIR@
LIBS=-L $(EXTRAE_HOME)/lib -lpacxtrace -lppacx @sub_BFD_LIBS@ @sub_LIBERTY_LIBS@ @sub_POSIX_CLOCK@

example:
	$(PACX_HOME)/bin/pacxcc -O -g example.c -o example $(LIBS)

clean:
	rm pacx_example

