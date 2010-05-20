
PACX_HOME=@sub_PACX_HOME@
MPITRACE_HOME=@sub_PREFIXDIR@

example:
	$(PACX_HOME)/bin/pacxcc -O example.c -L $(MPITRACE_HOME)/lib -lpacxtrace -lppacx -o example

clean:
	rm pacx_example

