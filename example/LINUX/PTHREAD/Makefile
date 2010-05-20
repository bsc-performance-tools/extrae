EXTRAE_HOME=@sub_PREFIXDIR@

CC= gcc
CFLAGS= -O -g -pthread
LDFLAGS= -O -g -pthread -L$(EXTRAE_HOME)/lib -lpttrace

example: example.o
	$(CC) $< $(LDFLAGS) -o $@ 

example.o: example.c
	$(CC) -c $< $(CFLAGS) -o $@

clean:
	rm -fr example example.o TRACE*

