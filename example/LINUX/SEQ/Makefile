EXTRAE_HOME=@sub_PREFIXDIR@
CC = gcc
CFLAGS = -O -g -I $(EXTRAE_HOME)/include
LFLAGS = -L $(EXTRAE_HOME)/lib -lseqtrace -lxml2   

targets: pi pi_instrumented

pi:	pi.c
	$(CC) $(CFLAGS) pi.c -o pi

pi_instrumented:	pi_instrumented.c
	$(CC) $(CFLAGS) pi_instrumented.c -o pi_instrumented $(LFLAGS)

clean:
	rm -fr TRACE*.mpit* TRACE.sym pi pi_instrumented set-0

