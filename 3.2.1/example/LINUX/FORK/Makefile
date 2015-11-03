include Makefile.inc

CFLAGS = -O -g -I $(EXTRAE_HOME)/include

targets: pi_forked

pi_forked:	pi_forked.c
	$(CC) $(CFLAGS) pi_forked.c -o pi_forked

clean:
	rm -fr TRACE*.mpit* TRACE.sym set-0 pi_forked

