EXTRAE_HOME=@sub_PREFIXDIR@
CC = gcc
CFLAGS = -O -g
LFLAGS = -L $(EXTRAE_HOME)/lib -lseqtrace -lxml2 -ldl

pi:	pi.c
	$(CC) $(CFLAGS) pi.c -o pi $(LFLAGS)

clean:
	rm -f TRACE*.mpit* pi

