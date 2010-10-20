EXTRAE_HOME=@sub_PREFIXDIR@
CC = gcc
CFLAGS = -O -g
LFLAGS = -L $(EXTRAE_HOME)/lib -lseqtrace -lxml2 @sub_BFD_LIBS@ @sub_LIBERTY_LIBS@

pi:	pi.c
	$(CC) $(CFLAGS) pi.c -o pi $(LFLAGS)

clean:
	rm -fr TRACE*.mpit* pi set-0

