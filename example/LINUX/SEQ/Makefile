EXTRAE_HOME=@sub_PREFIXDIR@
CC = gcc
CFLAGS = -O -g -I $(EXTRAE_HOME)/include
LFLAGS = -L $(EXTRAE_HOME)/lib -lseqtrace -lxml2 @sub_BFD_LIBS@ @sub_LIBERTY_LIBS@ @sub_POSIX_CLOCK@

pi:	pi.c
	$(CC) $(CFLAGS) pi.c -o pi $(LFLAGS)

clean:
	rm -fr TRACE*.mpit* pi set-0

