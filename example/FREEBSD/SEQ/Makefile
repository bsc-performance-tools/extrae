EXTRAE_HOME=@sub_PREFIXDIR@
CC = gcc
CFLAGS = -O -g
LFLAGS = -L $(EXTRAE_HOME)/lib -lseqtrace `xml2-config --libs` @sub_BFD_LIBS@ @sub_LIBERTY_LIBS@ @sub_libexecinfo_libs@

pi:	pi.c
	$(CC) $(CFLAGS) pi.c -o pi $(LFLAGS)

clean:
	rm -fr TRACE*.mpit* pi set-0

