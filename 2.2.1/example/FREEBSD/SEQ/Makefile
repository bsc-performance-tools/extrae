EXTRAE_HOME=@sub_PREFIXDIR@
CC = gcc
CFLAGS = -O -g -I $(EXTRAE_HOME)/include
LFLAGS = -L$(EXTRAE_HOME)/lib -lseqtrace `xml2-config --libs` @sub_BFD_LIBS@ @sub_LIBERTY_LIBS@ @sub_libexecinfo_libs@ @sub_POSIX_CLOCK@

pi:	pi.c
	$(CC) $(CFLAGS) pi.c -o pi

pi_instrumented:	pi_instrumented.c
	$(CC) $(CFLAGS) pi_instrumented.c -o pi_instrumented $(LFLAGS)

clean:
	rm -fr TRACE*.mpit* pi set-0

