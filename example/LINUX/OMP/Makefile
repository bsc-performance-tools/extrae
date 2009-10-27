
CC = gcc
CFLAGS = -O -fopenmp -g
LFLAGS = -L ../../../lib -lomptrace

pi:	pi.c
	$(CC) $(CFLAGS) pi.c -o pi $(LFLAGS)

clean:
	rm -f TRACE*.mpit pi

