include ../Makefile.inc

CC     = mpicc
CFLAGS = -O2 -g

all: ping

ping: ping.c
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -fr ping *.o  TRACE.* set-0

