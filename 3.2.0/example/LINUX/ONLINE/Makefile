include Makefile.inc

CFLAGS = -O2 -g

all: ping

ping: ping.c
	$(MPICC) $(CFLAGS) $^ -o $@

clean:
	rm -fr ping *.o  TRACE.* set-0

