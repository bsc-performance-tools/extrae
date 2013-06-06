include ../Makefile.inc

CC=gcc

CFLAGS= -g -I$(OPENCL_HOME)/include -O3 -lm -Wall
LIBS = $(OPENCL_LDFLAGS) -lOpenCL

vadd: vadd.c 
	$(CC) $^ $(CFLAGS) $(LIBS) -o $@

clean:
	rm -fr vadd set-0 TRACE.mpits TRACE.sym
