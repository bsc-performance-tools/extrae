include ../Makefile.inc

CFLAGS= -g -I$(OPENCL_HOME)/include -O3 -lm -Wall

PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LIBS = -framework OpenCL
else
	LIBS = $(OPENCL_LDFLAGS) -lOpenCL
endif

vadd: vadd.c 
	$(CC) $^ $(CFLAGS) $(LIBS) -o $@

clean:
	rm -fr vadd set-0 TRACE.mpits TRACE.sym
