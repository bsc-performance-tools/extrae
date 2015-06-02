include ../Makefile.inc

NVCCFLAGS=-O -g --cudart shared -I../../../include
LDFLAGS=-Wl,-rpath -Wl,$(EXTRAE_HOME)/lib -L$(EXTRAE_HOME)/lib -lcudatrace

all: hello hello_instrumented

hello_instrumented: hello_instrumented.cu
	nvcc $(NVCCFLAGS) hello_instrumented.cu -o hello_instrumented $(LDFLAGS)

hello: hello.cu
	nvcc $(NVCCFLAGS) hello.cu -o hello

clean:
	rm -fr hello hello_instrumented TRACE.mpits TRACE.sym set-0

