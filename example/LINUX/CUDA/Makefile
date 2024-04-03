include Makefile.inc

NVCCFLAGS=-O2 -g --generate-line-info --cudart shared -I$(EXTRAE_HOME)/include
LDFLAGS=-L$(EXTRAE_HOME)/lib -lcudatrace -Xlinker -rpath -Xlinker $(EXTRAE_HOME)/lib

all: hello hello_instrumented

hello_instrumented: hello_instrumented.cu
	$(NVCC) $(NVCCFLAGS) hello_instrumented.cu -o hello_instrumented $(LDFLAGS)

hello: hello.cu
	$(NVCC) $(NVCCFLAGS) hello.cu -o hello

clean:
	rm -fr hello hello_instrumented TRACE.mpits TRACE.sym set-0

