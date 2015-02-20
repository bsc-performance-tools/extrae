include ../Makefile.inc

all: hello hello_instrumented

hello_instrumented: hello_instrumented.cu
	nvcc -g --cudart shared hello_instrumented.cu -o hello_instrumented -I$(EXTRAE_HOME)/include -L$(EXTRAE_HOME)/lib -lcudatrace

hello: hello.cu
	nvcc -g --cudart shared hello.cu -o hello

clean:
	rm -fr hello hello_instrumented TRACE.mpits TRACE.sym set-0

