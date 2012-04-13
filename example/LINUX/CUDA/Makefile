EXTRAE_HOME = @sub_PREFIXDIR@

all: hello hello_instrumented

hello_instrumented: hello_instrumented.cu
	nvcc -g hello_instrumented.cu -o hello_instrumented -I$(EXTRAE_HOME)/include -L$(EXTRAE_HOME)/lib -lcudatrace @sub_CUPTI_LIBS@

hello: hello.cu
	nvcc -g hello.cu -o hello

clean:
	rm -fr hello hello_instrumented TRACE.mpits set-0

