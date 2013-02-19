EXTRAE_HOME = @sub_PREFIXDIR@

hello: hello.cu
	nvcc -g hello.cu -o hello -I$(EXTRAE_HOME)/include -L$(EXTRAE_HOME)/lib -lcudatrace @sub_CUPTI_LIBS@

clean:
	rm -f hello

