EXTRAE_HOME = @sub_PREFIXDIR@
PAPI_HOME = @sub_PAPI_HOME@
XML2_LDFLAGS = @sub_XML2_LDFLAGS@
XML2_LIBS = @sub_XML2_LIBS@

hello: hello.cu
	nvcc -g hello.cu -o hello -I$(EXTRAE_HOME)/include -L$(EXTRAE_HOME)/lib -lcudatrace -L$(PAPI_HOME)/lib @sub_PAPI_LIBS@ $(XML2_LDFLAGS) $(XML2_LIBS) @sub_BFD_LIBS@ @sub_LIBERTY_LIBS@ @sub_POSIX_CLOCK@

clean:
	rm -f hello

