CC=gcc

CCFLAGS=-O3 -lm 
LIBS = -lOpenCL

vadd: vadd.c 
	$(CC) $^ $(CCFLAGS) $(LIBS) -o $@

clean:
	rm -fr vadd set-0 TRACE.mpits TRACE.sym
