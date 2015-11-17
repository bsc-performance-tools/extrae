include Makefile.inc

CC	= imcc
LDFLAGS = -ldl -lrt -lpthread 
CFLAGS  = --ompss --instrument
OBJECTS = misc.o solver-ompss.o heat-ompss.o

heat-ompss-i: $(OBJECTS)
	$(CC) $(CFLAGS) -o heat-ompss-i $(OBJECTS) $(LDFLAGS)

misc.o: misc.c
	$(CC) -c $(CFLAGS) $< -o $@

solver-ompss.o: solver-ompss.c
	$(CC) -c $(CFLAGS) $< -o $@

heat-ompss.o: heat-ompss.c
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -fr TRACE.* heat-ompss-i *.o set-0
