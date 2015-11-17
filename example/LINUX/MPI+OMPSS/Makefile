include Makefile.inc

CC	= imcc

LDFLAGS = -g -O $(MPI_LDFLAGS) $(MPI_LIBS)
CFLAGS =  --ompss -g -O -instrument $(MPI_CFLAGS)

OBJECTS = misc.o solver-mpi-ompss.o heat-mpi-ompss.o

heatmpi-ompss-i: $(OBJECTS)
	$(CC) $(CFLAGS) -o heatmpi-ompss-i $(OBJECTS) $(LDFLAGS)

misc.o: misc.c
	$(CC) -c $(CFLAGS) $< -o $@

solver-mpi-ompss.o: solver-mpi-ompss.c
	$(CC) -c $(CFLAGS) $< -o $@

heat-mpi-ompss.o: heat-mpi-ompss.c
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -fr heatmpi-ompss-i *.o TRACE.* set-0
