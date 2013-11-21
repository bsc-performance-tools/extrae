include ../Makefile.inc

CC	= imcc --ompss

LDFLAGS = -L$(MPI_HOME)/lib -lmpi

CFLAGS = -I$(MPI_HOME)/include

CFLAGS += --instrument

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
	rm heatmpi-ompss-i *.o
