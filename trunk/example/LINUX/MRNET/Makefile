
CC     = mpicc
CFLAGS = -O2

all: test

test: test.c
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -f test *.o 

