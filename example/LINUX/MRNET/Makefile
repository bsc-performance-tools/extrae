
CC     = mpicc
CFLAGS = -O2 -g

all: test

test: test.c
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -f test *.o 

