
MPITRACE_HOME = ../../..

PPU_LIB = $(MPITRACE_HOME)/lib/libmpitrace.a
SPU_LIB = $(MPITRACE_HOME)/lib/libsputrace.a

SPU_CC = spu-cc
PPU_CC = /usr/local/share/mpich-1.2.7p1/bin/mpicc
PPU_CFLAGS = -O2 -Wall -I$(MPITRACE_HOME)/include
SPU_CFLAGS = -O2 -Wall -I$(MPITRACE_HOME)/include
EXTRA = -lxml2

all: master worker

bitmap.o: bitmap.c bitmap.h
	$(PPU_CC) -c bitmap.c $(PPU_CFLAGS) -o bitmap.o

master.o: master.c bitmap.h
	$(PPU_CC) -c master.c $(PPU_CFLAGS) -o master.o

master:	master.o bitmap.o
	$(PPU_CC) -o master master.o bitmap.o -lspe2 $(PPU_LIB) $(EXTRA) $(PPU_CFLAGS)

worker.o: worker.c
	$(SPU_CC) $(SPU_CFLAGS) -c worker.c -o worker.o

worker:	worker.o
	$(SPU_CC) worker.o -o worker $(SPU_LIB) $(SPU_CFLAGS)

clean: 
	rm -fr *.o worker master core.* *.mpit* *.prv *.pcf *.row

mrproper: clean
	rm -fr *.prv *.pcf *.mpit* *.tmp *.row
	
