
EXTRAE_HOME = @sub_PREFIXDIR@
MPI_HOME = @sub_MPI_HOME@

PPU_LIB = $(EXTRAE_HOME)/lib/libmpitrace.a
SPU_LIB = $(EXTRAE_HOME)/lib/libsputrace.a

SPU_CC = spu-cc
PPU_CC = $(MPI_HOME)/bin/mpicc
PPU_CFLAGS = -O2 -Wall -I$(EXTRAE_HOME)/include
SPU_CFLAGS = -O2 -Wall -I$(EXTRAE_HOME)/include
EXTRA = -lxml2 @sub_BFD_LIBS@ @sub_LIBERTY_LIBS@

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
	rm -fr *.o worker master core.* *.mpit* *.sym *.prv *.pcf *.row

mrproper: clean
	rm -fr *.prv *.pcf *.mpit* *.tmp *.row *.sym
	
