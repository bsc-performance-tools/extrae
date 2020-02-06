include ../../Makefile.inc

CFLAGS += $(GPI_CFLAGS) -fopenmp

EXTRAE_CLIBS = -L$(EXTRAE_HOME)/lib/ -Wl,-rpath=$(EXTRAE_HOME)/lib/ -lgaspitrace $(UNWIND_LIBS) $(PAPI_LIBS) $(XML2_LDFLAGS) $(XML2_LIBS) $(BFD_LIBS) $(POSIX_CLOCK_LIBS) $(ZLIB_LIBS) $(DL_LIBS)

CLIBS = $(GPI_LIBS) $(EXTRAE_CLIBS)

APP_NAME = write_bw

all: $(APP_NAME)

OBJECTS = common.o

%.o: %.c
	$(CC) $(CFLAGS) -c $< $(GPI_LIBS)

$(APP_NAME): $(APP_NAME).c $(OBJECTS)
	$(CC) $(CFLAGS) $< $(CLIBS) -o $@ $(OBJECTS)

clean:
	rm -fr $(APP_NAME) *.o pingtmp? TRACE.* set-* *.row *.prv *.pcf
