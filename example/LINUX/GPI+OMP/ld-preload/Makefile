include ../Makefile.inc

CFLAGS += $(GPI_CFLAGS) -fopenmp

APP_NAME = write_all_mtt

all: $(APP_NAME)

OBJECTS = utils.o

%.o: %.c
	$(CC) $(CFLAGS) -c $< $(GPI_LIBS)

$(APP_NAME): $(APP_NAME).c $(OBJECTS)
	$(CC) $(CFLAGS) $< $(CLIBS) -o $@ $(OBJECTS) $(GPI_LIBS)

clean:
	rm -fr $(APP_NAME) *.o pingtmp? TRACE.* set-* *.row *.prv *.pcf
