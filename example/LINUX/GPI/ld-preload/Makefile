include ../Makefile.inc

CFLAGS += $(GPI_CFLAGS)

APP_NAME = write_bw

all: $(APP_NAME)

OBJECTS = common.o

%.o: %.c
	$(CC) $(CFLAGS) -c $< $(GPI_LIBS)

$(APP_NAME): $(APP_NAME).c $(OBJECTS)
	$(CC) $(CFLAGS) $< -o $@ $(OBJECTS) $(GPI_LIBS)

clean:
	rm -fr $(APP_NAME) *.o pingtmp? TRACE.* set-* *.row *.prv *.pcf
