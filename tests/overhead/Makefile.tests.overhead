include ../../example/Makefile.inc

CFLAGS = -O -g -I $(EXTRAE_HOME)/include -I $(PAPI_HOME)/include
LFLAGS = -L$(EXTRAE_HOME)/lib -Wl,-rpath -Wl,$(EXTRAE_HOME)/lib -lseqtrace

TARGETS = posix_clock ia32_rdtsc_clock extrae_event extrae_nevent4 extrae_eventandcounters extrae_user_function extrae_get_caller1 extrae_get_caller6 extrae_trace_callers papi_read1 papi_read4

targets: $(TARGETS)

posix_clock:	posix_clock.c
	$(CC) $(CFLAGS) $< -o $@ $(LFLAGS)

ia32_rdtsc_clock:	ia32_rdtsc_clock.c
	$(CC) $(CFLAGS) $< -o $@ $(LFLAGS)

extrae_event:	extrae_event.c
	$(CC) $(CFLAGS) $< -o $@ $(LFLAGS)

extrae_nevent4:	extrae_nevent4.c
	$(CC) $(CFLAGS) $< -o $@ $(LFLAGS)

extrae_eventandcounters:	extrae_eventandcounters.c
	$(CC) $(CFLAGS) $< -o $@ $(LFLAGS)

extrae_user_function:	extrae_user_function.c
	$(CC) $(CFLAGS) $< -o $@ $(LFLAGS)

extrae_get_caller1:	extrae_get_caller1.c
	$(CC) $(CFLAGS) $< -o $@ $(LFLAGS)

extrae_get_caller6:	extrae_get_caller6.c
	$(CC) $(CFLAGS) $< -o $@ $(LFLAGS)

extrae_trace_callers:	extrae_trace_callers.c
	$(CC) $(CFLAGS) $< -o $@ $(LFLAGS)

papi_read1:	papi_read1.c
	$(CC) $(CFLAGS) $< -o $@ -L$(PAPI_HOME)/lib -Wl,-rpath -Wl,$(PAPI_HOME)/lib -lpapi -lrt

papi_read4:	papi_read4.c
	$(CC) $(CFLAGS) $< -o $@ -L$(PAPI_HOME)/lib -Wl,-rpath -Wl,$(PAPI_HOME)/lib -lpapi -lrt

JavaEvent.class: JavaEvent.java
	$(JAVAC) $< -classpath $(EXTRAE_HOME)/lib/javatrace.jar

JavaNEvent4.class: JavaNEvent4.java
	$(JAVAC) $< -classpath $(EXTRAE_HOME)/lib/javatrace.jar

JavaFakeRoutine.class: JavaFakeRoutine.java
	$(JAVAC) $<

clean:
	rm -fr TRACE*.mpit* TRACE.sym set-0 $(TARGETS) *.class tmp.* extraej.??????

