include Makefile.inc

pi.class: pi.java
	$(JAVAC) pi.java -classpath ../../../lib/javaseqtrace.jar

run: pi.class
	EXTRAE_CONFIG_FILE=extrae.xml java -Djava.library.path=$(EXTRAE_HOME)/lib -classpath $(EXTRAE_HOME)/lib/javaseqtrace.jar:. pi

clean:
	rm -fr pi.class java.prv java.pcf java.row set-0 TRACE.*
