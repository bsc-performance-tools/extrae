include ../Makefile.inc

PiExample.jar: PiSerial.class PiThread.class PiThreaded.class PiExample.class
	@echo "Main-Class: PiExample" > MANIFEST
	$(JAR) cfm PiExample.jar MANIFEST PiExample.class PiSerial.class PiThread.class PiThreaded.class
	@rm MANIFEST

PiExample.class: PiExample.java
	$(JAVAC) $< -classpath $(EXTRAE_HOME)/lib/javatrace.jar:.

PiSerial.class: PiSerial.java
	$(JAVAC) $< -classpath $(EXTRAE_HOME)/lib/javatrace.jar:.

PiThread.class: PiThread.java
	$(JAVAC) $< -classpath $(EXTRAE_HOME)/lib/javatrace.jar:.

PiThreaded.class: PiThreaded.java
	$(JAVAC) $< -classpath $(EXTRAE_HOME)/lib/javatrace.jar:.

run: PiExample.jar
	EXTRAE_CONFIG_FILE=extrae.xml $(EXTRAE_HOME)/bin/extraej -- PiExample

clean:
	rm -fr *class *jar set-0 TRACE.* *pcf *row *prv
