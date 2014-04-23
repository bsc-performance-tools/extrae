
pi.class: pi.java
	javac pi.java -classpath ../../../lib/javaseqtrace.jar

run: pi.class
	EXTRAE_CONFIG_FILE=extrae.xml LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):../../../lib CLASSPATH=../../../lib/javaseqtrace.jar:. java pi

clean:
	rm -fr pi.class java.prv java.pcf java.row set-0 TRACE.*
