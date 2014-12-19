EXTRAE_HOME=@sub_PREFIXDIR@

pi.class: pi.java
	javac pi.java -classpath ../../../lib/javaseqtrace.jar

run: pi.class
	EXTRAE_CONFIG_FILE=extrae.xml java -Djava.library.path="../../../lib" -classpath ../../../lib/javaseqtrace.jar:. pi

clean:
	rm -fr pi.class java.prv java.pcf java.row set-0 TRACE.*
