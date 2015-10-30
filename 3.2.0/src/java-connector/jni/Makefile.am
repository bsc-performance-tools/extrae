include $(top_srcdir)/PATHS

EXTRA_DIST=es

JNI_INCLUDE_DIR=include

all: javaseqtrace.jar

javaseqtrace.jar: es/bsc/cepbatools/extrae/Wrapper.java
	$(JAVAC) $(BASE_DIR)/src/java-connector/jni/es/bsc/cepbatools/extrae/Wrapper.java -d .
	$(JAVAH) -d $(JNI_INCLUDE_DIR) -jni es.bsc.cepbatools.extrae.Wrapper
	$(JAR) cvf javaseqtrace.jar es/bsc/cepbatools/extrae/Wrapper.class 

clean-local:
	rm -fr javaseqtrace.jar es/bsc/cepbatools/extrae/Wrapper.class $(JNI_INCLUDE_DIR)

install-data-local: javaseqtrace.jar
	cp javaseqtrace.jar $(libdir)

