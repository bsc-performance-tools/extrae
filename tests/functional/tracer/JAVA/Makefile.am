include $(top_srcdir)/PATHS

EXTRA_DIST = \
 extrae_JavaSimple.sh \
 extrae_JavaGC.sh \
 extrae_JavaException.sh \
 extrae_JavaThreads.sh \
 extrae_JavaAspectJ.sh \
 extrae.xml \
 extrae-function.xml \
 JavaException.java \
 JavaFunction.java \
 JavaGC.java \
 JavaSimple.java \
 JavaThreads.java

check_PROGRAMS = \
 JavaSimple.class \
 JavaGC.class \
 JavaException.class \
 JavaThreads.class

if WANT_JAVA_WITH_ASPECTJ
check_PROGRAMS += \
 JavaFunction.class
endif

JavaSimple.class$(EXEEXT): JavaSimple.java
	$(JAVAC) JavaSimple.java -d $(top_builddir)/tests/functional/tracer/JAVA

JavaGC.class$(EXEEXT): JavaGC.java
	$(JAVAC) JavaGC.java -d $(top_builddir)/tests/functional/tracer/JAVA

JavaException.class$(EXEEXT): JavaException.java
	$(JAVAC) JavaException.java -d $(top_builddir)/tests/functional/tracer/JAVA

JavaThreads.class$(EXEEXT): JavaThreads.java
	$(JAVAC) JavaThreads.java -d $(top_builddir)/tests/functional/tracer/JAVA

if WANT_JAVA_WITH_ASPECTJ
JavaFunction.class$(EXEEXT): JavaFunction.java
	$(JAVAC) JavaFunction.java -d $(top_builddir)/tests/functional/tracer/JAVA
endif

TESTS = \
	./extrae_JavaSimple.sh \
	./extrae_JavaGC.sh \
	./extrae_JavaException.sh \
	./extrae_JavaThreads.sh
if WANT_JAVA_WITH_ASPECTJ
TESTS += ./extrae_JavaAspectJ.sh
endif

TESTS_ENVIRONMENT = \
  EXTRAE_HOME=$(top_builddir)
