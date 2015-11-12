include $(top_srcdir)/PATHS

EXTRA_DIST = \
 extrae_JavaSimple.sh \
 extrae_JavaGC.sh \
 extrae_JavaException.sh \
 extrae_JavaThreads.sh \
 extrae.xml

check_PROGRAMS = \
 JavaSimple.class \
 JavaGC.class \
 JavaException.class \
 JavaThreads.class

TESTS = \
  extrae_JavaSimple.sh \
  extrae_JavaGC.sh \
  extrae_JavaException.sh \
  extrae_JavaThreads.sh

JavaSimple.class$(EXEEXT): JavaSimple.java
	$(JAVAC) JavaSimple.java

JavaGC.class$(EXEEXT): JavaGC.java
	$(JAVAC) JavaGC.java

JavaException.class$(EXEEXT): JavaException.java
	$(JAVAC) JavaException.java

JavaThreads.class$(EXEEXT): JavaThreads.java
	$(JAVAC) JavaThreads.java
