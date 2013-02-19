include $(top_srcdir)/PATHS

EXTRA_DIST = \
 extrae-nothing-to-instrument.xml \
 extrae.xml \
 function-list \
 test-binary-rewrite.sh \
 test-dyninst-2.reference \
 test-dyninst-2.sh test-dyninst.sh 

if HAVE_DYNINST

check_PROGRAMS = pi

TESTS = \
 test-binary-rewrite.sh \
 test-dyninst.sh \
 test-dyninst-2.sh

pi_SOURCES = pi.c

endif

