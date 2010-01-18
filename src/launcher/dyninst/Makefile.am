include $(top_srcdir)/PATHS

bin_PROGRAMS = ompitrace

ompitrace_SOURCES = ompitrace.C \
 ompSnippets.C \
 apiSnippets.C \
 mpiSnippets.C \
 commonSnippets.C \
 applicationType.C \
 mini-xml-parse.c

ompitrace_CXXFLAGS = -I@DYNINST_INCLUDES@ -I$(COMMON_DIR)
ompitrace_CFLAGS = -I@DYNINST_INCLUDES@ -I$(COMMON_DIR) @XML2_CFLAGS@
ompitrace_LDFLAGS =  @DYNINST_LDFLAGS@ -ldyninstAPI -lelf -liberty @UNWIND_LDFLAGS@ @UNWIND_LIBS@ -lsymtabAPI -lcommon @LIBERTY_LDFLAGS@ @LIBERTY_LIBS@ @XML2_LDFLAGS@ @XML2_LIBS@ -L/home/bsc42/bsc42145/local/lib -ldwarf
