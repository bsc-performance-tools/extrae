include $(top_srcdir)/PATHS

bin_PROGRAMS = ompitrace

noinst_PROGRAMS = load-module

ompitrace_SOURCES = ompitrace.C \
 ompSnippets.C ompSnippets.h \
 apiSnippets.C apiSnippets.h \
 mpiSnippets.C mpiSnippets.h \
 commonSnippets.C commonSnippets.h \
 applicationType.C applicationType.h \
 mini-xml-parse.c mini-xml-parse.h

ompitrace_CXXFLAGS = -I@DYNINST_INCLUDES@ -I$(COMMON_DIR)
ompitrace_CFLAGS = -I@DYNINST_INCLUDES@ -I$(COMMON_DIR) @XML2_CFLAGS@
ompitrace_LDFLAGS =  @DYNINST_LDFLAGS@ -ldyninstAPI -lelf -liberty @UNWIND_LDFLAGS@ @UNWIND_LIBS@ -lsymtabAPI -linstructionAPI -lcommon @LIBERTY_LDFLAGS@ @LIBERTY_LIBS@ @XML2_LDFLAGS@ @XML2_LIBS@ -L/home/bsc42/bsc42145/local/lib64 -ldwarf
load_module_SOURCES = load-module.C

load_module_CXXFLAGS = -I@DYNINST_INCLUDES@ -I$(COMMON_DIR)
load_module_LDFLAGS = @DYNINST_LDFLAGS@ -ldyninstAPI -lelf -liberty @UNWIND_LDFLAGS@ @UNWIND_LIBS@ -lsymtabAPI -linstructionAPI -lcommon @LIBERTY_LDFLAGS@ @LIBERTY_LIBS@ -L/home/bsc42/bsc42145/local/lib64 -ldwarf
