include $(top_srcdir)/PATHS

bin_PROGRAMS = extrae

noinst_PROGRAMS = load-module

extrae_SOURCES = extrae.C \
 ompSnippets.C ompSnippets.h \
 apiSnippets.C apiSnippets.h \
 mpiSnippets.C mpiSnippets.h \
 commonSnippets.C commonSnippets.h \
 applicationType.C applicationType.h \
 mini-xml-parse.c mini-xml-parse.h
extrae_CXXFLAGS = -I@DYNINST_INCLUDES@ -I$(COMMON_DIR)
extrae_CFLAGS = -I@DYNINST_INCLUDES@ -I$(COMMON_DIR) @XML2_CFLAGS@
extrae_LDFLAGS =  @DYNINST_LDFLAGS@ -ldyninstAPI @UNWIND_LDFLAGS@ @UNWIND_LIBS@ -lparseAPI -lsymtabAPI -linstructionAPI -lcommon @LIBERTY_LDFLAGS@ @LIBERTY_LIBS@ @XML2_LDFLAGS@ @XML2_LIBS@ @DWARF_LDFLAGS@ -ldwarf @LIBELF_LDFLAGS@ -lelf

load_module_SOURCES = load-module.C
load_module_CXXFLAGS = -I@DYNINST_INCLUDES@ -I$(COMMON_DIR)
load_module_LDFLAGS = @DYNINST_LDFLAGS@ -ldyninstAPI @UNWIND_LDFLAGS@ @UNWIND_LIBS@ -lparseAPI -lsymtabAPI -linstructionAPI -lcommon @LIBERTY_LDFLAGS@ @LIBERTY_LIBS@ @DWARF_LDFLAGS@ -ldwarf @LIBELF_LDFLAGS@ -lelf
