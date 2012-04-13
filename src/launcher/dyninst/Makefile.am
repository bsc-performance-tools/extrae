include $(top_srcdir)/PATHS

bin_PROGRAMS = extrae

noinst_PROGRAMS = load-module

if DYNINST_HAVE_PATCHAPI
DYNINST_EXTRA_LIBS = -lpatchAPI
endif

extrae_SOURCES = extrae.C \
 cudaSnippets.C cudaSnippets.h \
 ompSnippets.C ompSnippets.h \
 apiSnippets.C apiSnippets.h \
 mpiSnippets.C mpiSnippets.h \
 commonSnippets.C commonSnippets.h \
 applicationType.C applicationType.h \
 mini-xml-parse.c mini-xml-parse.h
extrae_CXXFLAGS = -I$(COMMON_DIR) -I@DYNINST_INCLUDES@
extrae_CFLAGS = -I$(COMMON_DIR) -I@DYNINST_INCLUDES@ @XML2_CFLAGS@
extrae_LDFLAGS =  @DYNINST_LDFLAGS@ -ldyninstAPI @UNWIND_LDFLAGS@ @UNWIND_LIBS@ -lparseAPI -lsymtabAPI -linstructionAPI $(DYNINST_EXTRA_LIBS) -lcommon @LIBERTY_LDFLAGS@ @LIBERTY_LIBS@ @XML2_LDFLAGS@ @XML2_LIBS@ @DWARF_LDFLAGS@ -ldwarf @ELF_LDFLAGS@ -lelf

load_module_SOURCES = load-module.C
load_module_CXXFLAGS = -I$(COMMON_DIR) -I@DYNINST_INCLUDES@
load_module_LDFLAGS = @DYNINST_LDFLAGS@ -ldyninstAPI @UNWIND_LDFLAGS@ @UNWIND_LIBS@ -lparseAPI -lsymtabAPI -linstructionAPI $(DYNINST_EXTRA_LIBS) -lcommon @LIBERTY_LDFLAGS@ @LIBERTY_LIBS@ @DWARF_LDFLAGS@ -ldwarf @ELF_LDFLAGS@ -lelf
