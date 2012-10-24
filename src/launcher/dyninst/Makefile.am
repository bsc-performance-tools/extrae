include $(top_srcdir)/PATHS

bin_PROGRAMS = extrae

noinst_PROGRAMS = load-module

DYNINST_EXTRA_LIBS=

# After 7.0.1 there were several steps to fragment dyninst
# Step #1, add patch api
if DYNINST_HAVE_PATCHAPI
DYNINST_EXTRA_LIBS += -lpatchAPI
endif
# Step #2, add pcontrol, stackwalk, dynDwarf/Elf and symLite
if DYNINST_HAVE_STACKWALK
DYNINST_EXTRA_LIBS += -lpcontrol -lstackwalk -lcommon -ldynDwarf -ldynElf -lsymLite 
endif

extrae_SOURCES = extrae.C \
 cudaSnippets.C cudaSnippets.h \
 ompSnippets.C ompSnippets.h \
 apiSnippets.C apiSnippets.h \
 mpiSnippets.C mpiSnippets.h \
 commonSnippets.C commonSnippets.h \
 applicationType.C applicationType.h \
 mini-xml-parse.c mini-xml-parse.h
extrae_CXXFLAGS = -I$(COMMON_DIR) -I@DYNINST_INCLUDES@ -I@BOOST_HOME@/include
extrae_CFLAGS = -I$(COMMON_DIR) -I@DYNINST_INCLUDES@ @XML2_CFLAGS@
extrae_LDFLAGS =  @DYNINST_LDFLAGS@ -ldyninstAPI @UNWIND_LDFLAGS@ @UNWIND_LIBS@ -lparseAPI -lsymtabAPI -linstructionAPI $(DYNINST_EXTRA_LIBS) -lcommon @DWARF_LDFLAGS@ -ldwarf @ELF_LDFLAGS@ -lelf @XML2_LDFLAGS@ @XML2_LIBS@

load_module_SOURCES = load-module.C
load_module_CXXFLAGS = -I$(COMMON_DIR) -I@DYNINST_INCLUDES@ -I@BOOST_HOME@/include
load_module_LDFLAGS = @DYNINST_LDFLAGS@ -ldyninstAPI @UNWIND_LDFLAGS@ @UNWIND_LIBS@ -lparseAPI -lsymtabAPI -linstructionAPI $(DYNINST_EXTRA_LIBS) -lcommon @DWARF_LDFLAGS@ -ldwarf @ELF_LDFLAGS@ -lelf @XML2_LDFLAGS@ @XML2_LIBS@
