include $(top_srcdir)/PATHS

EXTRA_DIST = test.C

bin_PROGRAMS = extrae extrae-test-dyninst

noinst_PROGRAMS = load-module demo-launcher list-functions

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
 forkSnippets.C forkSnippets.h \
 cudaSnippets.C cudaSnippets.h \
 ompSnippets.C ompSnippets.h \
 apiSnippets.C apiSnippets.h \
 mpiSnippets.C mpiSnippets.h \
 commonSnippets.C commonSnippets.h \
 cpp_utils.C cpp_utils.h \
 applicationType.C applicationType.h \
 mini-xml-parse.c mini-xml-parse.h
extrae_CXXFLAGS = -I$(COMMON_INC) @DYNINST_CXXFLAGS@ -I@DYNINST_INCLUDES@
if NEED_BOOST_HOME
extrae_CXXFLAGS += -I@BOOST_HOME@/include
endif
extrae_CFLAGS = -I$(COMMON_INC) -I@DYNINST_INCLUDES@ @XML2_CFLAGS@
extrae_LDFLAGS =  @DYNINST_LDFLAGS@ -R @DYNINST_LIBSDIR@ -ldyninstAPI @UNWIND_LDFLAGS@ @UNWIND_LIBS@ -lparseAPI -lsymtabAPI -linstructionAPI $(DYNINST_EXTRA_LIBS) -lcommon \
  @DWARF_LDFLAGS@ -R @DWARF_LIBSDIR@ -ldwarf \
  @ELF_LDFLAGS@ -R @ELF_LIBSDIR@ -lelf \
  @XML2_LDFLAGS@ @XML2_LIBS@

load_module_SOURCES = load-module.C
load_module_CXXFLAGS = -I$(COMMON_INC) @DYNINST_CXXFLAGS@ -I@DYNINST_INCLUDES@
if NEED_BOOST_HOME
load_module_CXXFLAGS += -I@BOOST_HOME@/include
endif
load_module_LDFLAGS = @DYNINST_LDFLAGS@ -R @DYNINST_LIBSDIR@ -ldyninstAPI @UNWIND_LDFLAGS@ @UNWIND_LIBS@ -lparseAPI -lsymtabAPI -linstructionAPI $(DYNINST_EXTRA_LIBS) -lcommon \
  @DWARF_LDFLAGS@ -R @DWARF_LIBSDIR@ -ldwarf \
  @ELF_LDFLAGS@ -R @ELF_LIBSDIR@ -lelf \
  @XML2_LDFLAGS@ @XML2_LIBS@

demo_launcher_SOURCES = demo-launcher.C
demo_launcher_CXXFLAGS = -I$(COMMON_INC) @DYNINST_CXXFLAGS@ -I@DYNINST_INCLUDES@
if NEED_BOOST_HOME
demo_launcher_CXXFLAGS += -I@BOOST_HOME@/include
endif
demo_launcher_LDFLAGS = @DYNINST_LDFLAGS@ -R @DYNINST_LIBSDIR@ -ldyninstAPI @UNWIND_LDFLAGS@ @UNWIND_LIBS@ -lparseAPI -lsymtabAPI -linstructionAPI $(DYNINST_EXTRA_LIBS) -lcommon \
  @DWARF_LDFLAGS@ -R @DWARF_LIBSDIR@ -ldwarf \
  @ELF_LDFLAGS@ -R @ELF_LIBSDIR@ -lelf \
  @XML2_LDFLAGS@ @XML2_LIBS@

list_functions_SOURCES = list-functions.C
list_functions_CXXFLAGS = -I$(COMMON_DIR) @DYNINST_CXXFLAGS@ -I@DYNINST_INCLUDES@
if NEED_BOOST_HOME
list_functions_CXXFLAGS += -I@BOOST_HOME@/include
endif
list_functions_LDFLAGS = @DYNINST_LDFLAGS@ -R @DYNINST_LIBSDIR@ -ldyninstAPI @UNWIND_LDFLAGS@ @UNWIND_LIBS@ -lparseAPI -lsymtabAPI -linstructionAPI $(DYNINST_EXTRA_LIBS) -lcommon \
  @DWARF_LDFLAGS@ -R @DWARF_LIBSDIR@ -ldwarf \
  @ELF_LDFLAGS@ -R @ELF_LIBSDIR@ -lelf

extrae_test_dyninst_SOURCES = extrae-test-dyninst.C \
 cpp_utils.C cpp_utils.h \
 commonSnippets.C commonSnippets.h 
extrae_test_dyninst_CXXFLAGS = -I$(COMMON_DIR) @DYNINST_CXXFLAGS@ -I@DYNINST_INCLUDES@
if NEED_BOOST_HOME
extrae_test_dyninst_CXXFLAGS += -I@BOOST_HOME@/include
endif
extrae_test_dyninst_LDFLAGS = @DYNINST_LDFLAGS@ -R @DYNINST_LIBSDIR@ -ldyninstAPI @UNWIND_LDFLAGS@ @UNWIND_LIBS@ -lparseAPI -lsymtabAPI -linstructionAPI $(DYNINST_EXTRA_LIBS) -lcommon \
  @DWARF_LDFLAGS@ -R @DWARF_LIBSDIR@ -ldwarf \
  @ELF_LDFLAGS@ -R @ELF_LIBSDIR@ -lelf
