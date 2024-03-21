include $(top_srcdir)/PATHS

EXTRA_DIST = test.C

bin_PROGRAMS = extrae extrae-test-dyninst

noinst_PROGRAMS = load-module demo-launcher list-functions

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
extrae_CXXFLAGS = -I$(COMMON_INC) @DYNINST_CXXFLAGS@ @BOOST_CXXFLAGS@
extrae_CFLAGS = -I$(COMMON_INC) @DYNINST_CFLAGS@ @XML2_CFLAGS@
extrae_LDFLAGS =  @DYNINST_LDFLAGS@ @DYNINST_RPATH@ @DYNINST_LIBS@ \
  @UNWIND_LDFLAGS@ @UNWIND_RPATH@ \
  @XML2_LDFLAGS@ @XML2_RPATH@ 

load_module_SOURCES = load-module.C
load_module_CXXFLAGS = -I$(COMMON_INC) @DYNINST_CXXFLAGS@ -I@DYNINST_INCLUDES@
load_module_CXXFLAGS += -I@BOOST_HOME@/include
load_module_LDFLAGS = @DYNINST_LDFLAGS@ -R @DYNINST_LIBSDIR@ @UNWIND_LDFLAGS@ @UNWIND_LIBS@ \
  @XML2_LDFLAGS@ @XML2_LIBS@

demo_launcher_SOURCES = demo-launcher.C
demo_launcher_CXXFLAGS = -I$(COMMON_INC) @DYNINST_CXXFLAGS@ -I@DYNINST_INCLUDES@
demo_launcher_CXXFLAGS += -I@BOOST_HOME@/include
demo_launcher_LDFLAGS = @DYNINST_LDFLAGS@ -R @DYNINST_LIBSDIR@ @UNWIND_LDFLAGS@ @UNWIND_LIBS@ \
  @XML2_LDFLAGS@ @XML2_LIBS@

list_functions_SOURCES = list-functions.C
list_functions_CXXFLAGS = -I$(COMMON_DIR) @DYNINST_CXXFLAGS@ -I@DYNINST_INCLUDES@
list_functions_CXXFLAGS += -I@BOOST_HOME@/include
list_functions_LDFLAGS = @DYNINST_LDFLAGS@ -R @DYNINST_LIBSDIR@ @UNWIND_LDFLAGS@ @UNWIND_LIBS@ 

extrae_test_dyninst_SOURCES = extrae-test-dyninst.C \
 cpp_utils.C cpp_utils.h \
 commonSnippets.C commonSnippets.h 
extrae_test_dyninst_CXXFLAGS = -I$(COMMON_DIR) @DYNINST_CXXFLAGS@ -I@DYNINST_INCLUDES@
extrae_test_dyninst_CXXFLAGS += -I@BOOST_HOME@/include
extrae_test_dyninst_LDFLAGS = @DYNINST_LDFLAGS@ -R @DYNINST_LIBSDIR@ @UNWIND_LDFLAGS@ @UNWIND_LIBS@ 
