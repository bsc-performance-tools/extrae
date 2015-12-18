include $(top_srcdir)/PATHS

lib_LTLIBRARIES = \
  libextrae-jvmti-agent.la

libextrae_jvmti_agent_la_SOURCES = extrae-jvmti-agent.c
libextrae_jvmti_agent_la_CFLAGS  = $(JAVA_INCLUDES)\
 -Ijni/include \
 -I$(INCLUDE_DIR) \
 -I$(COMMON_INC) \
 -I$(WRAPPERS_DIR)/JAVA \
 -I$(WRAPPERS_DIR)/API \
 -I$(TRACER_DIR) \
 -I$(COMMON_DIR) \
 $(CFLAGS)
libextrae_jvmti_agent_la_LIBADD  = $(TRACER_LIB)/libpttrace.la
libextrae_jvmti_agent_la_LDFLAGS = -release $(PACKAGE_VERSION)
