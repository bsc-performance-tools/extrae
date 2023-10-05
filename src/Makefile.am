
SUBDIRS = loader launcher common merger tracer others 

if WANT_JAVA
SUBDIRS += java-connector
endif
if !IS_BG_MACHINE
SUBDIRS += cmd-line
endif

if HAVE_PAPI
SUBDIRS += extrae-uncore
endif

