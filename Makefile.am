
SUBDIRS = src include manuals scripts etc
if GENERATE_DOCUMENTATION
SUBDIRS += doc
endif

EXTRA_DIST = example \
  INSTALL-examples \
	substitute \
	substitute-all

VPATH=@srcdir@
SRCDIR=@srcdir@

@SET_MAKE@

dist-hook:
	rm -rf `find $(distdir)/example -name .svn`

#
# Create EXAMPLE directory and copy there the example
# Create ETC directory and create an empty license file there
#


if IS_CELL_MACHINE
SEQ_EXAMPLE_FILES=example/CELL/SEQ/worker.c \
 example/CELL/SEQ/spu.h \
 example/CELL/SEQ/run.sh \
 example/CELL/SEQ/merge.sh \
 example/CELL/SEQ/master.c \
 example/CELL/SEQ/mapa.pnm \
 example/CELL/SEQ/chroma.pnm \
 example/CELL/SEQ/bitmap.c \
 example/CELL/SEQ/bitmap.h \
 example/CELL/SEQ/README \
 example/CELL/SEQ/Makefile

MPI_EXAMPLE_FILES=example/CELL/MPI/worker.c \
 example/CELL/MPI/spu.h \
 example/CELL/MPI/run.sh \
 example/CELL/MPI/merge.sh \
 example/CELL/MPI/master.c \
 example/CELL/MPI/mapa.pnm \
 example/CELL/MPI/chroma.pnm \
 example/CELL/MPI/bitmap.c \
 example/CELL/MPI/bitmap.h \
 example/CELL/MPI/README \
 example/CELL/MPI/trace-xml.sh \
 example/CELL/MPI/Makefile

PTHREAD_EXAMPLE_FILES=/dev/null
OMP_EXAMPLE_FILES=/dev/null
PACX_EXAMPLE_FILES=/dev/null
MRNET_EXAMPLE_FILES=/dev/null
else
if IS_BG_MACHINE
if IS_BGL_MACHINE
MPI_EXAMPLE_FILES= example/BGL/mpi_ping.f \
 example/BGL/Makefile \
 example/BGL/ll_merge.sh \
 example/BGL/ll_trace.sh \
 example/BGL/papi_bgl_counters.txt \
 example/BGL/papi_native_bgl_counters.txt

PTHREAD_EXAMPLE_FILES=/dev/null
OMP_EXAMPLE_FILES=/dev/null
SEQ_EXAMPLE_FILES=/dev/null
PACX_EXAMPLE_FILES=/dev/null
MRNET_EXAMPLE_FILES=/dev/null
else
if IS_BGP_MACHINE
MPI_EXAMPLE_FILES= example/BGP/ll_merge.sh \
 example/BGP/ll_parmerge.sh \
 example/BGP/Makefile \
 example/BGP/papi_bgp_avail.txt \
 example/BGP/ll_papi.sh \
 example/BGP/ll_trace.sh \
 example/BGP/mpi_ping.f \
 example/BGP/papi_bgp_native_avail.txt

PTHREAD_EXAMPLE_FILES=/dev/null
OMP_EXAMPLE_FILES=/dev/null
SEQ_EXAMPLE_FILES=/dev/null
PACX_EXAMPLE_FILES=/dev/null
MRNET_EXAMPLE_FILES=/dev/null
endif
endif
else
if OS_AIX
MPI_EXAMPLE_FILES=example/AIX/Makefile \
 example/AIX/merge.sh \
 example/AIX/script.ll \
 example/AIX/mpi_ping.f \
 example/AIX/mpitrace_explained.xml \
 example/AIX/mpitrace.xml \
 example/AIX/detailed_trace_basic.xml \
 example/AIX/summarized_trace_basic.xml

PTHREAD_EXAMPLE_FILES=/dev/null
OMP_EXAMPLE_FILES=/dev/null
SEQ_EXAMPLE_FILES=/dev/null
PACX_EXAMPLE_FILES=/dev/null
MRNET_EXAMPLE_FILES=/dev/null
else
MPI_EXAMPLE_FILES=\
 example/LINUX/MPI/run.sh \
 example/LINUX/MPI/run-dynamic.sh \
 example/LINUX/MPI/trace.sh \
 example/LINUX/MPI/mpi_ping.f \
 example/LINUX/MPI/mpi2prv.sh \
 example/LINUX/MPI/Makefile \
 example/LINUX/MPI/slurm_trace.sh \
 example/LINUX/MPI/slurm_merge.sh \
 example/LINUX/MPI/slurm_parallel_merge.sh \
 example/LINUX/MPI/mpitrace_explained.xml \
 example/LINUX/MPI/mpitrace.xml \
 example/LINUX/MPI/detailed_trace_basic.xml \
 example/LINUX/MPI/summarized_trace_basic.xml 

OMP_EXAMPLE_FILES=\
 example/LINUX/OMP/Makefile \
 example/LINUX/OMP/merge.sh \
 example/LINUX/OMP/pi.c \
 example/LINUX/OMP/pi.f \
 example/LINUX/OMP/run.sh \
 example/LINUX/OMP/mpitrace.xml

SEQ_EXAMPLE_FILES=\
 example/LINUX/SEQ/Makefile \
 example/LINUX/SEQ/merge.sh \
 example/LINUX/SEQ/pi.c \
 example/LINUX/SEQ/run.sh \
 example/LINUX/SEQ/mpitrace.xml

MRNET_EXAMPLE_FILES=\
 example/LINUX/MRNET/Makefile \
 example/LINUX/MRNET/merge.sh \
 example/LINUX/MRNET/test.c \
 example/LINUX/MRNET/mpitrace.xml \
 example/LINUX/MRNET/cl.I.IPC.xml \
 example/LINUX/MRNET/run.cmd \
 example/LINUX/MRNET/trace.sh \
 example/LINUX/MRNET/README

PTHREAD_EXAMPLE_FILES=\
 example/LINUX/PTHREAD/Makefile \
 example/LINUX/PTHREAD/pthread-example.c \
 example/LINUX/PTHREAD/user-comms.c \
 example/LINUX/PTHREAD/README

PACX_EXAMPLE_FILES=example/LINUX/PACX/example.c \
 example/LINUX/PACX/Makefile
endif
endif
endif

install-data-hook:
	$(mkdir_p) $(datadir)/example
	$(mkdir_p) $(datadir)/example/SEQ
	cp -f $(SEQ_EXAMPLE_FILES) $(datadir)/example/SEQ
	./substitute-all $(SED) "@sub_PREFIXDIR@" "${prefix}" $(datadir)/example/SEQ
	./substitute-all $(SED) "@sub_XMLID@" "`grep 'rcsid\[\]' src/tracer/xml-parse.c | $(AWK) -F'$$' '{ print $$2"$$" }'`" $(datadir)/example/SEQ
	./substitute-all $(SED) "@sub_PAPI_HOME@" "${PAPI_HOME}" $(datadir)/example/SEQ
	./substitute-all $(SED) "@sub_PAPI_LIBS@" "${PAPI_LIBS}" $(datadir)/example/SEQ
if HAVE_PACX
	$(mkdir_p) $(datadir)/example/PACX
	cp -f $(PACX_EXAMPLE_FILES) $(datadir)/example/PACX
	./substitute-all $(SED) "@sub_PREFIXDIR@" "${prefix}" $(datadir)/example/PACX
	./substitute-all $(SED) "@sub_XMLID@" "`grep 'rcsid\[\]' src/tracer/xml-parse.c | $(AWK) -F'$$' '{ print $$2"$$" }'`" $(datadir)/example/PACX
	./substitute-all $(SED) "@sub_PACX_HOME@" "${PACX_HOME}" $(datadir)/example/PACX
	./substitute-all $(SED) "@sub_PAPI_HOME@" "${PAPI_HOME}" $(datadir)/example/PACX
	./substitute-all $(SED) "@sub_PAPI_LIBS@" "${PAPI_LIBS}" $(datadir)/example/PACX
endif
if HAVE_MPI
	$(mkdir_p) $(datadir)/example/MPI
	cp -f $(MPI_EXAMPLE_FILES) $(datadir)/example/MPI
	./substitute-all $(SED) "@sub_PREFIXDIR@" "${prefix}" $(datadir)/example/MPI
	./substitute-all $(SED) "@sub_XMLID@" "`grep 'rcsid\[\]' src/tracer/xml-parse.c | $(AWK) -F'$$' '{ print $$2"$$" }'`" $(datadir)/example/MPI
	./substitute-all $(SED) "@sub_MPI_HOME@" "${MPI_HOME}" $(datadir)/example/MPI
	./substitute-all $(SED) "@sub_PAPI_HOME@" "${PAPI_HOME}" $(datadir)/example/MPI
	./substitute-all $(SED) "@sub_PAPI_LIBS@" "${PAPI_LIBS}" $(datadir)/example/MPI
endif
if WANT_OPENMP
	$(mkdir_p) $(datadir)/example/OMP
	cp -f $(OMP_EXAMPLE_FILES) $(datadir)/example/OMP
	./substitute-all $(SED) "@sub_PREFIXDIR@" "${prefix}" $(datadir)/example/OMP
	./substitute-all $(SED) "@sub_XMLID@" "`grep 'rcsid\[\]' src/tracer/xml-parse.c | $(AWK) -F'$$' '{ print $$2"$$" }'`" $(datadir)/example/OMP
	./substitute-all $(SED) "@sub_PAPI_HOME@" "${PAPI_HOME}" $(datadir)/example/OMP
	./substitute-all $(SED) "@sub_PAPI_LIBS@" "${PAPI_LIBS}" $(datadir)/example/OMP
endif
if HAVE_MRNET
	$(mkdir_p) $(datadir)/example/MRNET
	cp -f $(MRNET_EXAMPLE_FILES) $(datadir)/example/MRNET
	chmod u+x $(datadir)/example/MRNET/trace.sh $(datadir)/example/MRNET/merge.sh
	./substitute-all $(SED) "@sub_PREFIXDIR@" "${prefix}" $(datadir)/example/MRNET
	./substitute-all $(SED) "@sub_XMLID@" "`grep 'rcsid\[\]' src/tracer/xml-parse.c | $(AWK) -F'$$' '{ print $$2"$$" }'`" $(datadir)/example/MRNET
	./substitute-all $(SED) "@sub_PAPI_HOME@" "${PAPI_HOME}" $(datadir)/example/MRNET
	./substitute-all $(SED) "@sub_PAPI_LIBS@" "${PAPI_LIBS}" $(datadir)/example/MRNET
endif
if WANT_PTHREAD
	$(mkdir_p) $(datadir)/example/PTHREAD
	cp -f $(PTHREAD_EXAMPLE_FILES) $(datadir)/example/PTHREAD
	./substitute-all $(SED) "@sub_PREFIXDIR@" "${prefix}" $(datadir)/example/PTHREAD
	./substitute-all $(SED) "@sub_XMLID@" "`grep 'rcsid\[\]' src/tracer/xml-parse.c | $(AWK) -F'$$' '{ print $$2"$$" }'`" $(datadir)/example/PTHREAD
	./substitute-all $(SED) "@sub_PAPI_HOME@" "${PAPI_HOME}" $(datadir)/example/PTHREAD
	./substitute-all $(SED) "@sub_PAPI_LIBS@" "${PAPI_LIBS}" $(datadir)/example/PTHREAD
endif

uninstall-local:
	rm -f $(sysconfdir)/license.dat
