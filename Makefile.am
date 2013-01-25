
SUBDIRS = src include manuals scripts etc tests
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
	rm -fr `find $(distdir) -name .svn`

#
# Create EXAMPLE directory and copy there the example
#


if IS_CELL_MACHINE
SEQ_EXAMPLE_FILES=example/CELL/SEQ/worker.c \
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
ONLINE_EXAMPLE_FILES=/dev/null
else
if IS_BG_MACHINE
if IS_BGL_MACHINE
MPI_EXAMPLE_FILES= example/BGL/mpi_ping.f \
 example/BGL/Makefile \
 example/BGL/ll_merge.sh \
 example/BGL/ll_trace.sh \
 example/BGL/papi_bgl_counters.txt \
 example/BGL/papi_native_bgl_counters.txt
endif
if IS_BGP_MACHINE
MPI_EXAMPLE_FILES= example/BGP/ll_merge.sh \
 example/BGP/ll_parmerge.sh \
 example/BGP/Makefile \
 example/BGP/papi_bgp_avail.txt \
 example/BGP/ll_papi.sh \
 example/BGP/ll_trace.sh \
 example/BGP/mpi_ping.f \
 example/BGP/papi_bgp_native_avail.txt
endif
if IS_BGQ_MACHINE
MPI_EXAMPLE_FILES= example/BGQ/ll_merge.sh \
 example/BGQ/ll_parmerge.sh \
 example/BGQ/ll_trace.sh \
 example/BGQ/Makefile example/BGQ/mpi_ping.f \
 example/BGQ/papi_bgq_avail.txt \
 example/BGQ/papi_bgq_native_avail.txt \
 example/BGQ/extrae.xml

PTHREAD_EXAMPLE_FILES=/dev/null
OMP_EXAMPLE_FILES=/dev/null
SEQ_EXAMPLE_FILES=/dev/null
PACX_EXAMPLE_FILES=/dev/null
ONLINE_EXAMPLE_FILES=/dev/null
endif
else
if OS_AIX
MPI_EXAMPLE_FILES=example/AIX/Makefile \
 example/AIX/merge.sh \
 example/AIX/script.ll \
 example/AIX/mpi_ping.f \
 example/AIX/extrae_explained.xml \
 example/AIX/extrae.xml \
 example/AIX/detailed_trace_basic.xml \
 example/AIX/summarized_trace_basic.xml

PTHREAD_EXAMPLE_FILES=/dev/null
OMP_EXAMPLE_FILES=/dev/null
SEQ_EXAMPLE_FILES=/dev/null
PACX_EXAMPLE_FILES=/dev/null
ONLINE_EXAMPLE_FILES=/dev/null
else

if OS_FREEBSD

MPI_EXAMPLE_FILES=\
 example/FREEBSD/MPI/detailed_trace_basic.xml \
 example/FREEBSD/MPI/summarized_trace_basic.xml \
 example/FREEBSD/MPI/static/slurm_parallel_merge.sh \
 example/FREEBSD/MPI/static/slurm_merge.sh \
 example/FREEBSD/MPI/static/mpi_ping.f \
 example/FREEBSD/MPI/static/mpi2prv.sh \
 example/FREEBSD/MPI/static/Makefile \
 example/FREEBSD/MPI/static/trace.sh \
 example/FREEBSD/MPI/static/slurm_trace.sh \
 example/FREEBSD/MPI/extrae_explained.xml \
 example/FREEBSD/MPI/README \
 example/FREEBSD/MPI/extrae.xml \
 example/FREEBSD/MPI/ld-preload/slurm_parallel_merge.sh \
 example/FREEBSD/MPI/ld-preload/slurm_merge.sh \
 example/FREEBSD/MPI/ld-preload/mpi_ping.f \
 example/FREEBSD/MPI/ld-preload/mpi2prv.sh \
 example/FREEBSD/MPI/ld-preload/Makefile \
 example/FREEBSD/MPI/ld-preload/trace.sh \
 example/FREEBSD/MPI/ld-preload/slurm_trace.sh 

OMP_EXAMPLE_FILES=\
 example/FREEBSD/OMP/pi.c \
 example/FREEBSD/OMP/pi_instrumented.c \
 example/FREEBSD/OMP/README \
 example/FREEBSD/OMP/Makefile \
 example/FREEBSD/OMP/extrae.xml \
 example/FREEBSD/OMP/run_instrumented.sh \
 example/FREEBSD/OMP/run_dyninst.sh \
 example/FREEBSD/OMP/merge.sh

SEQ_EXAMPLE_FILES=\
 example/FREEBSD/SEQ/pi.c \
 example/FREEBSD/SEQ/pi_instrumented.c \
 example/FREEBSD/SEQ/README \
 example/FREEBSD/SEQ/Makefile \
 example/FREEBSD/SEQ/extrae.xml \
 example/FREEBSD/SEQ/run_instrumented.sh \
 example/FREEBSD/SEQ/function-list \
 example/FREEBSD/SEQ/run_dyninst.sh

PTHREAD_EXAMPLE_FILES=\
 example/FREEBSD/PTHREAD/pthread-example.c \
 example/FREEBSD/PTHREAD/README \
 example/FREEBSD/PTHREAD/Makefile \
 example/FREEBSD/PTHREAD/extrae.xml \
 example/FREEBSD/PTHREAD/user-comms.c

else

if OS_DARWIN

MPI_EXAMPLE_FILES=\
 example/FREEBSD/MPI/detailed_trace_basic.xml \
 example/FREEBSD/MPI/summarized_trace_basic.xml \
 example/FREEBSD/MPI/static/slurm_parallel_merge.sh \
 example/FREEBSD/MPI/static/slurm_merge.sh \
 example/FREEBSD/MPI/static/mpi_ping.f \
 example/FREEBSD/MPI/static/mpi2prv.sh \
 example/FREEBSD/MPI/static/Makefile \
 example/FREEBSD/MPI/static/trace.sh \
 example/FREEBSD/MPI/static/slurm_trace.sh \
 example/FREEBSD/MPI/extrae_explained.xml \
 example/FREEBSD/MPI/README \
 example/FREEBSD/MPI/extrae.xml \
 example/FREEBSD/MPI/ld-preload/slurm_parallel_merge.sh \
 example/FREEBSD/MPI/ld-preload/slurm_merge.sh \
 example/FREEBSD/MPI/ld-preload/mpi_ping.f \
 example/FREEBSD/MPI/ld-preload/mpi2prv.sh \
 example/FREEBSD/MPI/ld-preload/Makefile \
 example/FREEBSD/MPI/ld-preload/trace.sh \
 example/FREEBSD/MPI/ld-preload/slurm_trace.sh 

OMP_EXAMPLE_FILES=\
 example/FREEBSD/OMP/pi.c \
 example/FREEBSD/OMP/pi_instrumented.c \
 example/FREEBSD/OMP/README \
 example/FREEBSD/OMP/Makefile \
 example/FREEBSD/OMP/extrae.xml \
 example/FREEBSD/OMP/run_instrumented.sh \
 example/FREEBSD/OMP/run_dyninst.sh \
 example/FREEBSD/OMP/merge.sh

SEQ_EXAMPLE_FILES=\
 example/FREEBSD/SEQ/pi.c \
 example/FREEBSD/SEQ/pi_instrumented.c \
 example/FREEBSD/SEQ/README \
 example/FREEBSD/SEQ/Makefile \
 example/FREEBSD/SEQ/extrae.xml \
 example/FREEBSD/SEQ/run_instrumented.sh \
 example/FREEBSD/SEQ/function-list \
 example/FREEBSD/SEQ/run_dyninst.sh

PTHREAD_EXAMPLE_FILES=\
 example/FREEBSD/PTHREAD/pthread-example.c \
 example/FREEBSD/PTHREAD/README \
 example/FREEBSD/PTHREAD/Makefile \
 example/FREEBSD/PTHREAD/extrae.xml \
 example/FREEBSD/PTHREAD/user-comms.c

else 

MPI_EXAMPLE_FILES=\
 example/LINUX/MPI/README \
 example/LINUX/MPI/extrae.xml \
 example/LINUX/MPI/detailed_trace_basic.xml \
 example/LINUX/MPI/summarized_trace_basic.xml \
 example/LINUX/MPI/extrae_explained.xml \
 example/LINUX/MPI/static \
 example/LINUX/MPI/ld-preload \
 example/LINUX/MPI/dyninst

MPICUDA_EXAMPLE_FILES=\
 example/LINUX/MPI+CUDA/ld-preload \
 example/LINUX/MPI+CUDA/dyninst \
 example/LINUX/MPI+CUDA/extrae.xml

OMP_EXAMPLE_FILES=\
 example/LINUX/OMP/extrae.xml \
 example/LINUX/OMP/README \
 example/LINUX/OMP/Makefile \
 example/LINUX/OMP/pi.c \
 example/LINUX/OMP/pi_instrumented.c \
 example/LINUX/OMP/run_dyninst.sh \
 example/LINUX/OMP/run_instrumented.sh

SEQ_EXAMPLE_FILES=\
 example/LINUX/SEQ/extrae.xml \
 example/LINUX/SEQ/function-list \
 example/LINUX/SEQ/Makefile \
 example/LINUX/SEQ/pi.c \
 example/LINUX/SEQ/pi_instrumented.c \
 example/LINUX/SEQ/run_dyninst.sh \
 example/LINUX/SEQ/run_instrumented.sh \
 example/LINUX/SEQ/README

ONLINE_EXAMPLE_FILES=\
 example/LINUX/ONLINE/Makefile \
 example/LINUX/ONLINE/merge.sh \
 example/LINUX/ONLINE/test.c \
 example/LINUX/ONLINE/extrae.xml \
 example/LINUX/ONLINE/slurm_trace.sh \
 example/LINUX/ONLINE/trace.sh \
 example/LINUX/ONLINE/README

PTHREAD_EXAMPLE_FILES=\
 example/LINUX/PTHREAD/Makefile \
 example/LINUX/PTHREAD/pthread-example.c \
 example/LINUX/PTHREAD/user-comms.c \
 example/LINUX/PTHREAD/extrae.xml \
 example/LINUX/PTHREAD/README

PACX_EXAMPLE_FILES=\
 example/LINUX/PACX/example.c \
 example/LINUX/PACX/Makefile

CUDA_EXAMPLE_FILES=\
 example/LINUX/CUDA/extrae.xml \
 example/LINUX/CUDA/hello.cu \
 example/LINUX/CUDA/hello_instrumented.cu \
 example/LINUX/CUDA/Makefile \
 example/LINUX/CUDA/README \
 example/LINUX/CUDA/run_dyninst.sh \
 example/LINUX/CUDA/run_instrumented.sh

endif
endif

endif
endif
endif

UNICORE_EXAMPLE_FILES= example/UNICORE/*

install-data-hook:
	$(mkdir_p) $(DESTDIR)$(datadir)/example
	cp -f example/Makefile.inc $(DESTDIR)$(datadir)/example
	$(mkdir_p) $(DESTDIR)$(datadir)/example/SEQ
	cp -f $(SEQ_EXAMPLE_FILES) $(DESTDIR)$(datadir)/example/SEQ
if HAVE_PACX
	$(mkdir_p) $(DESTDIR)$(datadir)/example/PACX
	cp -f $(PACX_EXAMPLE_FILES) $(DESTDIR)$(datadir)/example/PACX
endif
if HAVE_MPI
	$(mkdir_p) $(DESTDIR)$(datadir)/example/MPI
	cp -fr $(MPI_EXAMPLE_FILES) $(DESTDIR)$(datadir)/example/MPI
if WANT_CUDAorCUPTI
	$(mkdir_p) $(DESTDIR)$(datadir)/example/MPI+CUDA
	cp -fr $(MPICUDA_EXAMPLE_FILES) $(DESTDIR)$(datadir)/example/MPI+CUDA
endif
endif
if WANT_OPENMP
	$(mkdir_p) $(DESTDIR)$(datadir)/example/OMP
	cp -f $(OMP_EXAMPLE_FILES) $(DESTDIR)$(datadir)/example/OMP
endif
if HAVE_ONLINE
	$(mkdir_p) $(DESTDIR)$(datadir)/example/ONLINE
	cp -f $(ONLINE_EXAMPLE_FILES) $(DESTDIR)$(datadir)/example/ONLINE
endif
if WANT_PTHREAD
	$(mkdir_p) $(DESTDIR)$(datadir)/example/PTHREAD
	cp -f $(PTHREAD_EXAMPLE_FILES) $(DESTDIR)$(datadir)/example/PTHREAD
endif
if WANT_CUDAorCUPTI
	$(mkdir_p) $(DESTDIR)$(datadir)/example/CUDA
	cp -f $(CUDA_EXAMPLE_FILES) $(DESTDIR)$(datadir)/example/CUDA
endif
	$(mkdir_p) $(DESTDIR)$(datadir)/example/UNICORE
	cp -fr $(UNICORE_EXAMPLE_FILES) $(DESTDIR)$(datadir)/example/UNICORE
# Apply post-installation modifications
	./substitute $(SED) "@sub_MPI_HOME@" "${MPI_HOME}" $(DESTDIR)$(datadir)/example/Makefile.inc
	./substitute $(SED) "@sub_PACX_HOME@" "${PACX_HOME}" $(DESTDIR)$(datadir)/example/Makefile.inc
	./substitute $(SED) "@sub_PAPI_HOME@" "${PAPI_HOME}" $(DESTDIR)$(datadir)/example/Makefile.inc
	./substitute $(SED) "@sub_PAPI_LIBS@" "${PAPI_LIBS}" $(DESTDIR)$(datadir)/example/Makefile.inc
	./substitute $(SED) "@sub_UNWIND_HOME@" "${UNWIND_HOME}" $(DESTDIR)$(datadir)/example/Makefile.inc
	./substitute $(SED) "@sub_UNWIND_LIBS@" "${UNWIND_LIBS}" $(DESTDIR)$(datadir)/example/Makefile.inc
	./substitute $(SED) "@sub_XML2_HOME@" "${XML2_HOME}" $(DESTDIR)$(datadir)/example/Makefile.inc
	./substitute $(SED) "@sub_XML2_LDFLAGS@" "${XML2_LDFLAGS}" $(DESTDIR)$(datadir)/example/Makefile.inc
	./substitute $(SED) "@sub_XML2_LIBS@" "${XML2_LIBS}" $(DESTDIR)$(datadir)/example/Makefile.inc
	./substitute $(SED) "@sub_BFD_HOME@" "${BFD_HOME}" $(DESTDIR)$(datadir)/example/Makefile.inc
	./substitute $(SED) "@sub_LIBERTY_HOME@" "${LIBERTY_HOME}" $(DESTDIR)$(datadir)/example/Makefile.inc
	./substitute $(SED) "@sub_BFD_LIBS@" "${BFD_LDFLAGS} ${BFD_LIBS}" $(DESTDIR)$(datadir)/example/Makefile.inc
	./substitute $(SED) "@sub_LIBERTY_LIBS@" "${LIBERTY_LDFLAGS} ${LIBERTY_LIBS}" $(DESTDIR)$(datadir)/example/Makefile.inc
	./substitute $(SED) "@sub_ZLIB_HOME@" "${LIBZ_HOME}" $(DESTDIR)$(datadir)/example/Makefile.inc
	./substitute $(SED) "@sub_ZLIB_LIBS@" "${LIBZ_LIBS}" $(DESTDIR)$(datadir)/example/Makefile.inc
if WANT_CUPTI
	./substitute $(SED) "@sub_CUPTI_LIBS@" "-L${CUPTI_HOME}/lib -lcupti" $(DESTDIR)$(datadir)/example/Makefile.inc
else
	./substitute $(SED) "@sub_CUPTI_LIBS@" "" $(DESTDIR)$(datadir)/example/Makefile.inc
endif
	./substitute $(SED) "@sub_POSIX_CLOCK@" "${POSIX_CLOCK_LIB}" $(DESTDIR)$(datadir)/example/Makefile.inc
	./substitute-all $(SED) "@sub_PREFIXDIR@" "${prefix}" $(DESTDIR)$(datadir)/example
	./substitute-all $(SED) "@sub_XMLID@" "`grep 'rcsid\[\]' src/tracer/xml-parse.c | $(AWK) -F'$$' '{ print $$2"$$" }'`" $(DESTDIR)$(datadir)/example
# Apply post-installation modifications (FreeBSD specific)
if OS_FREEBSD
	./substitute $(SED) "@sub_libexecinfo_libs@" "${libexecinfo_LDFLAGS} -lexecinfo" $(DESTDIR)$(datadir)/example/Makefile.inc
else
	./substitute $(SED) "@sub_libexecinfo_libs@" "" $(DESTDIR)$(datadir)/example/Makefile.inc
endif
# SVN installations may have some subversion control files
	rm -fr `find $(DESTDIR)$(datadir)/example -name .svn`
