
SUBDIRS = src include scripts etc tests
if GENERATE_DOCUMENTATION
SUBDIRS += docs
endif

EXTRA_DIST = \
  bootstrap \
  ChangeLog \
  example \
  INSTALL-examples \
  substitute \
  substitute-all

VPATH=@srcdir@
SRCDIR=@srcdir@

ACLOCAL_AMFLAGS=-I config

@SET_MAKE@

dist-hook:
	rm -fr `find $(distdir) -name .svn`

#
# Create EXAMPLE directory and copy there the example
#
MPI_EXAMPLE_FILES=null
MPICUDA_EXAMPLE_FILES=null
OMP_EXAMPLE_FILES=null
MPIOMP_EXAMPLE_FILES=null
SEQ_EXAMPLE_FILES=null
ONLINE_EXAMPLE_FILES=null
PYTHON_EXAMPLE_FILES=null
PTHREAD_EXAMPLE_FILES=null
CUDA_EXAMPLE_FILES=null
OPENCL_EXAMPLE_FILES=null
MPIOPENCL_EXAMPLE_FILES=null
OMPSS_EXAMPLE_FILES=null
MPIOMPSS_EXAMPLE_FILES=null
JAVA_EXAMPLE_FILES=null
FORK_EXAMPLE_FILES=null
UNICORE_EXAMPLE_FILES= $(top_srcdir)/example/UNICORE/*

if IS_BG_MACHINE
if IS_BGL_MACHINE
MPI_EXAMPLE_FILES+=\
 $(top_srcdir)/example/BGL/mpi_ping.f \
 $(top_srcdir)/example/BGL/Makefile \
 $(top_srcdir)/example/BGL/ll_merge.sh \
 $(top_srcdir)/example/BGL/ll_trace.sh \
 $(top_srcdir)/example/BGL/papi_bgl_counters.txt \
 $(top_srcdir)/example/BGL/papi_native_bgl_counters.txt
endif
if IS_BGP_MACHINE
MPI_EXAMPLE_FILES+=\
 $(top_srcdir)/example/BGP/ll_merge.sh \
 $(top_srcdir)/example/BGP/ll_parmerge.sh \
 $(top_srcdir)/example/BGP/Makefile \
 $(top_srcdir)/example/BGP/ll_papi.sh \
 $(top_srcdir)/example/BGP/ll_trace.sh \
 $(top_srcdir)/example/BGP/mpi_ping.f \
 $(top_srcdir)/example/BGP/papi_bgp_avail.txt \
 $(top_srcdir)/example/BGP/papi_bgp_native_avail.txt
endif
if IS_BGQ_MACHINE
MPI_EXAMPLE_FILES+=\
 $(top_srcdir)/example/BGQ/ll_merge.sh \
 $(top_srcdir)/example/BGQ/ll_parmerge.sh \
 $(top_srcdir)/example/BGQ/ll_trace.sh \
 $(top_srcdir)/example/BGQ/Makefile \
 $(top_srcdir)/example/BGQ/mpi_ping.f \
 $(top_srcdir)/example/BGQ/papi_bgq_avail.txt \
 $(top_srcdir)/example/BGQ/papi_bgq_native_avail.txt \
 $(top_srcdir)/example/BGQ/extrae.xml
endif
endif

if OS_AIX
MPI_EXAMPLE_FILES+=\
 $(top_srcdir)/example/AIX/Makefile \
 $(top_srcdir)/example/AIX/merge.sh \
 $(top_srcdir)/example/AIX/script.ll \
 $(top_srcdir)/example/AIX/mpi_ping.f \
 $(top_srcdir)/example/AIX/extrae_explained.xml \
 $(top_srcdir)/example/AIX/extrae.xml \
 $(top_srcdir)/example/AIX/detailed_trace_basic.xml \
 $(top_srcdir)/example/AIX/summarized_trace_basic.xml
endif

if OS_FREEBSD
MPI_EXAMPLE_FILES+=\
 $(top_srcdir)/example/FREEBSD/MPI/detailed_trace_basic.xml \
 $(top_srcdir)/example/FREEBSD/MPI/summarized_trace_basic.xml \
 $(top_srcdir)/example/FREEBSD/MPI/static/slurm_parallel_merge.sh \
 $(top_srcdir)/example/FREEBSD/MPI/static/slurm_merge.sh \
 $(top_srcdir)/example/FREEBSD/MPI/static/mpi_ping.f \
 $(top_srcdir)/example/FREEBSD/MPI/static/mpi2prv.sh \
 $(top_srcdir)/example/FREEBSD/MPI/static/Makefile \
 $(top_srcdir)/example/FREEBSD/MPI/static/trace.sh \
 $(top_srcdir)/example/FREEBSD/MPI/static/slurm_trace.sh \
 $(top_srcdir)/example/FREEBSD/MPI/extrae_explained.xml \
 $(top_srcdir)/example/FREEBSD/MPI/README \
 $(top_srcdir)/example/FREEBSD/MPI/extrae.xml \
 $(top_srcdir)/example/FREEBSD/MPI/ld-preload/slurm_parallel_merge.sh \
 $(top_srcdir)/example/FREEBSD/MPI/ld-preload/slurm_merge.sh \
 $(top_srcdir)/example/FREEBSD/MPI/ld-preload/mpi_ping.f \
 $(top_srcdir)/example/FREEBSD/MPI/ld-preload/mpi2prv.sh \
 $(top_srcdir)/example/FREEBSD/MPI/ld-preload/Makefile \
 $(top_srcdir)/example/FREEBSD/MPI/ld-preload/trace.sh \
 $(top_srcdir)/example/FREEBSD/MPI/ld-preload/slurm_trace.sh 

OMP_EXAMPLE_FILES+=\
 $(top_srcdir)/example/FREEBSD/OMP/pi.c \
 $(top_srcdir)/example/FREEBSD/OMP/pi_instrumented.c \
 $(top_srcdir)/example/FREEBSD/OMP/README \
 $(top_srcdir)/example/FREEBSD/OMP/Makefile \
 $(top_srcdir)/example/FREEBSD/OMP/extrae.xml \
 $(top_srcdir)/example/FREEBSD/OMP/run_instrumented.sh \
 $(top_srcdir)/example/FREEBSD/OMP/run_dyninst.sh \
 $(top_srcdir)/example/FREEBSD/OMP/merge.sh

SEQ_EXAMPLE_FILES+=\
 $(top_srcdir)/example/FREEBSD/SEQ/pi.c \
 $(top_srcdir)/example/FREEBSD/SEQ/pi_instrumented.c \
 $(top_srcdir)/example/FREEBSD/SEQ/pi_instrumented.f90 \
 $(top_srcdir)/example/FREEBSD/SEQ/README \
 $(top_srcdir)/example/FREEBSD/SEQ/Makefile \
 $(top_srcdir)/example/FREEBSD/SEQ/extrae.xml \
 $(top_srcdir)/example/FREEBSD/SEQ/run_instrumented.sh \
 $(top_srcdir)/example/FREEBSD/SEQ/run_instrumentedf.sh \
 $(top_srcdir)/example/FREEBSD/SEQ/function-list \
 $(top_srcdir)/example/FREEBSD/SEQ/run_dyninst.sh

PTHREAD_EXAMPLE_FILES+=\
 $(top_srcdir)/example/FREEBSD/PTHREAD/pthread-example.c \
 $(top_srcdir)/example/FREEBSD/PTHREAD/README \
 $(top_srcdir)/example/FREEBSD/PTHREAD/Makefile \
 $(top_srcdir)/example/FREEBSD/PTHREAD/extrae.xml \
 $(top_srcdir)/example/FREEBSD/PTHREAD/user-comms.c

JAVA_EXAMPLE_FILES+=\
 $(top_srcdir)/example/FREEBSD/JAVA/manual

if WANT_JAVA_WITH_ASPECTJ
JAVA_EXAMPLE_FILES+=\
 $(top_srcdir)/example/FREEBSD/JAVA/automatic 
endif
endif

if OS_DARWIN
MPI_EXAMPLE_FILES+=\
 $(top_srcdir)/example/FREEBSD/MPI/detailed_trace_basic.xml \
 $(top_srcdir)/example/FREEBSD/MPI/summarized_trace_basic.xml \
 $(top_srcdir)/example/FREEBSD/MPI/static/slurm_parallel_merge.sh \
 $(top_srcdir)/example/FREEBSD/MPI/static/slurm_merge.sh \
 $(top_srcdir)/example/FREEBSD/MPI/static/mpi_ping.f \
 $(top_srcdir)/example/FREEBSD/MPI/static/mpi2prv.sh \
 $(top_srcdir)/example/FREEBSD/MPI/static/Makefile \
 $(top_srcdir)/example/FREEBSD/MPI/static/trace.sh \
 $(top_srcdir)/example/FREEBSD/MPI/static/slurm_trace.sh \
 $(top_srcdir)/example/FREEBSD/MPI/extrae_explained.xml \
 $(top_srcdir)/example/FREEBSD/MPI/README \
 $(top_srcdir)/example/FREEBSD/MPI/extrae.xml \
 $(top_srcdir)/example/FREEBSD/MPI/ld-preload/slurm_parallel_merge.sh \
 $(top_srcdir)/example/FREEBSD/MPI/ld-preload/slurm_merge.sh \
 $(top_srcdir)/example/FREEBSD/MPI/ld-preload/mpi_ping.f \
 $(top_srcdir)/example/FREEBSD/MPI/ld-preload/mpi2prv.sh \
 $(top_srcdir)/example/FREEBSD/MPI/ld-preload/Makefile \
 $(top_srcdir)/example/FREEBSD/MPI/ld-preload/trace.sh \
 $(top_srcdir)/example/FREEBSD/MPI/ld-preload/slurm_trace.sh 

OMP_EXAMPLE_FILES+=\
 $(top_srcdir)/example/FREEBSD/OMP/pi.c \
 $(top_srcdir)/example/FREEBSD/OMP/pi_instrumented.c \
 $(top_srcdir)/example/FREEBSD/OMP/README \
 $(top_srcdir)/example/FREEBSD/OMP/Makefile \
 $(top_srcdir)/example/FREEBSD/OMP/extrae.xml \
 $(top_srcdir)/example/FREEBSD/OMP/run_instrumented.sh \
 $(top_srcdir)/example/FREEBSD/OMP/run_dyninst.sh \
 $(top_srcdir)/example/FREEBSD/OMP/merge.sh

SEQ_EXAMPLE_FILES+=\
 $(top_srcdir)/example/FREEBSD/SEQ/pi.c \
 $(top_srcdir)/example/FREEBSD/SEQ/pi_instrumented.c \
 $(top_srcdir)/example/FREEBSD/SEQ/README \
 $(top_srcdir)/example/FREEBSD/SEQ/Makefile \
 $(top_srcdir)/example/FREEBSD/SEQ/extrae.xml \
 $(top_srcdir)/example/FREEBSD/SEQ/run_instrumented.sh \
 $(top_srcdir)/example/FREEBSD/SEQ/function-list \
 $(top_srcdir)/example/FREEBSD/SEQ/run_dyninst.sh

PTHREAD_EXAMPLE_FILES+=\
 $(top_srcdir)/example/FREEBSD/PTHREAD/pthread-example.c \
 $(top_srcdir)/example/FREEBSD/PTHREAD/README \
 $(top_srcdir)/example/FREEBSD/PTHREAD/Makefile \
 $(top_srcdir)/example/FREEBSD/PTHREAD/extrae.xml \
 $(top_srcdir)/example/FREEBSD/PTHREAD/user-comms.c

JAVA_EXAMPLE_FILES+=\
 $(top_srcdir)/example/FREEBSD/JAVA/manual

if WANT_JAVA_WITH_ASPECTJ
JAVA_EXAMPLE_FILES+=\
 $(top_srcdir)/example/FREEBSD/JAVA/automatic 
endif

OPENCL_EXAMPLE_FILES+=\
 $(top_srcdir)/example/LINUX/OPENCL/README \
 $(top_srcdir)/example/LINUX/OPENCL/C++ \
 $(top_srcdir)/example/LINUX/OPENCL/C

MPIOPENCL_EXAMPLE_FILES+=\
 $(top_srcdir)/example/LINUX/MPI+OPENCL/extrae.xml \
 $(top_srcdir)/example/LINUX/MPI+OPENCL/Makefile \
 $(top_srcdir)/example/LINUX/MPI+OPENCL/trace.sh \
 $(top_srcdir)/example/LINUX/MPI+OPENCL/README \
 $(top_srcdir)/example/LINUX/MPI+OPENCL/mpi-vadd.c
endif

if OS_LINUX
if !IS_BG_MACHINE
MPI_EXAMPLE_FILES+=\
 $(top_srcdir)/example/LINUX/MPI/README \
 $(top_srcdir)/example/LINUX/MPI/extrae.xml \
 $(top_srcdir)/example/LINUX/MPI/detailed_trace_basic.xml \
 $(top_srcdir)/example/LINUX/MPI/summarized_trace_basic.xml \
 $(top_srcdir)/example/LINUX/MPI/extrae_explained.xml \
 $(top_srcdir)/example/LINUX/MPI/extrae_bursts_1ms.xml \
 $(top_srcdir)/example/LINUX/MPI/static \
 $(top_srcdir)/example/LINUX/MPI/ld-preload \
 $(top_srcdir)/example/LINUX/MPI/dyninst

MPICUDA_EXAMPLE_FILES+=\
 $(top_srcdir)/example/LINUX/MPI+CUDA/ld-preload \
 $(top_srcdir)/example/LINUX/MPI+CUDA/dyninst \
 $(top_srcdir)/example/LINUX/MPI+CUDA/extrae.xml

OMP_EXAMPLE_FILES+=\
 $(top_srcdir)/example/LINUX/OMP/extrae.xml \
 $(top_srcdir)/example/LINUX/OMP/README \
 $(top_srcdir)/example/LINUX/OMP/Makefile \
 $(top_srcdir)/example/LINUX/OMP/pi.c \
 $(top_srcdir)/example/LINUX/OMP/pi_instrumented.c \
 $(top_srcdir)/example/LINUX/OMP/run_dyninst.sh \
 $(top_srcdir)/example/LINUX/OMP/run_ldpreload.sh \
 $(top_srcdir)/example/LINUX/OMP/run_instrumented.sh

MPIOMP_EXAMPLE_FILES+=\
 $(top_srcdir)/example/LINUX/MPI+OMP/extrae.xml \
 $(top_srcdir)/example/LINUX/MPI+OMP/README \
 $(top_srcdir)/example/LINUX/MPI+OMP/ld-preload

SEQ_EXAMPLE_FILES+=\
 $(top_srcdir)/example/LINUX/SEQ/extrae.xml \
 $(top_srcdir)/example/LINUX/SEQ/function-list \
 $(top_srcdir)/example/LINUX/SEQ/Makefile \
 $(top_srcdir)/example/LINUX/SEQ/pi.c \
 $(top_srcdir)/example/LINUX/SEQ/pi_instrumented.c \
 $(top_srcdir)/example/LINUX/SEQ/pi_instrumented.f90 \
 $(top_srcdir)/example/LINUX/SEQ/run_dyninst.sh \
 $(top_srcdir)/example/LINUX/SEQ/run_instrumented.sh \
 $(top_srcdir)/example/LINUX/SEQ/run_instrumentedf.sh \
 $(top_srcdir)/example/LINUX/SEQ/README

ONLINE_EXAMPLE_FILES+=\
 $(top_srcdir)/example/LINUX/ONLINE/Makefile \
 $(top_srcdir)/example/LINUX/ONLINE/ping.c \
 $(top_srcdir)/example/LINUX/ONLINE/extrae_online.xml \
 $(top_srcdir)/example/LINUX/ONLINE/run_mn.sh \
 $(top_srcdir)/example/LINUX/ONLINE/run_mt.sh \
 $(top_srcdir)/example/LINUX/ONLINE/trace_online.sh \
 $(top_srcdir)/example/LINUX/ONLINE/merge.sh \
 $(top_srcdir)/example/LINUX/ONLINE/merge_parallel_mn.sh \
 $(top_srcdir)/example/LINUX/ONLINE/merge_parallel_mt.sh \
 $(top_srcdir)/example/LINUX/ONLINE/README

PYTHON_EXAMPLE_FILES+=\
 $(top_srcdir)/example/LINUX/PYTHON/test.py \
 $(top_srcdir)/example/LINUX/PYTHON/extrae.xml \
 $(top_srcdir)/example/LINUX/PYTHON/function-list \
 $(top_srcdir)/example/LINUX/PYTHON/run.sh 

PTHREAD_EXAMPLE_FILES+=\
 $(top_srcdir)/example/LINUX/PTHREAD/Makefile \
 $(top_srcdir)/example/LINUX/PTHREAD/pthread-example.c \
 $(top_srcdir)/example/LINUX/PTHREAD/user-comms.c \
 $(top_srcdir)/example/LINUX/PTHREAD/extrae.xml \
 $(top_srcdir)/example/LINUX/PTHREAD/README

CUDA_EXAMPLE_FILES+=\
 $(top_srcdir)/example/LINUX/CUDA/extrae.xml \
 $(top_srcdir)/example/LINUX/CUDA/hello.cu \
 $(top_srcdir)/example/LINUX/CUDA/hello_instrumented.cu \
 $(top_srcdir)/example/LINUX/CUDA/Makefile \
 $(top_srcdir)/example/LINUX/CUDA/README \
 $(top_srcdir)/example/LINUX/CUDA/run_ldpreload.sh \
 $(top_srcdir)/example/LINUX/CUDA/run_dyninst.sh \
 $(top_srcdir)/example/LINUX/CUDA/run_instrumented.sh

OPENCL_EXAMPLE_FILES+=\
 $(top_srcdir)/example/LINUX/OPENCL/README \
 $(top_srcdir)/example/LINUX/OPENCL/C++ \
 $(top_srcdir)/example/LINUX/OPENCL/C

MPIOPENCL_EXAMPLE_FILES+=\
 $(top_srcdir)/example/LINUX/MPI+OPENCL/extrae.xml \
 $(top_srcdir)/example/LINUX/MPI+OPENCL/Makefile \
 $(top_srcdir)/example/LINUX/MPI+OPENCL/trace.sh \
 $(top_srcdir)/example/LINUX/MPI+OPENCL/README \
 $(top_srcdir)/example/LINUX/MPI+OPENCL/mpi-vadd.c

OMPSS_EXAMPLE_FILES+=\
 $(top_srcdir)/example/LINUX/OMPSS/extrae.xml \
 $(top_srcdir)/example/LINUX/OMPSS/heat.h \
 $(top_srcdir)/example/LINUX/OMPSS/heat-ompss.c \
 $(top_srcdir)/example/LINUX/OMPSS/Makefile \
 $(top_srcdir)/example/LINUX/OMPSS/misc.c \
 $(top_srcdir)/example/LINUX/OMPSS/solver-ompss.c \
 $(top_srcdir)/example/LINUX/OMPSS/trace.sh \
 $(top_srcdir)/example/LINUX/OMPSS/job.lsf \
 $(top_srcdir)/example/LINUX/OMPSS/test.dat

MPIOMPSS_EXAMPLE_FILES+=\
 $(top_srcdir)/example/LINUX/MPI+OMPSS/extrae.xml \
 $(top_srcdir)/example/LINUX/MPI+OMPSS/heat.h \
 $(top_srcdir)/example/LINUX/MPI+OMPSS/heat-mpi-ompss.c \
 $(top_srcdir)/example/LINUX/MPI+OMPSS/Makefile \
 $(top_srcdir)/example/LINUX/MPI+OMPSS/misc.c \
 $(top_srcdir)/example/LINUX/MPI+OMPSS/solver-mpi-ompss.c \
 $(top_srcdir)/example/LINUX/MPI+OMPSS/trace.sh \
 $(top_srcdir)/example/LINUX/MPI+OMPSS/job.lsf \
 $(top_srcdir)/example/LINUX/MPI+OMPSS/test.dat

JAVA_EXAMPLE_FILES+=\
 $(top_srcdir)/example/LINUX/JAVA/manual
if WANT_JAVA_WITH_ASPECTJ
JAVA_EXAMPLE_FILES+=\
 $(top_srcdir)/example/LINUX/JAVA/automatic
endif

FORK_EXAMPLE_FILES+=\
 $(top_srcdir)/example/LINUX/FORK/extrae.xml \
 $(top_srcdir)/example/LINUX/FORK/function-list \
 $(top_srcdir)/example/LINUX/FORK/Makefile \
 $(top_srcdir)/example/LINUX/FORK/pi_forked.c \
 $(top_srcdir)/example/LINUX/FORK/README \
 $(top_srcdir)/example/LINUX/FORK/run_dyninst.sh
endif
endif


install-data-hook:
	touch null # To ease the installation rules
#
# Apply post-installation modifications
#
	$(MKDIR_P) $(DESTDIR)$(datadir)/example
	cp -f $(top_srcdir)/example/Makefile.inc $(DESTDIR)$(datadir)/example
#
# Apply substitutions
#
	$(top_srcdir)/substitute $(SED) "@sub_MPI_HOME@" "${MPI_HOME}" $(DESTDIR)$(datadir)/example/Makefile.inc
	$(top_srcdir)/substitute $(SED) "@sub_MPI_CFLAGS@" "${MPI_CFLAGS}" $(DESTDIR)$(datadir)/example/Makefile.inc
	$(top_srcdir)/substitute $(SED) "@sub_MPI_LDFLAGS@" "${MPI_LDFLAGS}" $(DESTDIR)$(datadir)/example/Makefile.inc
	$(top_srcdir)/substitute $(SED) "@sub_MPI_LIBS@" "${MPI_LIBS}" $(DESTDIR)$(datadir)/example/Makefile.inc
	$(top_srcdir)/substitute $(SED) "@sub_PAPI_HOME@" "${PAPI_HOME}" $(DESTDIR)$(datadir)/example/Makefile.inc
	$(top_srcdir)/substitute $(SED) "@sub_PAPI_LIBS@" "${PAPI_LIBS}" $(DESTDIR)$(datadir)/example/Makefile.inc
	$(top_srcdir)/substitute $(SED) "@sub_UNWIND_HOME@" "${UNWIND_HOME}" $(DESTDIR)$(datadir)/example/Makefile.inc
	$(top_srcdir)/substitute $(SED) "@sub_UNWIND_LIBS@" "${UNWIND_LIBS}" $(DESTDIR)$(datadir)/example/Makefile.inc
	$(top_srcdir)/substitute $(SED) "@sub_XML2_HOME@" "${XML2_HOME}" $(DESTDIR)$(datadir)/example/Makefile.inc
	$(top_srcdir)/substitute $(SED) "@sub_XML2_LDFLAGS@" "${XML2_LDFLAGS}" $(DESTDIR)$(datadir)/example/Makefile.inc
	$(top_srcdir)/substitute $(SED) "@sub_XML2_LIBS@" "${XML2_LIBS}" $(DESTDIR)$(datadir)/example/Makefile.inc
	$(top_srcdir)/substitute $(SED) "@sub_BFD_HOME@" "${BFD_HOME}" $(DESTDIR)$(datadir)/example/Makefile.inc
	$(top_srcdir)/substitute $(SED) "@sub_LIBERTY_HOME@" "${LIBERTY_HOME}" $(DESTDIR)$(datadir)/example/Makefile.inc
	$(top_srcdir)/substitute $(SED) "@sub_BFD_LIBS@" "${BFD_LDFLAGS} ${BFD_LIBS}" $(DESTDIR)$(datadir)/example/Makefile.inc
	$(top_srcdir)/substitute $(SED) "@sub_LIBERTY_LIBS@" "${LIBERTY_LDFLAGS} ${LIBERTY_LIBS}" $(DESTDIR)$(datadir)/example/Makefile.inc
	$(top_srcdir)/substitute $(SED) "@sub_ZLIB_HOME@" "${LIBZ_HOME}" $(DESTDIR)$(datadir)/example/Makefile.inc
	$(top_srcdir)/substitute $(SED) "@sub_ZLIB_LIBS@" "${LIBZ_LIBS}" $(DESTDIR)$(datadir)/example/Makefile.inc
	$(top_srcdir)/substitute $(SED) "@sub_DL_LIBS@" "${DL_LIBS}" $(DESTDIR)$(datadir)/example/Makefile.inc
if WANT_CUPTI
	$(top_srcdir)/substitute $(SED) "@sub_CUPTI_LIBS@" "-L${CUPTI_HOME}/lib -lcupti" $(DESTDIR)$(datadir)/example/Makefile.inc
else
	$(top_srcdir)/substitute $(SED) "@sub_CUPTI_LIBS@" "" $(DESTDIR)$(datadir)/example/Makefile.inc
endif
if WANT_OPENCL
	$(top_srcdir)/substitute $(SED) "@sub_OPENCL_HOME@" "${OPENCL_HOME}" $(DESTDIR)$(datadir)/example/Makefile.inc
	$(top_srcdir)/substitute $(SED) "@sub_OPENCL_LDFLAGS@" "${OPENCL_LDFLAGS}" $(DESTDIR)$(datadir)/example/Makefile.inc
else
	$(top_srcdir)/substitute $(SED) "@sub_OPENCL_HOME@" "" $(DESTDIR)$(datadir)/example/Makefile.inc
	$(top_srcdir)/substitute $(SED) "@sub_OPENCL_LDFLAGS@" "" $(DESTDIR)$(datadir)/example/Makefile.inc
endif
	$(top_srcdir)/substitute $(SED) "@sub_POSIX_CLOCK@" "${POSIX_CLOCK_LIB}" $(DESTDIR)$(datadir)/example/Makefile.inc
	$(top_srcdir)/substitute $(SED) "@sub_CC@" "`which ${CC}`" $(DESTDIR)$(datadir)/example/Makefile.inc
	$(top_srcdir)/substitute $(SED) "@sub_CXX@" "`which ${CXX}`" $(DESTDIR)$(datadir)/example/Makefile.inc
	$(top_srcdir)/substitute $(SED) "@sub_FC@" "`which ${FC}`" $(DESTDIR)$(datadir)/example/Makefile.inc
if HAVE_MPI
	$(top_srcdir)/substitute $(SED) "@sub_MPICC@" "`which ${MPICC}`" $(DESTDIR)$(datadir)/example/Makefile.inc
	$(top_srcdir)/substitute $(SED) "@sub_MPIF77@" "`which ${MPIF77}`" $(DESTDIR)$(datadir)/example/Makefile.inc
	$(top_srcdir)/substitute $(SED) "@sub_MPIF90@" "`which ${MPIF90}`" $(DESTDIR)$(datadir)/example/Makefile.inc
else
	$(top_srcdir)/substitute $(SED) "@sub_MPICC@" "" $(DESTDIR)$(datadir)/example/Makefile.inc
	$(top_srcdir)/substitute $(SED) "@sub_MPIF77@" "" $(DESTDIR)$(datadir)/example/Makefile.inc
	$(top_srcdir)/substitute $(SED) "@sub_MPIF90@" "" $(DESTDIR)$(datadir)/example/Makefile.inc
endif
if WANT_CUDAorCUPTI
	$(top_srcdir)/substitute $(SED) "@sub_NVCC@" "`which ${NVCC}`" $(DESTDIR)$(datadir)/example/Makefile.inc
else
	$(top_srcdir)/substitute $(SED) "@sub_NVCC@" "" $(DESTDIR)$(datadir)/example/Makefile.inc
endif
if WANT_JAVA
	$(top_srcdir)/substitute $(SED) "@sub_JAVAC@" "`which ${JAVAC}`" $(DESTDIR)$(datadir)/example/Makefile.inc
	$(top_srcdir)/substitute $(SED) "@sub_JAR@" "`which ${JAR}`" $(DESTDIR)$(datadir)/example/Makefile.inc
else
	$(top_srcdir)/substitute $(SED) "@sub_JAVAC@" "" $(DESTDIR)$(datadir)/example/Makefile.inc
	$(top_srcdir)/substitute $(SED) "@sub_JAR@" "" $(DESTDIR)$(datadir)/example/Makefile.inc
endif
# Apply post-installation modifications (FreeBSD specific)
if OS_FREEBSD
	$(top_srcdir)/substitute $(SED) "@sub_libexecinfo_libs@" "${libexecinfo_LDFLAGS} -lexecinfo" $(DESTDIR)$(datadir)/example/Makefile.inc
else
	$(top_srcdir)/substitute $(SED) "@sub_libexecinfo_libs@" "" $(DESTDIR)$(datadir)/example/Makefile.inc
endif
#
# Now copy examples
#
	$(MKDIR_P) $(DESTDIR)$(datadir)/example/SEQ
	cp -f $(SEQ_EXAMPLE_FILES) $(DESTDIR)$(datadir)/example/SEQ
	cd $(DESTDIR)$(datadir)/example/SEQ ; cp ../Makefile.inc .
	$(MKDIR_P) $(DESTDIR)$(datadir)/example/FORK
	cp -f $(FORK_EXAMPLE_FILES) $(DESTDIR)$(datadir)/example/FORK
	cd $(DESTDIR)$(datadir)/example/FORK ; cp ../Makefile.inc .
	$(MKDIR_P) $(DESTDIR)$(datadir)/example/PYTHON
	cp -f $(PYTHON_EXAMPLE_FILES) $(DESTDIR)$(datadir)/example/PYTHON
if HAVE_MPI
	$(MKDIR_P) $(DESTDIR)$(datadir)/example/MPI
	cp -fr $(MPI_EXAMPLE_FILES) $(DESTDIR)$(datadir)/example/MPI
	cd $(DESTDIR)$(datadir)/example/MPI ; cp ../Makefile.inc .
if WANT_CUDAorCUPTI
	$(MKDIR_P) $(DESTDIR)$(datadir)/example/MPI+CUDA
	cp -fr $(MPICUDA_EXAMPLE_FILES) $(DESTDIR)$(datadir)/example/MPI+CUDA
	cd $(DESTDIR)$(datadir)/example/MPI+CUDA ; cp ../Makefile.inc .
endif
if WANT_OPENCL
	$(MKDIR_P) $(DESTDIR)$(datadir)/example/MPI+OPENCL
	cp -fr $(MPIOPENCL_EXAMPLE_FILES) $(DESTDIR)$(datadir)/example/MPI+OPENCL
	cd $(DESTDIR)$(datadir)/example/MPI+OPENCL ; cp ../Makefile.inc .
endif
if WANT_NANOS
	$(MKDIR_P) $(DESTDIR)$(datadir)/example/MPI+OMPSS
	cp -fr $(MPIOMPSS_EXAMPLE_FILES) $(DESTDIR)$(datadir)/example/MPI+OMPSS
	cd $(DESTDIR)$(datadir)/example/MPI+OMPSS ; cp ../Makefile.inc .
endif
if WANT_OPENMP
	$(MKDIR_P) $(DESTDIR)$(datadir)/example/MPI+OMP
	cp -fr $(MPIOMP_EXAMPLE_FILES) $(DESTDIR)$(datadir)/example/MPI+OMP
	cd $(DESTDIR)$(datadir)/example/MPI+OMP ; cp ../Makefile.inc .
endif
endif
if WANT_OPENMP
	$(MKDIR_P) $(DESTDIR)$(datadir)/example/OMP
	cp -f $(OMP_EXAMPLE_FILES) $(DESTDIR)$(datadir)/example/OMP
	cd $(DESTDIR)$(datadir)/example/OMP ; cp ../Makefile.inc .
endif
if HAVE_ONLINE
	$(MKDIR_P) $(DESTDIR)$(datadir)/example/ONLINE
	cp -f $(ONLINE_EXAMPLE_FILES) $(DESTDIR)$(datadir)/example/ONLINE
	cd $(DESTDIR)$(datadir)/example/ONLINE ; cp ../Makefile.inc .
endif
if WANT_PTHREAD
	$(MKDIR_P) $(DESTDIR)$(datadir)/example/PTHREAD
	cp -f $(PTHREAD_EXAMPLE_FILES) $(DESTDIR)$(datadir)/example/PTHREAD
	cd $(DESTDIR)$(datadir)/example/PTHREAD ; cp ../Makefile.inc .
endif
if WANT_CUDAorCUPTI
	$(MKDIR_P) $(DESTDIR)$(datadir)/example/CUDA
	cp -f $(CUDA_EXAMPLE_FILES) $(DESTDIR)$(datadir)/example/CUDA
	cd $(DESTDIR)$(datadir)/example/CUDA ; cp ../Makefile.inc .
endif
if WANT_OPENCL
	$(MKDIR_P) $(DESTDIR)$(datadir)/example/OPENCL
	cp -fr $(OPENCL_EXAMPLE_FILES) $(DESTDIR)$(datadir)/example/OPENCL
	cd $(DESTDIR)$(datadir)/example/OPENCL ; cp ../Makefile.inc .
endif
if WANT_NANOS
	$(MKDIR_P) $(DESTDIR)$(datadir)/example/OMPSS
	cp -fr $(OMPSS_EXAMPLE_FILES) $(DESTDIR)$(datadir)/example/OMPSS
	cd $(DESTDIR)$(datadir)/example/OMPSS ; cp ../Makefile.inc .
endif
if WANT_JAVA
	$(MKDIR_P) $(DESTDIR)$(datadir)/example/JAVA
	cp -fr $(JAVA_EXAMPLE_FILES) $(DESTDIR)$(datadir)/example/JAVA
	cd $(DESTDIR)$(datadir)/example/JAVA ; cp ../Makefile.inc .
endif
	$(MKDIR_P) $(DESTDIR)$(datadir)/example/UNICORE
	cp -fr $(UNICORE_EXAMPLE_FILES) $(DESTDIR)$(datadir)/example/UNICORE
#
# Apply modifications to all directories installed
#
	$(top_srcdir)/substitute-all $(SED) "@sub_PREFIXDIR@" "${prefix}" $(DESTDIR)$(datadir)/example
	rm null # Remove null file created at the beginning
# SVN installations may have some subversion control files
	find $(DESTDIR)$(datadir)/example -name .svn -delete
	find $(DESTDIR)$(datadir)/example -name null -delete

uninstall-local:
	rm -r $(DESTDIR)$(datadir)/example/SEQ \
	      $(DESTDIR)$(datadir)/example/FORK \
	      $(DESTDIR)$(datadir)/example/PYTHON
if HAVE_MPI
	rm -r $(DESTDIR)$(datadir)/example/MPI
if WANT_CUDAorCUPTI
	rm -r $(DESTDIR)$(datadir)/example/MPI+CUDA
endif
if WANT_OPENCL
	rm -r $(DESTDIR)$(datadir)/example/MPI+OPENCL
endif
if WANT_NANOS
	rm -r $(DESTDIR)$(datadir)/example/MPI+OMPSS
endif
if WANT_OPENMP
	rm -r $(DESTDIR)$(datadir)/example/MPI+OMP
endif
endif
if WANT_OPENMP
	rm -r $(DESTDIR)$(datadir)/example/OMP
endif
if HAVE_ONLINE
	rm -r $(DESTDIR)$(datadir)/example/ONLINE
endif
if WANT_PTHREAD
	rm -r $(DESTDIR)$(datadir)/example/PTHREAD
endif
if WANT_CUDAorCUPTI
	rm -r $(DESTDIR)$(datadir)/example/CUDA
endif
if WANT_OPENCL
	rm -r $(DESTDIR)$(datadir)/example/OPENCL
endif
if WANT_NANOS
	rm -r $(DESTDIR)$(datadir)/example/OMPSS
endif
if WANT_JAVA
	rm -r $(DESTDIR)$(datadir)/example/JAVA
endif
	rm -r $(DESTDIR)$(datadir)/example/UNICORE
	rm    $(DESTDIR)$(datadir)/example/Makefile.inc
	rmdir --ignore-fail-on-non-empty $(DESTDIR)$(datadir)/example
	rmdir --ignore-fail-on-non-empty $(DESTDIR)$(bindir)
	rmdir --ignore-fail-on-non-empty $(DESTDIR)$(libdir)
	rmdir --ignore-fail-on-non-empty $(DESTDIR)$(libexecdir)
	rmdir --ignore-fail-on-non-empty $(DESTDIR)$(datadir)
	rmdir --ignore-fail-on-non-empty $(DESTDIR)$(prefix)
