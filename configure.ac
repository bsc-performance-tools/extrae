
# Process this file with autoconf to produce a configure script.

#########################
#    Initializations    #
#########################

m4_define([extrae_major], [3])
m4_define([extrae_minor], [5])
m4_define([extrae_micro], [2])

m4_define([extrae_version], [extrae_major.extrae_minor.extrae_micro])

# Initialize autoconf & define package name, version and bug-report address 
AC_INIT(Extrae, [extrae_version], tools@bsc.es)

# Safety check to ensure that the directory told with `--srcdir` contains the source code
#AC_CONFIG_SRCDIR(src/mpitrace.c)

# GNU Autotools intermediate files are stored in the following directory
AC_CONFIG_AUX_DIR([config])
AC_CONFIG_MACRO_DIR([config])

echo Configuring package ${PACKAGE_STRING}

# Loads some shell variables like host_cpu and host_os, to get the host information 
AC_CANONICAL_SYSTEM

AX_SYSTEM_TYPE

# Check if exist /proc/self/maps for address translation
if test ${cross_compiling} == "no"; then
    AC_CHECK_FILE(
        [/proc/self/maps],
        [AC_DEFINE([HAVE_PROC_MAPS],[1],[Define if have /proc/self/maps])],
        AC_MSG_WARN(["/proc/self/maps" not found. Address translation may be not fully supported.])
    )
fi

# Initialize automake
AM_INIT_AUTOMAKE([tar-ustar subdir-objects no-dependencies])

dnl Enable silent rules (if these exist), use with make V=0  or V=1
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

dnl AM_PROG_MKDIR_P
AC_PROG_MKDIR_P

# Specify the output configuration header file
AC_CONFIG_HEADER(config.h)

#############################
#    Checks for programs    #
#############################

# Search for available compilers and preprocessor
# AC_PROG_CC does not let us give it a absolute path with the compiler
if test "${IS_BG_MACHINE}" = "yes" ; then

   if test "${CC}" ; then
      AC_MSG_NOTICE([Attention!])
      AC_MSG_NOTICE([Attention! CC is defined on a BG/* system. Use it with caution!])
      AC_MSG_NOTICE([Attention!])
   else
      if test "${IS_BGL_MACHINE}" = "yes" ; then
         FC="${BG_HOME}/blrts-gnu/bin/powerpc-bgl-blrts-gnu-gfortran"
         CC="${BG_HOME}/blrts-gnu/bin/powerpc-bgl-blrts-gnu-gcc"
         CXX="${BG_HOME}/blrts-gnu/bin/powerpc-bgl-blrts-gnu-g++"
      elif test "${IS_BGP_MACHINE}" = "yes" ; then
         FC="${BG_HOME}/gnu-linux/bin/powerpc-bgp-linux-gfortran"
         CC="${BG_HOME}/gnu-linux/bin/powerpc-bgp-linux-gcc"
         CXX="${BG_HOME}/gnu-linux/bin/powerpc-bgp-linux-g++"
      elif test "${IS_BGQ_MACHINE}" = "yes" ; then
         FC="${BG_HOME}/gnu-linux/bin/powerpc64-bgq-linux-gfortran"
         CC="${BG_HOME}/gnu-linux/bin/powerpc64-bgq-linux-gcc"
         CXX="${BG_HOME}/gnu-linux/bin/powerpc64-bgq-linux-g++"
         AR="${BG_HOME}/gnu-linux/bin/powerpc64-bgq-linux-ar"
         LD="${BG_HOME}/gnu-linux/bin/powerpc64-bgq-linux-ld"
      fi
   fi
elif test "${IS_SPARC64_MACHINE}" = "yes" ; then

	if test "${CC}" ; then
      AC_MSG_NOTICE([Attention!])
      AC_MSG_NOTICE([Attention! CC is defined on a SPARC64 system. Use it with caution!])
      AC_MSG_NOTICE([Attention!])
	else
		FC=`which frtpx`
		CC=`which fccpx` 
		CXX=`which FCCpx`
		#AR=/opt/FJSVXosDevkit/sparc64fx/V01L03E07/sparc64-unknown-linux-gnu/bin/ar
		#LD=/opt/FJSVXosDevkit/sparc64fx/V01L03E07/sparc64-unknown-linux-gnu/bin/ld
	fi

else
   wanted_C_Compilers="gcc xlc icc cc"
   wanted_F_Compilers="gfortran xlf ifort f90 ftn"
   wanted_CXX_Compilers="g++ xlC icpc CC"
fi
AC_PROG_CC(${wanted_C_Compilers})
if test "${IS_SPARC64_MACHINE}" != "yes" ; then
	AX_COMPILER_VERSION
fi
AC_PROG_CXX(${wanted_CXX_Compilers})
AC_PROG_CPP
AC_PROG_FC(${wanted_F_Compilers})

AM_CONDITIONAL(HAVE_FC, test "$FC" != "")

if test "${IS_SPARC64_MACHINE}" != "yes" ; then
	AC_FC_WRAPPERS
fi
AX_JAVA
AX_JAVA_ASPECTJ
AC_PROG_INSTALL
AC_PROG_SED
AC_PROG_AWK

if test "${GCC}" = "yes" ; then
	if test "${IS_SPARC64_MACHINE}" != "yes" ; then
		CFLAGS="${CFLAGS} -fno-optimize-sibling-calls"
		CXXFLAGS="${CXXFLAGS} -fno-optimize-sibling-calls"
	fi
	if test "${Architecture}" = "arm" ; then
		CFLAGS="${CFLAGS} -funwind-tables"
		CXXFLAGS="${CXXFLAGS} -funwind-tables"
	fi
fi

# Automake 1.10 reports a problem if this is not defined
AM_PROG_CC_C_O

# Check for the AR command (only if AM_PROG_AR exists)
m4_ifdef([AM_PROG_AR], [AM_PROG_AR])

# Check for /proc/cpuinfo & /proc/meminfo
AX_CHECK_PROC_CPUINFO
AX_CHECK_PROC_MEMINFO

# Check whether we can determine on which CPU we're working
AX_CHECK_GETCPU

# PGI compilers do not handle well GCC attributes
AX_CHECK_PGI

# Check if attribute weak, alias works
AX_CHECK_ALIAS_ATTRIBUTE
AX_CHECK_WEAK_ALIAS_ATTRIBUTE
AX_CHECK_UNUSED_ATTRIBUTE

# Checks for 64bit files
AX_OFF_T_64BIT
AC_CHECK_FUNC(fopen64, [AC_DEFINE([HAVE_FOPEN64],[1],[Define if have fopen64])])
AC_CHECK_FUNC(ftell64, [AC_DEFINE([HAVE_FTELL64],[1],[Define if have ftell64])])
AC_CHECK_FUNC(ftello64, [AC_DEFINE([HAVE_FTELLO64],[1],[Define if have ftello64])])
AC_CHECK_FUNC(fseek64, [AC_DEFINE([HAVE_FSEEK64],[1],[Define if have fseek64])])
AC_CHECK_FUNC(fseeko64, [AC_DEFINE([HAVE_FSEEKO64],[1],[Define if have fseeko64])])
AC_CHECK_FUNC(fgetpos64, [AC_DEFINE([HAVE_FGETPOS64],[1],[Define if have fgetpos64])])
AC_CHECK_FUNC(fsetpos64, [AC_DEFINE([HAVE_FSETPOS64],[1],[Define if have fsetpos64])])
AC_CHECK_FUNC(stat, [AC_DEFINE([HAVE_STAT],[1],[Define if have stat])])
AC_CHECK_FUNC(stat64, [AC_DEFINE([HAVE_STAT64],[1],[Define if have stat64])])
AC_CHECK_FUNC(access, [AC_DEFINE([HAVE_ACCESS],[1],[Define if have access])])

# Checks for sleep operations
AC_CHECK_FUNC(sleep, [have_sleep="yes"], [have_sleep="no"])
AC_CHECK_FUNC(usleep, [have_usleep="yes"], [have_usleep="no"])
AC_CHECK_FUNC(nanosleep, [have_nanosleep="yes"], [have_nanosleep="no"])
AM_CONDITIONAL(HAVE_SLEEP, test "${have_sleep}" = "yes")
AM_CONDITIONAL(HAVE_USLEEP, test "${have_usleep}" = "yes")
AM_CONDITIONAL(HAVE_NANOSLEEP, test "${have_nanosleep}" = "yes")

# Check if sysconf is available
AC_CHECK_FUNC(sysconf, [AC_DEFINE([HAVE_SYSCONF],[1],[Define if have sysconf])])

## Check if we have MPI
AX_PROG_MPI

## Check if we have OpenSHMEM
AX_PROG_OPENSHMEM

## Check how we hook PMPI
AX_PMPI_HOOK

# Search for libtool support
AC_LIBTOOL_DLOPEN
#if test "${MPI_SHARED_LIB_FOUND}" != "yes" ; then
#   AC_DISABLE_SHARED
#else
   AC_ENABLE_SHARED
#fi
AC_PROG_LIBTOOL

AC_CHECK_LIB(dl, dlopen, have_dl=yes, have_dl=no)
AM_CONDITIONAL(HAVE_LIBDL, test "${have_dl}" = "yes" )
if test "${have_dl}" = "yes"; then
	AC_SUBST([DL_LIBS], ["-ldl"])
fi

# Search for headers
AC_CHECK_HEADERS(
 [stdint.h netdb.h errno.h string.h strings.h unistd.h rts.h \
  byteswap.h limits.h malloc.h stdio.h stdlib.h values.h assert.h \
  bgl_perfctr.h bgl_perfctr_events.h ctype.h dlfcn.h excpt.h fcntl.h getopt.h \
  libgen.h libspe.h libspe2.h pdsc.h signal.h stdarg.h math.h inttypes.h time.h \
  ucontext.h pthread.h semaphore.h execinfo.h dirent.h]
)
AC_CHECK_HEADERS(
  [sys/types.h sys/socket.h sys/utsname.h sys/wait.h sys/resource.h \
   sys/sysctl.h sys/time.h sys/stat.h sys/procfs.h sys/mman.h sys/ioctl.h \
   sys/file.h sys/endian.h sys/systeminfo.h sys/uio.h]
)
AC_CHECK_HEADERS(
  [asm-ppc/atomic.h asm-ppc64/atomic.h]
)
AC_CHECK_HEADERS(
  [linux/limits.h linux/mmtimer.h]
)
AC_CHECK_HEADERS(
  [netinet/in.h]
)
saved_CFLAGS=${CFLAGS}
CFLAGS="-I/usr/src/linux"
AC_CHECK_HEADERS(
  [arch/powerpc/include/asm/atomic.h]
)
CFLAGS=${saved_CFLAGS}

AC_CHECK_HEADERS(
  [firmware/include/personality.h spi/include/kernel/process.h \
   spi/include/kernel/location.h]
)

AX_CHECK_UCONTEXT

##
## Check for clock routines
##
AX_CHECK_POSIX_CLOCK
AX_CHECK_GETTIMEOFDAY_CLOCK

##
## Check for MPI things
##

AX_CHECK_PERUSE
if test "${MPI_INSTALLED}" = "yes" ; then
   AX_CHECK_PMPI_NAME_MANGLING
   AX_CHECK_MPI_STATUS_SIZE
   AX_CHECK_MPI_SOURCE_OFFSET
   AX_CHECK_MPI_TAG_OFFSET
   AX_CHECK_MPI_LIB_HAS_MPI_INIT_THREAD
   AX_CHECK_MPI_C_HAS_FORTRAN_MPI_INIT
   AX_CHECK_MPI_SUPPORTS_MPI_IO
   AX_CHECK_MPI_SUPPORTS_MPI_1SIDED
   AX_CHECK_MPI_SUPPORTS_MPI_COMM_SPAWN
   AX_CHECK_MPI_F_STATUS_IGNORE
   AX_CHECK_MPI3_VERSION
fi
AX_ENABLE_SINGLE_MPI_LIBRARY
AX_CHECK_MPI_LIB_HAS_C_AND_FORTRAN_SYMBOLS

dnl If MPI installation supports MPI3
AM_CONDITIONAL(HAVE_MPI3, test "${mpi3_version}" = "yes")

AC_ARG_ENABLE(parallel-merge,
   AC_HELP_STRING(
      [--enable-parallel-merge],
      [Build the parallel mergers (mpimpi2prv/mpimpi2dim) based on MPI (enabled by default)]
   ),
   [enable_parallel_merge="${enableval}"],
   [enable_parallel_merge="yes"]
)
AM_CONDITIONAL(WANT_PARALLEL_MERGE, test "${enable_parallel_merge}" = "yes" )

AC_ARG_ENABLE(merge-in-trace,
   AC_HELP_STRING(
      [--enable-merge-in-trace],
      [Embed the merging functionality (mpi2prv) into the tracing library itself (disabled by default)]
   ),
   [enable_merge_in_trace="${enableval}"],
   [enable_merge_in_trace="${enable_parallel_merge}"] # the auto value for merge in trace depends whether if parallel merge was enabled
)
AM_CONDITIONAL(WANT_MERGE_IN_TRACE, test "${enable_merge_in_trace=}" = "yes" )

#if test "${OperatingSystem}" = "linux" ; then
#  if test "${Architecture}" = "ia64" -o "${target_cpu}" = "amd64" -o "${target_cpu}" = "x86_64" ; then
#    AX_CHECK_UNWIND
#  fi
#fi
AX_CHECK_UNWIND
AM_CONDITIONAL(HAVE_UNWIND, test "${libunwind_works}" = "yes" )

AX_CHECK_LIBZ

AX_PROG_GM
AX_PROG_MX
AX_PROG_LIBEXECINFO
AX_PROG_DYNINST
AX_PROG_COUNTERS # Check for PAPI and/or PMAPI
AX_PROG_SIONLIB

AX_PROG_BINUTILS

#######################################
# Enable/Disable some tracing options #
#######################################

AX_CHECK_OPENMP

AX_CUDA
AX_CUPTI
AX_OPENCL
AX_PTHREAD

AC_ARG_ENABLE(smpss,
   AC_HELP_STRING(
      [--enable-smpss],
      [Enable support for tracing SMP-superscalar (enabled by default)]
   ),
   [enable_smpss="${enableval}"],
   [enable_smpss="yes"]
)
AM_CONDITIONAL(WANT_SMPSS, test "${enable_smpss}" = "yes" )

AC_ARG_ENABLE(nanos, 
   AC_HELP_STRING(
      [--enable-nanos],
      [Enable support for tracing Nanos run-time (enabled by default)]
   ),
   [enable_nanos="${enableval}"],
   [enable_nanos="yes"]
)
AM_CONDITIONAL(WANT_NANOS, test "${enable_nanos}" = "yes" )

AC_ARG_ENABLE(heterogeneous,
   AC_HELP_STRING(
      [--enable-heterogeneous],
      [Enable support for heterogeneous tracing (disabled by default)]
   ),
   [enable_hetero="${enableval}"],
   [enable_hetero="no"]
)
if test "${enable_hetero}" = "yes" ; then
	AC_DEFINE([HETEROGENEOUS_SUPPORT], [1], [Determine if the heterogeneous support is enabled])
fi

AC_ARG_ENABLE(xml,
   AC_HELP_STRING(
      [--enable-xml],
      [Enable support for XML configuration (enabled by default - disabled on BG/*)]
   ),
   [enable_xml="${enableval}"],
   [enable_xml="yes"]
)
if test "${enable_xml}" = "yes" ; then
   # BG/L does not have XML2
   if test "${IS_BGL_MACHINE}" = "yes" -o "${IS_BGP_MACHINE}" = "yes" ; then
      XML_enabled="no"
   else
      XML_enabled="yes"
      AM_PATH_XML2(
          [2.5.0],
          [AC_DEFINE([HAVE_XML2], [1], [Defined if libxml2 exists])],
          [AC_MSG_ERROR(Cannot find xml2-config of libXML 2.5.0 (or above))]
      )
      AX_PROG_XML2
   fi
else
   XML_enabled="no"
fi
AM_CONDITIONAL(HAVE_XML2, test "${XML_enabled}" = "yes" )

AC_ARG_ENABLE(doc,
   AC_HELP_STRING(
      [--enable-doc],
      [Generates the documentation of this instrumentation package (requires sphinx-doc)]
   ),
   [enable_doc="${enableval}"],
   [enable_doc="no"]
)
AM_CONDITIONAL(GENERATE_DOCUMENTATION, test "${enable_doc}" = "yes" )

AC_ARG_ENABLE(instrument-io,
	AC_HELP_STRING(
		[--disable-instrument-io],
		[Disables instrumentation for basic I/O (read, write) calls (experimental, enabled by default)]
	),
	[enable_instrument_io="${enableval}"],
	[enable_instrument_io="yes"]
)
if test "${enable_instrument_io}" = "yes" ; then
	AC_DEFINE([INSTRUMENT_IO], [1], [Determine whether I/O calls are instrumented])
fi
AM_CONDITIONAL(INSTRUMENT_IO, test "${enable_instrument_io}" = "yes" )

AC_ARG_ENABLE(instrument-dynamic-memory,
	AC_HELP_STRING(
		[--disable-instrument-dynamic-memory],
		[Disables instrumentation for dynamic memory (malloc, free, realloc) calls (experimental, enabled by defaultl)]
	),
	[enable_instrument_dynamic_memory="${enableval}"],
	[enable_instrument_dynamic_memory="yes"]
)

if test "${enable_instrument_dynamic_memory}" = "yes" ; then
	AC_DEFINE([INSTRUMENT_DYNAMIC_MEMORY], [1], [Determine whether dynamic memory calls are instrumented])
fi
AM_CONDITIONAL(INSTRUMENT_DYNAMIC_MEMORY, test "${enable_instrument_dynamic_memory}" = "yes" )

AX_PROG_MEMKIND

AC_ARG_ENABLE(instrument-syscall,
  AC_HELP_STRING(
    [--disable-instrument-syscall],
    [Disables instrumentation for system calls (other than I/O and mallocs) (experimental, enabled by default)]
  ),
  [enable_instrument_syscall="${enableval}"],
  [enable_instrument_syscall="yes"]
)
if test "${enable_instrument_syscall}" = "yes" ; then
  AC_DEFINE([INSTRUMENT_SYSCALL], [1], [Determine whether system calls (other than I/O and mallocs) are instrumented])
fi
AM_CONDITIONAL(INSTRUMENT_SYSCALL, test "${enable_instrument_syscall}" = "yes" )

AC_ARG_ENABLE(bfd-generate-addresses,
	AC_HELP_STRING(
		[--enable-bfd-generate-addresses],
		[Generates information for data objects found in binaries (experimental, enabled if --enable-instrument-dynamic-memory is enabled)]
	),
	[enable_bfd_generate_addresses="${enableval}"],
	[
		if test "${enable_instrument_dynamic_memory}" = "yes" -a "${have_binutils}" = "yes" ; then
      enable_bfd_generate_addresses="yes"
    fi
	]
)
if test "${enable_bfd_generate_addresses}" = "yes" ; then
	AC_DEFINE([BFD_MANAGER_GENERATE_ADDRESSES], [1], [Determine if BFD must generate information for data objects in binary])
fi

AC_ARG_ENABLE(dcarrera-hadoop,
	AC_HELP_STRING(
		[--enable-dcarrera-hadoop],
		[Specific changes for David Carrera to use in Hadopp (experimental, disabled by default)]
	),
	[enable_dcarrera_hadoop="${enableval}"],
	[enable_dcarrera_hadoop="no"]
)
if test "${enable_dcarrera_hadoop}" = "yes" ; then
	AC_DEFINE([DCARRERA_HADOOP], [1], [Determine if Davids tweaks must be applied])
fi

#################################
#        On-line checks         #
#################################

AX_PROG_ONLINE

######################################################################
#    Checks for typedefs, structures and compiler characteristics    #
######################################################################

# Check the existence of the following data types and its size

if test "${IS_BGL_MACHINE}" = "yes" -o "${IS_BGP_MACHINE}" = "yes" ; then
   cross_compiling="yes" # Force AC_CHECK_SIZEOF use the following values
   long_long_size=8
   long_size=4
   int_size=4
   pid_t_size=4
   ssize_t_size=4
   size_t_size=4
   voidp_size=4
   short_size=2
   char_size=1
elif test "${IS_BGQ_MACHINE}" = "yes" ; then
   cross_compiling="yes" # Force AC_CHECK_SIZEOF use the following values
   long_long_size=8
   long_size=8
   int_size=4
   pid_t_size=4
   ssize_t_size=8
   size_t_size=8
   voidp_size=8
   short_size=2
   char_size=1
elif test "${IS_MIC_MACHINE}" = "yes" ; then
	cross_compiling="yes"
	long_long_size=8
	long_size=8
	int_size=4
	pid_t_size=4
	ssize_t_size=8
	size_t_size=8
	voidp_size=8
	short_size=2
	char_size=1
elif test "${IS_ARM_MACHINE}" = "yes" ; then
	cross_compiling="yes"
	long_long_size=8
	long_size=4
	int_size=4
	pid_t_size=4
	ssize_t_size=4
	size_t_size=4
	voidp_size=4
	short_size=2
	char_size=1
elif test "${IS_SPARC64_MACHINE}" = "yes" ; then
	cross_compiling="yes"
	long_long_size=8
	long_size=8
	int_size=4
	pid_t_size=4
	ssize_t_size=8
	size_t_size=8
	voidp_size=8
	short_size=2
	char_size=1
else
	cross_compiling="no" # Force AC_CHECK_SIZEOF calculate these values
	long_long_size=0
	long_size=0
	int_size=0
	pid_t_size=0
	ssize_t_size=0
	size_t_size=0
	voidp_size=0
	short_size=0
	char_size=0
fi

AC_CHECK_TYPES([int64_t, uint64_t, int32_t, uint32_t, int16_t, uint16_t, \
int8_t, uint8_t, off_t, ssize_t, size_t, pid_t, long, long long, char, \
int, short])

AC_CHECK_SIZEOF(long long,${long_long_size})
AC_CHECK_SIZEOF(long,${long_size})
AC_CHECK_SIZEOF(short,${short_size})
AC_CHECK_SIZEOF(int,${int_size})
AC_CHECK_SIZEOF(char,${char_size})
AC_CHECK_SIZEOF(off_t,${off_t_size})
AC_CHECK_SIZEOF(pid_t,${pid_t_size})
AC_CHECK_SIZEOF(ssize_t,${size_t_size})
AC_CHECK_SIZEOF(size_t,${size_t_size})
AC_CHECK_SIZEOF(void*,${voidp_size})

AC_CHECK_MEMBER(struct mallinfo.arena,
   [AC_DEFINE([HAVE_MALLINFO], [1], [Whether the system supports mallinfo structure])],
   [],
   [#include <malloc.h>])

if test "${IS_BGL_MACHINE}" = "yes" -o "${IS_BGP_MACHINE}" = "yes" -o "${IS_BGQ_MACHINE}" = "yes" ; then
   cross_compiling="no"
fi

cross_compiling="no"

# Test if the architecture is little or big endian
AX_CHECK_ENDIANNESS

# Check whether the compiler defines the __FUNCTION__ macro
AX_CHECK__FUNCTION__MACRO

# Select compiler-specific flags to print warnings
if test "${IS_SPARC64_MACHINE}" != "yes" ; then 
	AX_CFLAGS_WARN_ALL
	AX_CXXFLAGS_WARN_ALL
	#AX_CFLAGS_WARN_ALL_ANSI
	#AX_CXXFLAGS_WARN_ALL_ANSI
fi

# Test whether the compiler accepts function inlining
AC_C_INLINE

AX_PEBS_SAMPLING

AC_CHECK_PROGS(latex,[latex elatex lambda],no)
AC_CHECK_PROGS(dvips,[dvips],no)
AC_CHECK_PROGS(dvipdf,[dvipdf],no)
AC_CHECK_PROGS(latex2html,[latex2html],no)

AM_CONDITIONAL(BUILD_DOCS_PS, test "${latex}" != "no" -a "${dvips}" != "no")
AM_CONDITIONAL(BUILD_DOCS_PDF, test "${latex}" != "no" -a "${dvipdf}" != "no")
AM_CONDITIONAL(BUILD_HTML_DOCS, test "${latex}" != "no" -a "${latex2html}" != "no")

#########################################
#    Checks for user defined options    #
#########################################

AC_CONFIG_FILES([
  Makefile \
  src/Makefile \
  include/Makefile \
  src/loader/Makefile \
  src/launcher/Makefile \
  src/launcher/static/Makefile \
  src/launcher/dyninst/Makefile \
  src/launcher/java/Makefile \
  scripts/Makefile \
  src/common/Makefile \
  src/common/MPI/Makefile \
  src/tracer/Makefile \
  src/tracer/clocks/Makefile \
  src/tracer/interfaces/Makefile 
  src/tracer/interfaces/API/Makefile \
  src/tracer/interfaces/MPI/Makefile \
  src/tracer/wrappers/Makefile 
  src/tracer/wrappers/MPI/Makefile \
  src/tracer/wrappers/OMP/Makefile \
  src/tracer/wrappers/OMP/intel-kmpc-11-intermediate/Makefile \
  src/tracer/wrappers/CUDA/Makefile \
  src/tracer/wrappers/OPENCL/Makefile \
  src/tracer/wrappers/pthread/Makefile \
  src/tracer/wrappers/openshmem/Makefile \
  src/tracer/wrappers/MALLOC/Makefile \
  src/tracer/wrappers/IO/Makefile \
  src/tracer/wrappers/SYSCALL/Makefile \
  src/tracer/wrappers/fork/Makefile \
  src/tracer/wrappers/JAVA/Makefile \
  src/tracer/stats/Makefile \
  src/tracer/hwc/Makefile \
  src/tracer/sampling/Makefile \
  src/tracer/sampling/common/Makefile \
  src/tracer/sampling/timer/Makefile \
  src/tracer/sampling/intel-pebs/Makefile \
  src/merger/Makefile \
  src/merger/parallel/Makefile \
  src/others/Makefile \
  src/cmd-line/Makefile \
  src/java-connector/Makefile \
  src/java-connector/jni/Makefile \
  src/java-connector/jvmti-agent/Makefile \
  etc/Makefile])

AC_CONFIG_FILES([\
  tests/Makefile
  tests/src/Makefile \
  tests/src/common/Makefile \
  tests/src/tracer/Makefile \
  tests/src/tracer/clocks/Makefile \
  tests/functional/Makefile \
  tests/functional/launcher/Makefile \
  tests/functional/tracer/Makefile \
  tests/functional/tracer/OTHER/Makefile \
  tests/functional/tracer/PTHREAD/Makefile \
  tests/functional/tracer/MPI/Makefile \
  tests/functional/tracer/JAVA/Makefile \
  tests/functional/tracer/OPENCL/Makefile \
  tests/functional/tracer/CUDA/Makefile \
  tests/functional/merger/Makefile \
  tests/functional/merger/dump-events/Makefile \
  tests/functional/merger/shared-libraries/Makefile \
  tests/functional/xml/Makefile \
  tests/functional/hw-counters/Makefile \
  tests/functional/callstack/Makefile \
  tests/overhead/Makefile])

AC_CONFIG_FILES([src/tracer/online/Makefile src/tracer/online/extractors/Makefile])
### TEMPORARILY_DISABLED
### AC_CONFIG_FILES([src/tracer/hwc/myrinet_hwc/Makefile src/tracer/hwc/myrinet_hwc/GM/Makefile src/tracer/hwc/myrinet_hwc/MX/Makefile])

AC_CONFIG_FILES([docs/Makefile])

AC_CONFIG_LINKS([tests/functional/helper_functions.bash:tests/functional/helper_functions.bash] \
		[tests/functional/launcher/extrae.xml:tests/functional/launcher/extrae.xml] \
		[tests/functional/launcher/extrae-nothing-to-instrument.xml:tests/functional/launcher/extrae-nothing-to-instrument.xml] \
		[tests/functional/tracer/CUDA/trace.sh:tests/functional/tracer/CUDA/trace.sh] \
		[tests/functional/tracer/CUDA/extrae.xml:tests/functional/tracer/CUDA/extrae.xml] \
		[tests/functional/tracer/CUDA/hello.cu:tests/functional/tracer/CUDA/hello.cu] \
		[tests/functional/tracer/MPI/trace-static.sh:tests/functional/tracer/MPI/trace-static.sh] \
		[tests/functional/tracer/MPI/trace-ldpreload.sh:tests/functional/tracer/MPI/trace-ldpreload.sh] \
		[tests/functional/tracer/MPI/trace-ldpreloadf.sh:tests/functional/tracer/MPI/trace-ldpreloadf.sh] \
		[tests/functional/tracer/MPI/extrae.xml:tests/functional/tracer/MPI/extrae.xml] \
		[tests/functional/tracer/PTHREAD/trace-ldpreload.sh:tests/functional/tracer/PTHREAD/trace-ldpreload.sh] \
		[tests/functional/tracer/PTHREAD/extrae.xml:tests/functional/tracer/PTHREAD/extrae.xml] \
		[tests/functional/tracer/JAVA/JavaSimple.java:tests/functional/tracer/JAVA/JavaSimple.java] \
		[tests/functional/tracer/JAVA/extrae_JavaSimple.sh:tests/functional/tracer/JAVA/extrae_JavaSimple.sh] \
		[tests/functional/tracer/JAVA/JavaGC.java:tests/functional/tracer/JAVA/JavaGC.java] \
		[tests/functional/tracer/JAVA/extrae_JavaGC.sh:tests/functional/tracer/JAVA/extrae_JavaGC.sh] \
		[tests/functional/tracer/JAVA/JavaException.java:tests/functional/tracer/JAVA/JavaException.java] \
		[tests/functional/tracer/JAVA/extrae_JavaException.sh:tests/functional/tracer/JAVA/extrae_JavaException.sh] \
		[tests/functional/tracer/JAVA/JavaThreads.java:tests/functional/tracer/JAVA/JavaThreads.java] \
		[tests/functional/tracer/JAVA/extrae_JavaThreads.sh:tests/functional/tracer/JAVA/extrae_JavaThreads.sh] \
		[tests/functional/tracer/JAVA/extrae.xml:tests/functional/tracer/JAVA/extrae.xml] \
		[tests/functional/merger/dump-events/dump-events.reference:tests/functional/merger/dump-events/dump-events.reference] \
		[tests/functional/hw-counters/extrae-PAPI_TOT_CYC.xml:tests/functional/hw-counters/extrae-PAPI_TOT_CYC.xml] \
		[tests/functional/hw-counters/extrae-PAPI_TOT_INS.xml:tests/functional/hw-counters/extrae-PAPI_TOT_INS.xml] \
		[tests/functional/hw-counters/extrae-PAPI_TOT_INS_CYC.xml:tests/functional/hw-counters/extrae-PAPI_TOT_INS_CYC.xml] \
		[tests/functional/xml/extrae_envvar_merge.xml:tests/functional/xml/extrae_envvar_merge.xml] \
		[tests/functional/xml/extrae_envvar_counters.xml:tests/functional/xml/extrae_envvar_counters.xml] \
		[tests/functional/merger/shared-libraries/main.c:tests/functional/merger/shared-libraries/main.c] \
		[tests/functional/merger/shared-libraries/fA.c:tests/functional/merger/shared-libraries/fA.c] \
		[tests/functional/merger/shared-libraries/fB.c:tests/functional/merger/shared-libraries/fB.c] \
		[etc/extrae.sh:etc/extrae.sh] \
		[example/LINUX/MPI/detailed_trace_basic.xml:example/LINUX/MPI/detailed_trace_basic.xml] \
		[example/LINUX/MPI/extrae.xml:example/LINUX/MPI/extrae.xml] \
		[example/LINUX/MPI/extrae_bursts_1ms.xml:example/LINUX/MPI/extrae_bursts_1ms.xml] \
		[example/LINUX/MPI/extrae_explained.xml:example/LINUX/MPI/extrae_explained.xml] \
		[example/LINUX/MPI/summarized_trace_basic.xml:example/LINUX/MPI/summarized_trace_basic.xml] \
		[src/others/pyextrae/multiprocessing/__init__.py:src/others/pyextrae/multiprocessing/__init__.py:] \
		[src/others/pyextrae/__init__.py:src/others/pyextrae/__init__.py] \
		[src/others/pyextrae/sequential/__init__.py:src/others/pyextrae/sequential/__init__.py] \
		[src/others/pyextrae/mpi/__init__.py:src/others/pyextrae/mpi/__init__.py] \
		[src/others/pyextrae/common/extrae.py.in:src/others/pyextrae/common/extrae.py.in] \
		[src/others/pyextrae/common/__init__.py:src/others/pyextrae/common/__init__.py]
		[docs/source/conf.py:docs/source/conf.py] \
		[docs/source/index.rst:docs/source/index.rst] \
		[docs/source/quick-guide.rst:docs/source/quick-guide.rst] \
		[docs/source/introduction.rst:docs/source/introduction.rst] \
		[docs/source/configure-installation.rst:docs/source/configure-installation.rst] \
		[docs/source/xml.rst:docs/source/xml.rst] \
		[docs/source/api.rst:docs/source/api.rst] \
		[docs/source/merge.rst:docs/source/merge.rst] \
		[docs/source/online.rst:docs/source/online.rst] \
		[docs/source/examples.rst:docs/source/examples.rst] \
		[docs/source/wholeXML.rst:docs/source/wholeXML.rst] \
		[docs/source/envvars.rst:docs/source/envvars.rst] \
		[docs/source/pnmpi.rst:docs/source/pnmpi.rst] \
		[docs/source/regression-tests.rst:docs/source/regression-tests.rst] \
		[docs/source/overhead.rst:docs/source/overhead.rst] \
		[docs/source/FAQ.rst:docs/source/FAQ.rst] \
		[docs/source/submit-bug.rst:docs/source/submit-bug.rst] \
		[docs/source/instrumented-routines.rst:docs/source/instrumented-routines.rst] \
		[docs/source/see-also.rst:docs/source/see-also.rst] \
		[docs/source/man/UFlist.sh.1.rst:docs/source/man/UFlist.sh.1.rst] \
		[docs/source/man/extrae_event.3.rst:docs/source/man/extrae_event.3.rst] \
		[docs/source/man/mpi2prv.1.rst:docs/source/man/mpi2prv.1.rst] \
		[docs/source/xml/config.xml:docs/source/xml/config.xml] \
		[docs/source/xml/mpi.xml:docs/source/xml/mpi.xml] \
		[docs/source/xml/pthread.xml:docs/source/xml/pthread.xml] \
		[docs/source/xml/openmp.xml:docs/source/xml/openmp.xml] \
		[docs/source/xml/callers.xml:docs/source/xml/callers.xml] \
		[docs/source/xml/userfunctions.xml:docs/source/xml/userfunctions.xml] \
		[docs/source/xml/counters.xml:docs/source/xml/counters.xml] \
		[docs/source/xml/storage.xml:docs/source/xml/storage.xml] \
		[docs/source/xml/buffer.xml:docs/source/xml/buffer.xml] \
		[docs/source/xml/trace-control.xml:docs/source/xml/trace-control.xml] \
		[docs/source/xml/bursts.xml:docs/source/xml/bursts.xml] \
		[docs/source/xml/others.xml:docs/source/xml/others.xml] \
		[docs/source/xml/sampling.xml:docs/source/xml/sampling.xml] \
		[docs/source/xml/cuda.xml:docs/source/xml/cuda.xml] \
		[docs/source/xml/opencl.xml:docs/source/xml/opencl.xml] \
		[docs/source/xml/input-output.xml:docs/source/xml/input-output.xml] \
		[docs/source/xml/dynamic-memory.xml:docs/source/xml/dynamic-memory.xml] \
		[docs/source/xml/intel-pebs.xml:docs/source/xml/intel-pebs.xml] \
		[docs/source/xml/merge.xml:docs/source/xml/merge.xml] \
		[docs/source/xml/merge-reduced.xml:docs/source/xml/merge-reduced.xml] \
		[docs/source/xml/extrae.xml:docs/source/xml/extrae.xml] \
		[docs/source/overheads/overheads.eps:docs/source/overheads/overheads.eps] \
		[include/extrae_version.h.in:include/extrae_version.h.in]
	)

#########################################
#   Defines source and build path vars  #
#########################################
AC_SUBST( [INCLUDE_DIR], ['$(top_srcdir)/include'] )
AC_SUBST( [SRC_DIR], ['$(top_srcdir)/src'] )

AC_SUBST( [COMMON_INC], ['$(top_srcdir)/src/common'] )
AC_SUBST( [COMMON_MPI_INC], ['$(top_srcdir)/src/common/MPI'] )
AC_SUBST( [TRACER_INC], ['$(top_srcdir)/src/tracer'] )
AC_SUBST( [MERGER_INC], ['$(top_srcdir)/src/merger'] )
AC_SUBST( [CLOCKS_INC], ['$(TRACER_INC)/clocks'] )
AC_SUBST( [HWC_INC], ['$(TRACER_INC)/hwc'] )
AC_SUBST( [STATS_INC], ['$(TRACER_INC)/stats'] )
AC_SUBST( [WRAPPERS_INC], ['$(TRACER_INC)/wrappers'] )
AC_SUBST( [INTERFACES_INC], ['$(TRACER_INC)/interfaces'] )
AC_SUBST( [PROBES_INC], ['$(TRACER_INC)/probes'] )
AC_SUBST( [ONLINE_INC], ['$(TRACER_INC)/online'] )
AC_SUBST( [PRV_MERGER_INC], ['$(MERGER_INC)/paraver'] )
AC_SUBST( [TRF_MERGER_INC], ['$(MERGER_INC)/dimemas'] )
AC_SUBST( [BUFFERS_INC], ['$(WRAPPERS_INC)/API'] )

AC_SUBST( [COMMON_LIB], ['$(top_builddir)/src/common'] )
AC_SUBST( [COMMON_MPI_LIB], ['$(top_builddir)/src/common/MPI'] )
AC_SUBST( [TRACER_LIB], ['$(top_builddir)/src/tracer'] )
AC_SUBST( [MERGER_LIB], ['$(top_builddir)/src/merger'] )
AC_SUBST( [CLOCKS_LIB], ['$(TRACER_LIB)/clocks'] )
AC_SUBST( [HWC_LIB], ['$(TRACER_LIB)/hwc'] )
AC_SUBST( [STATS_LIB], ['$(TRACER_LIB)/stats'] )
AC_SUBST( [WRAPPERS_LIB], ['$(TRACER_LIB)/wrappers'] )
AC_SUBST( [INTERFACES_LIB], ['$(TRACER_LIB)/interfaces'] )
AC_SUBST( [PROBES_LIB], ['$(TRACER_LIB)/probes'] )
AC_SUBST( [ONLINE_LIB], ['$(TRACER_LIB)/online'] )
AC_SUBST( [PRV_MERGER_LIB], ['$(MERGER_LIB)/paraver'] )
AC_SUBST( [TRF_MERGER_LIB], ['$(MERGER_LIB)/dimemas'] )
AC_SUBST( [BUFFERS_LIB], ['$(WRAPPERS_LIB)/API'] )

AC_SUBST( [EXTRAE_MAJOR], extrae_major )
AC_SUBST( [EXTRAE_MINOR], extrae_minor )
AC_SUBST( [EXTRAE_MICRO], extrae_micro )

AC_OUTPUT

AX_SHOW_CONFIGURATION

echo ${prefix} > PREFIX
