
# Process this file with autoconf to produce a configure script.

#########################
#    Initializations    #
#########################

# Initialize autoconf & define package name, version and bug-report address 
#  -- keep version in sync with src/common/extrae_version.h
AC_INIT(Extrae, 3.0rc4, tools@bsc.es)

# Safety check to ensure that the directory told with `--srcdir` contains the source code
#AC_CONFIG_SRCDIR(src/mpitrace.c)

# GNU Autotools intermediate files are stored in the following directory
AC_CONFIG_AUX_DIR([config])
AC_CONFIG_MACRO_DIR([config])

SVN=`which svn`

# If SVN command exists and .svn directory also exists, check for the versioning
if test $? -eq 0 -a -d .svn ; then
	${SVN} info . >& /dev/null
	if test $? -eq 0 ; then
		SVN_branch=`${SVN} info . | grep URL | cut -d/ -f 6-`
		SVN_branch_res=$?
		SVN_revision=`${SVN} info . | grep "Last Changed Rev:"  | cut -d: -f 2`
		SVN_revision_res=$?
		if test ${SVN_branch_res} -eq 0 -a ${SVN_revision_res} -eq 0 ; then
			echo ${SVN_branch} > SVN-branch
			echo ${SVN_revision} > SVN-revision
		fi
	fi
fi

if test -f SVN-branch -a -f SVN-revision ; then
	SVN_branch=`cat SVN-branch`
	SVN_revision=`cat SVN-revision`
else
	SVN_branch=unknown
	SVN_revision=0
fi
echo Configuring package ${PACKAGE_STRING} based on ${SVN_branch} revision ${SVN_revision}

AC_DEFINE_UNQUOTED([EXTRAE_SVN_REVISION], [${SVN_revision}], [Define SVN version for this Extrae])
AC_DEFINE_UNQUOTED([EXTRAE_SVN_BRANCH], ["${SVN_branch}"], [Define SVN branch for this Extrae])

# Loads some shell variables like host_cpu and host_os, to get the host information 
AC_CANONICAL_SYSTEM

# Write defines in the output header file for the architecture and operating system
case "${target_cpu}" in
  arm*)      Architecture="arm"
             AC_DEFINE([ARCH_ARM], [1], [Define if architecture is ARM]) ;;
  i*86|x86_64|amd64)
             Architecture="ia32"
             AC_DEFINE([ARCH_IA32], [1], [Define if architecture is IA32])
             if test "${target_cpu}" = "amd64" -o "${target_cpu}" = "x86_64" ; then
                AC_DEFINE([ARCH_IA32_x64], [1], [Define if architecture is IA32 (with 64bit extensions)])
             fi
             ;;
  powerpc* ) Architecture="powerpc"
             AC_DEFINE([ARCH_PPC], [1], [Define if architecture is PPC]) ;;
  ia64     ) Architecture="ia64"
             AC_DEFINE([ARCH_IA64], [1], [Define if architecture is IA64]) ;;
  alpha*   ) Architecture="alpha"
             AC_DEFINE([ARCH_ALPHA], [1], [Define if architecture is ALPHA]) ;;
  mips     ) Architecture="mips"
             AC_DEFINE([ARCH_MIPS], [1], [Define if architecture is MIPS]) ;;
esac

case "${target_os}" in
  linux*   ) OperatingSystem="linux"
             AC_DEFINE([OS_LINUX], [1], [Define if operating system is Linux]) ;;
  aix*     ) OperatingSystem="aix"
             AC_DEFINE([OS_AIX], [1], [Define if operating system is AIX]) ;;
  osf*     ) OperatingSystem="dec"
             AC_DEFINE([OS_DEC], [1], [Define if operating system is DEC]) ;;
  irix*    ) OperatingSystem="irix"
             AC_DEFINE([OS_IRIX], [1], [Define if operating system is IRIX]) ;;
  freebsd* ) OperatingSystem="freebsd"
             AC_DEFINE([OS_FREEBSD], [1], [Define if operating system is FreeBSD]) ;;
  solaris* ) OperatingSystem="solaris"
             AC_DEFINE([OS_SOLARIS], [1], [Define if operating system is Solaris]) ;;
  darwin*  ) OperatingSystem="darwin"
             AC_DEFINE([OS_DARWIN], [1], [Define if operating system is Darwin]) ;;
esac

# Initialize automake
AM_INIT_AUTOMAKE

dnl Enable silent rules (if these exist), use with make V=0  or V=1
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

dnl AM_PROG_MKDIR_P
AC_PROG_MKDIR_P

# Publish these defines for conditional compilation 
AM_CONDITIONAL(ARCH_IA32,    test "${Architecture}"    = "ia32"    )
AM_CONDITIONAL(ARCH_POWERPC, test "${Architecture}"    = "powerpc" )
AM_CONDITIONAL(ARCH_IA64,    test "${Architecture}"    = "ia64"    )
AM_CONDITIONAL(ARCH_ALPHA,   test "${Architecture}"    = "alpha"   )
AM_CONDITIONAL(ARCH_MIPS,    test "${Architecture}"    = "mips"    )
  
AM_CONDITIONAL(OS_LINUX,     test "${OperatingSystem}" = "linux"   )
AM_CONDITIONAL(OS_AIX,       test "${OperatingSystem}" = "aix"     )
AM_CONDITIONAL(OS_DEC,       test "${OperatingSystem}" = "dec"     )
AM_CONDITIONAL(OS_IRIX,      test "${OperatingSystem}" = "irix"    )
AM_CONDITIONAL(OS_FREEBSD,   test "${OperatingSystem}" = "freebsd" )
AM_CONDITIONAL(OS_DARWIN,    test "${OperatingSystem}" = "darwin" )
AM_CONDITIONAL(OS_SOLARIS,   test "${OperatingSystem}" = "solaris" )

if test "${OperatingSystem}" = "freebsd" ; then
	CFLAGS="${CFLAGS} -I/usr/local/include"
fi

# Specify the output configuration header file
AC_CONFIG_HEADER(config.h)

#############################
#    Checks for programs    #
#############################

AC_ARG_ENABLE(mic,
   AC_HELP_STRING(
      [--enable-mic],
      [Enable compilation for the Intel MIC architecture (disabled by default; needed when cross-compiling for Intel MIC/Xeon Phi)]
   ),
   [enable_mic="${enableval}"],
   [enable_mic="no"]
)
IS_MIC_MACHINE=${enable_mic}

AC_ARG_ENABLE(arm,
   AC_HELP_STRING(
      [--enable-arm],
      [Enable compilation for ARM architecture (disabled by default; needed when cross-compiling for ARM)]
   ),
   [enable_arm="${enableval}"],
   [enable_arm="no"]
)
IS_ARM_MACHINE=${enable_arm}

# Check if this is an Altix machine and if it has an /dev/mmtimer device
# (which is a global clock!)
AC_ARG_ENABLE(check-altix,
   AC_HELP_STRING(
      [--enable-check-altix],
      [Enable check to known if this is an Altix machine (enabled by default)]
   ),
   [enable_check_altix="${enableval}"],
   [enable_check_altix="yes"]
)
if test "${enable_check_altix}" = "yes" ; then
   AX_IS_ALTIX_MACHINE
   AX_HAVE_MMTIMER_DEVICE
fi

AX_IS_CRAY_XT
AX_IS_BGL_MACHINE
AX_IS_BGP_MACHINE
AX_IS_BGQ_MACHINE
if test "${IS_BGL_MACHINE}" = "yes" -o "${IS_BGP_MACHINE}" = "yes" -o "${IS_BGQ_MACHINE}" = "yes" ; then
  AC_DEFINE([IS_BG_MACHINE], 1, [Defined if this machine is a BG machine])
  IS_BG_MACHINE="yes"
fi
AM_CONDITIONAL(IS_BG_MACHINE, test "${IS_BGL_MACHINE}" = "yes" -o "${IS_BGP_MACHINE}" = "yes" -o "${IS_BGQ_MACHINE}" = "yes")
AX_IS_CELL_MACHINE

if test "${IS_CELL_MACHINE}" = "yes" ; then
   AC_ARG_ENABLE(spu-write,
      AC_HELP_STRING(
         [--enable-spu-write],
         [Enable direct write operations to disk from SPUs in CELL machines (disabled by default)]
      ),
      [enable_spu_write="${enableval}"],
      [enable_spu_write="no"]
    )
fi

if test "${IS_CELL_MACHINE}" = "yes" -a "${enable_spu_write}" = "yes" ; then
   AC_DEFINE([SPU_USES_WRITE], 1, [Defined if direct write operations to disk from SPUs are enabled])
   AC_MSG_NOTICE([SPUs will use write operations to flush the buffers])
elif test "${IS_CELL_MACHINE}" = "yes" -a "${enable_spu_write}" = "no" ; then
   AC_MSG_NOTICE([SPUs will not use write operations to flush the buffers])
fi
AM_CONDITIONAL(SPU_USES_WRITE, test "${IS_CELL_MACHINE}" = "yes" -a "${enable_spu_write}" = "yes")


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
else
   wanted_C_Compilers="gcc xlc icc cc"
   wanted_F_Compilers="gfortran xlf ifort f90"
   wanted_CXX_Compilers="g++ xlC icpc CC"
fi
AC_PROG_CC(${wanted_C_Compilers})
AC_PROG_CXX(${wanted_CXX_Compilers})
AC_PROG_CPP
AC_PROG_FC(${wanted_F_Compilers})
AX_JAVA
AC_PROG_INSTALL
AC_PROG_SED
AC_PROG_AWK

if test "${GCC}" = "yes" ; then
  CFLAGS="${CFLAGS} -fno-optimize-sibling-calls"
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

# Search for headers
AC_CHECK_HEADERS(
 [stdint.h netdb.h errno.h string.h strings.h unistd.h rts.h \
  byteswap.h limits.h malloc.h stdint.h stdio.h stdlib.h values.h assert.h \
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
   AX_CHECK_MPI_F_STATUS_IGNORE
fi
AX_ENABLE_SINGLE_MPI_LIBRARY
AX_CHECK_MPI_LIB_HAS_C_AND_FORTRAN_SYMBOLS

AX_PROG_PACX
if test "${PACX_INSTALLED}" = "yes" ; then
#   PACX and fortran?
#   AX_CHECK_PPACX_NAME_MANGLING 
   AX_CHECK_PACX_STATUS_SIZE
   AX_CHECK_PACX_SOURCE_OFFSET
   AX_CHECK_PACX_TAG_OFFSET
fi

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

AX_CHECK_LOAD_BALANCING

#AC_ARG_ENABLE(load-balancing,
#   AC_HELP_STRING(
#      [--enable-load-balancing],
#      [Enable experimental support for Dynamic Load Balancing (disabled by default)]
#   ),
#   [enable_load_balancing="${enableval}"],
#   [enable_load_balancing="no"]
#)
#AM_CONDITIONAL(GENERATE_LOAD_BALANCING, test "${enable_load_balancing}" = "yes" )

#######################################
# Enable/Disable some tracing options #
#######################################
AC_ARG_ENABLE(openmp,
   AC_HELP_STRING(
      [--enable-openmp],
      [Enable support for tracing OpenMP -IBM and GNU runtimes- (enabled by default, IBM only on PPC)]
   ),
   [enable_openmp="${enableval}"],
   [enable_openmp="yes"]
)
if test "${enable_openmp}" = "yes" ; then
   AX_OPENMP()
fi
AM_CONDITIONAL(WANT_OPENMP, test "${enable_openmp}" = "yes" )

AX_CUDA
AX_CUPTI
AX_OPENCL

AC_ARG_ENABLE(pthread-support-in-all-libs,
   AC_HELP_STRING(
      [--enable-pthread-support-in-all-libs],
      [Allows all the instrumentation libraries to work with pthreads. Caution! May add dependencies with pthread library (disabled by default)]
   ),
   [enable_pthread_in_all_libs="${enableval}"],
   [enable_pthread_in_all_libs="no"]
)
AM_CONDITIONAL(PTHREAD_SUPPORT_IN_ALL_LIBS, test "${enable_pthread_in_all_libs}" = "yes" )

AC_ARG_ENABLE(smpss,
   AC_HELP_STRING(
      [--enable-smpss],
      [Enable support for tracing SMP-superscalar (enabled by default)]
   ),
   [enable_smpss="${enableval}"],
   [enable_smpss="yes"]
)
AM_CONDITIONAL(WANT_SMPSS, test "${enable_smpss}" = "yes" )

AC_ARG_ENABLE(upc, 
   AC_HELP_STRING(
      [--enable-upc],
      [Enable support for tracing UPC run-time (disabled by default)]
   ),
   [enable_upc="${enableval}"],
   [enable_upc="no"]
)
AM_CONDITIONAL(WANT_UPC, test "${enable_upc}" = "yes" )

AC_ARG_ENABLE(nanos, 
   AC_HELP_STRING(
      [--enable-nanos],
      [Enable support for tracing Nanos run-time (enabled by default)]
   ),
   [enable_nanos="${enableval}"],
   [enable_nanos="yes"]
)
AM_CONDITIONAL(WANT_NANOS, test "${enable_nanos}" = "yes" )

AC_ARG_ENABLE(trt,
   AC_HELP_STRING(
      [--enable-trt],
      [Enable support for tracing Intel TRT library (disabled by default)]
   ),
   [enable_trt="${enableval}"],
   [enable_trt="no"]
)
AM_CONDITIONAL(WANT_TRT, test "${enable_trt}" = "yes" )

AC_ARG_ENABLE(pthread,
   AC_HELP_STRING(
      [--enable-pthread],
      [Enable support for tracing pthread library (enabled by default)]
   ),
   [enable_pthread="${enableval}"],
   [enable_pthread="yes"]
)
if test "${enable_pthread}" = "yes" ; then
	ACX_PTHREAD([],[AC_MSG_ERROR([Unable to determine pthread library support])])
fi
AM_CONDITIONAL(WANT_PTHREAD, test "${enable_pthread}" = "yes" )

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
      [Generates the documentation of this instrumentation package (enabled by default)]
   ),
   [enable_doc="${enableval}"],
   [enable_doc="yes"]
)
AM_CONDITIONAL(GENERATE_DOCUMENTATION, test "${enable_doc}" = "yes" )

AC_ARG_ENABLE(instrument-io,
	AC_HELP_STRING(
		[--enable-instrument-io],
		[Enables instrumentation for basic I/O (read, write) calls  (experimental, disabled by default)]
	),
	[enable_instrument_io="${enableval}"],
	[enable_instrument_io="no"]
)
if test "${enable_instrument_io}" = "yes" ; then
	AC_DEFINE([INSTRUMENT_IO], [1], [Determine whether read/write are meant to be instrumented])
fi

AC_ARG_ENABLE(instrument-dynamic-memory,
	AC_HELP_STRING(
		[--enable-instrument-dynamic-memory],
		[Enables instrumentation for dynamic memory (malloc, free, realloc) calls (experimental, disabled by default)]
	),
	[enable_instrument_dynamic_memory="${enableval}"],
	[enable_instrument_dynamic_memory="no"]
)
if test "${enable_instrument_dynamic_memory}" = "yes" ; then
	AC_DEFINE([INSTRUMENT_DYNAMIC_MEMORY], [1], [Determine whether dynamic memory calls are meant to be instrumented])
fi

AC_ARG_ENABLE(bfd-generate-addresses,
	AC_HELP_STRING(
		[--enable-bfd-generate-addresses],
		[Generates information for data objects found in binaries (experimental, disabled by default)]
	),
	[enable_bfd_generate_addresses="${enableval}"],
	[enable_bfd_generate_addresses="no"]
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

#################################
#    External library checks    #
#################################

AX_PROG_BINUTILS

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
AX_CFLAGS_WARN_ALL
AX_CXXFLAGS_WARN_ALL
#AX_CFLAGS_WARN_ALL_ANSI
#AX_CXXFLAGS_WARN_ALL_ANSI

# Test whether the compiler accepts function inlining
AC_C_INLINE

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
  include/Makefile manuals/Makefile \
  src/launcher/Makefile \
  src/launcher/static/Makefile \
  src/launcher/dyninst/Makefile \
  scripts/Makefile \
  src/common/Makefile \
  src/tracer/Makefile \
  src/tracer/clocks/Makefile \
  src/tracer/interfaces/Makefile 
  src/tracer/interfaces/API/Makefile \
  src/tracer/interfaces/MPI/Makefile \
  src/tracer/interfaces/PACX/Makefile \
  src/tracer/stats/Makefile \
  src/tracer/hwc/Makefile \
  src/tracer/spu/Makefile \
  src/merger/Makefile \
  src/merger/parallel/Makefile \
  src/others/Makefile \
  src/cmd-line/Makefile \
  src/java-connector/Makefile \
  src/java-connector/jni/Makefile \
  doc/Makefile etc/Makefile ])

AC_CONFIG_FILES([\
  tests/Makefile
  tests/src/Makefile \
  tests/src/common/Makefile \
  tests/src/tracer/Makefile \
  tests/src/tracer/clocks/Makefile \
  tests/functional/Makefile \
  tests/functional/launcher/Makefile \
  tests/functional/tracer/Makefile \
  tests/functional/tracer/OPENCL/Makefile \
  tests/functional/tracer/CUDA/Makefile \
  tests/functional/merger/Makefile \
  tests/functional/merger/dump-events/Makefile \
  tests/functional/merger/shared-libraries/Makefile \
  tests/overhead/Makefile])
AC_CONFIG_FILES([src/tracer/online/Makefile src/tracer/online/extractors/Makefile])
### TEMPORARILY_DISABLED
### AC_CONFIG_FILES([src/tracer/hwc/myrinet_hwc/Makefile src/tracer/hwc/myrinet_hwc/GM/Makefile src/tracer/hwc/myrinet_hwc/MX/Makefile])
AC_OUTPUT

AX_SHOW_CONFIGURATION

echo ${prefix} > PREFIX
