
# Process this file with autoconf to produce a configure script.

#########################
#    Initializations    #
#########################

# Initialize autoconf & define package name, version and bug-report address 
AC_INIT(MPItrace, 1.0, tools@bsc.es)

# Safety check to ensure that the directory told with `--srcdir` contains the source code
#AC_CONFIG_SRCDIR(src/mpitrace.c)

# GNU Autotools intermediate files are stored in the following directory
AC_CONFIG_AUX_DIR(config)

# Loads some shell variables like host_cpu and host_os, to get the host information 
AC_CANONICAL_SYSTEM

# Write defines in the output header file for the architecture and operating system
case "${target_cpu}" in
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
esac

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
AM_CONDITIONAL(OS_SOLARIS,   test "${OperatingSystem}" = "solaris" )

if test "${OperatingSystem}" = "linux" ; then
	if test "${Architecture}" = "ia64" ; then
		GenerarLlibreriaOMP="yes"
	elif test "${Architecture}" = "powerpc" ; then
		GenerarLlibreriaOMP="yes"
	else
		GenerarLlibreriaOMP="no"
	fi
fi

if test "${OperatingSystem}" = "freebsd" ; then
	CFLAGS="${CFLAGS} -I/usr/local/include"
fi

# Initialize automake
AM_INIT_AUTOMAKE
AM_PROG_MKDIR_P

# Specify the output configuration header file
AM_CONFIG_HEADER(config.h)

#############################
#    Checks for programs    #
#############################

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

AX_IS_BGL_MACHINE
AX_IS_BGP_MACHINE
if test "${IS_BGL_MACHINE}" = "yes" -o "${IS_BGP_MACHINE}" = "yes" ; then
  AC_DEFINE([IS_BG_MACHINE], 1, [Defined if this machine is a BG machine])
fi
AM_CONDITIONAL(IS_BG_MACHINE, test "${IS_BGL_MACHINE}" = "yes" -o "${IS_BGP_MACHINE}" = "yes")

AX_IS_CELL_MACHINE
AX_IS_MN_MACHINE

# Search for available compilers and preprocessor
# AC_PROG_CC does not let us give it a absolute path with the compiler
if test "${IS_BGL_MACHINE}" = "yes" -o "${IS_BGP_MACHINE}" = "yes" ; then
#   wanted_C_Compilers="powerpc-bgl-blrts-gnu-gcc ${BG_HOME}/blrts-gnu/bin/powerpc-bgl-blrts-gnu-gcc blrts_xlc /usr/bin/blrts_xlc"
#   wanted_CXX_Compilers="powerpc-bgl-blrts-gnu-g++ ${BG_HOME}/blrts-gnu/bin/powerpc-bgl-blrts-gnu-g++ blrts_xlC /usr/bin/blrts_xlC"

   if test "${CC}" ; then
      AC_MSG_NOTICE([Attention!])
      AC_MSG_NOTICE([Attention! CC is defined on a BG/L system. Use it with caution!])
      AC_MSG_NOTICE([Attention!])
   else
      if test "${IS_BGL_MACHINE}" = "yes" ; then
         CC="${BG_HOME}/blrts-gnu/bin/powerpc-bgl-blrts-gnu-gcc"
    	 CXX="${BG_HOME}/blrts-gnu/bin/powerpc-bgl-blrts-gnu-g++"
      elif test "${IS_BGP_MACHINE}" = "yes" ; then
         CC="${BG_HOME}/gnu-linux/bin/powerpc-bgp-linux-gcc"
         CXX="${BG_HOME}/gnu-linux/bin/powerpc-bgp-linux-g++"
      fi
   fi
else
   wanted_C_Compilers="gcc xlc cc"
   wanted_CXX_Compilers="g++ xlC CC"
fi
AC_PROG_CC(${wanted_C_Compilers})
AC_PROG_CXX(${wanted_CXX_Compilers})
AC_PROG_CPP
AC_PROG_INSTALL
AC_PROG_SED
AC_PROG_AWK

# Automake 1.10 reports a problem if this is not defined
AM_PROG_CC_C_O

# Check for /proc/cpuinfo & /proc/meminfo
AX_CHECK_PROC_CPUINFO
AX_CHECK_PROC_MEMINFO

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
 [execinfo.h stdint.h netdb.h errno.h string.h strings.h unistd.h rts.h \
  byteswap.h limits.h malloc.h stdint.h stdio.h stdlib.h values.h assert.h \
  bgl_perfctr.h bgl_perfctr_events.h ctype.h dlfcn.h excpt.h fcntl.h getopt.h \
  libgen.h libspe.h libspe2.h pdsc.h signal.h stdarg.h math.h inttypes.h time.h \
  ucontext.h pthread.h semaphore.h]
)
AC_CHECK_HEADERS(
  [sys/types.h sys/socket.h sys/utsname.h sys/wait.h sys/resource.h \
   sys/sysctl.h sys/time.h sys/stat.h sys/procfs.h sys/mman.h sys/ioctl.h \
   sys/file.h sys/endian.h sys/systeminfo.h sys/uio.h]
)
AC_CHECK_HEADERS(
  [asm-ppc/atomic.h asm-ppc64/atomic.h asm/msr.h]
)
AC_CHECK_HEADERS(
  [linux/limits.h linux/mmtimer.h]
)
AC_CHECK_HEADERS(
  [netinet/in.h]
)

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
      [Build the parallel mergers (mpimpi2prv/mpimpi2dim) based on MPI]
   ),
   [enable_parallel_merge="${enableval}"],
   [enable_parallel_merge="yes"]
)
AM_CONDITIONAL(WANT_PARALLEL_MERGE, test "${enable_parallel_merge}" = "yes" )

if test "${Architecture}" = "ia64" -a "${OperatingSystem}" = "linux" ; then
   AX_CHECK_UNWIND
fi

AX_CHECK_LIBZ

AX_PROG_GM
AX_PROG_MX
AX_PROG_DYNINST
AX_PROG_PAPI
AX_PROG_PMAPI
AX_PROG_COUNTERS
AX_PROG_MRNET

AX_CHECK_LOAD_BALANCING

#######################################
# Enable/Disable some tracing options #
#######################################
AC_ARG_ENABLE(openmp,
   AC_HELP_STRING(
      [--enable-openmp],
      [Enable support for tracing OpenMP -IBM runtime- (disabled by default)]
   ),
   [enable_openmp="${enableval}"],
   [enable_openmp="no"]
)
if test "${enable_openmp}" = "yes" ; then
#   if test "${GenerarLlibreriaOMP}" = "no" ; then
#      AC_MSG_WARN([OpenMP tracing is not supported on this system])
#      enable_openmp="no"
#   fi
#   AX_OPENMP([AC_DEFINE([OMP_SUPPORT], [1], [Defined if tracing must support OpenMP])])
   AX_OPENMP([])
fi
AM_CONDITIONAL(WANT_OPENMP, test "${enable_openmp}" = "yes" )

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
      [Enable support for tracing SMP-superscalar (disabled by default)]
   ),
   [enable_smpss="${enableval}"],
   [enable_smpss="no"]
)
AM_CONDITIONAL(WANT_SMPSS, test "${enable_smpss}" = "yes" )

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
      [Enable support for XML configuration (enabled by default - disabled on BG/L)]
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

AC_ARG_ENABLE(license,
   AC_HELP_STRING(
      [--enable-license],
      [Enable license management in the mpi2prv application (enabled by default)]
   ),
   [enable_license="${enableval}"],
   [enable_license="yes"]
)
if test "${enable_license}" = "yes" ; then
   if test "${OperatingSystem}" = "freebsd" ; then
      AC_MSG_WARN([We don't support license system on FreeBSD systems])
   elif test "${IS_BGL_MACHINE}" = "yes" -o "${IS_BGP_MACHINE}" = "yes" ; then
      AC_MSG_WARN([We don't support license system on BG/L and BG/P systems])
   else
      AC_DEFINE([LICENSE], [1], [Defined if license management must be activated])
      AC_DEFINE([LICENSE_IN_MERGE], [1], [Defined if license management must be activated])
      AC_DEFINE([MPITRACE_LICENSE], [1], [Defined if license management must be activated])
   fi
fi

AC_ARG_ENABLE(doc,
   AC_HELP_STRING(
      [--enable-doc],
      [Generates the documentation of this instrumentation package (enabled by default)]
   ),
   [enable_doc="${enableval}"],
   [enable_doc="yes"]
)
AM_CONDITIONAL(GENERATE_DOCUMENTATION, test "${enable_doc}" = "yes" )

#################################
#    External library checks    #
#################################

AX_PROG_BFD

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

if test "${IS_BGL_MACHINE}" = "yes" -o "${IS_BGP_MACHINE}" = "yes" ; then
   cross_compiling="no"
fi

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

AC_CONFIG_FILES([Makefile src/Makefile include/Makefile manuals/Makefile \
  src/launcher/Makefile src/launcher/dyninst/Makefile scripts/Makefile \
  src/common/Makefile src/tracer/Makefile src/tracer/clocks/Makefile \
  src/tracer/interfaces/Makefile src/tracer/interfaces/API/Makefile \
  src/tracer/interfaces/MPI/Makefile src/tracer/interfaces/PACX/Makefile \
  src/tracer/hwc/Makefile \
  src/tracer/hwc/myrinet_hwc/Makefile \
  src/tracer/hwc/myrinet_hwc/GM/Makefile src/tracer/hwc/myrinet_hwc/MX/Makefile\
  src/tracer/spu/Makefile \
  src/merger/Makefile src/merger/parallel/Makefile src/others/Makefile  \
  doc/Makefile etc/Makefile])
if test "${MRNET_INSTALLED}" = "yes" ; then
AC_CONFIG_FILES([src/tracer/mrnet/Makefile])
fi
AC_OUTPUT

AX_SHOW_CONFIGURATION
