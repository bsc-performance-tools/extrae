/* mpichconf.h.  Generated automatically by configure.  */
/* mpichconf.h.in.  Generated automatically from .tmp by autoheader.  */

/* Define if on AIX 3.
   System headers sometimes define this.
   We just want to avoid a redefinition error message.  */
#ifndef _ALL_SOURCE
/* #undef _ALL_SOURCE */
#endif

/* Define if the `long double' type works.  */
#define HAVE_LONG_DOUBLE 1

/* Define if on MINIX.  */
/* #undef _MINIX */

/* Define if the system does not provide POSIX.1 features except
   with this defined.  */
/* #undef _POSIX_1_SOURCE */

/* Define if you need to in order for stat and other things to work.  */
/* #undef _POSIX_SOURCE */

/* Define if you have the ANSI C header files.  */
#define STDC_HEADERS 1

/* Define if your processor stores words with the most significant
   byte first (like Motorola and SPARC, unlike Intel and VAX).  */
#define WORDS_BIGENDIAN 1

/* Define is C supports volatile declaration */
#define HAS_VOLATILE 1

/* Define if XDR libraries available */
/* #undef HAS_XDR */

/* Define if message catalog programs available */
#define HAVE_GENCAT 1

/* Define if getdomainname function available */
/* #undef HAVE_GETDOMAINNAME */

/* Define in gethostbyname function available */
#define HAVE_GETHOSTBYNAME 1

/* Define if C has long long int */
#define HAVE_LONG_LONG_INT 1

/* Define if C does NOT support const */
/* #undef HAVE_NO_C_CONST */

/* Define if C supports prototypes (but isn't ANSI C) */
/* #undef HAVE_PROTOTYPES */

/* Define if C preprocessor does not accept ## for concatenation */
/* #undef OLD_STYLE_CPP_CONCAT */

/* Define if uname function available */
#define HAVE_UNAME 1

/* Define if an int is smaller than void * */
#define INT_LT_POINTER 1

/* Define if malloc returns void * (and is an error to return char *) */
#define MALLOC_RET_VOID 1

/* Define if MPE extensions are included in MPI libraries */
#define MPE_USE_EXTENSIONS 1

/* Define if MPID supports ADI collective */
/* #undef MPID_USE_ADI_COLLECTIVE */

/* Define is ADI should maintain a send queue for debugging */
/* #undef MPI_KEEP_SEND_QUEUE */

/* Define if mpe debug features should NOT be included */
/* #undef MPI_NO_MPEDBG */

/* Define if void * is 8 bytes */
#define POINTER_64_BITS 1

/* Define if stdarg can be used */
#define USE_STDARG 1

/* Define if oldstyle stdarg (one arg va_start) can be used */
/* #undef USE_OLDSTYLE_STDARG */

/* Define if stdarg.h exists */
#define HAVE_STDARG_H 1

/* For Cray, define two word character descriptors in use */
/* #undef _TWO_WORD_FCD */

/* Define if extra traceback information should be kept */
/* #undef DEBUG_TRACE */

/* Define if Fortran is NOT available */
/* #undef MPID_NO_FORTRAN */

/* Define if memory debugging should be enabled */
/* #undef MPIR_MEMDEBUG */

/* Define if object debugging should be enabled */
/* #undef MPIR_OBJDEBUG */

/* Define if ptr conversion debugging should be enabled */
/* #undef MPIR_PTRDEBUG */

/* Define if ADI is ADI-2 (required!) */
#define MPI_ADI2 1

/* Define if mmap does not work correctly for anonymous memory */
/* #undef HAVE_NO_ANON_MMAP */

/* Define if MPI Structs should align on 2 bytes */
/* #undef USE_BASIC_TWO_ALIGNMENT */

/* Define if MPI Structs should align on 4 bytes */
/* #undef USE_BASIC_FOUR_ALIGNMENT */

/* Define if MPI Structs should align on 8 bytes */
/* #undef USE_BASIC_EIGHT_ALIGNMENT */

/* Define if MPI Structs should align on the largest basic element */
#define USE_BASIC_ALIGNMENT 1

/* The number of processors expected on an SMP.  Usually undefined */
/* #undef PROCESSOR_COUNT */

/* Define this if weak symbols are supported */
#define HAVE_WEAK_SYMBOLS 1

/* Define this if the weak symbol support is pragma weak */
#define HAVE_PRAGMA_WEAK 1

/* Define this if the weak symbol support is pragma _HP_SECONDARY_DEF */
/* #undef HAVE_PRAGMA_HP_SEC_DEF */

/* Define this if the weak symbol support is pragma _CRI duplicate */
/* #undef HAVE_PRAGMA_CRI_DUP */

/* These provide information for initutil about the configuration options */
#define CONFIGURE_ARGS_CLEAN "--with-device=ch_mx -prefix=/home/bsc19/bsc19265/pub/Linux/MX-MPITRACE/64 -opt=-O -lib= -L/opt/osshpc/mx/lib64 -lmyriexpress -lpthread"
#define MPIRUN_MACHINE "ch_mx"
#define MPIRUN_DEVICE "ch_mx"

/* These enable tracking and output of debugging messages while a program is 
   running */
/* #undef USE_HOLD_LAST_DEBUG */
/* #undef USE_PRINT_LAST_ON_ERROR */

/* Define if there is a routine to print tracebacks */
/* #undef HAVE_PRINT_BACKTRACE */

/* Define for POSIX Clocks.  See the tests to see why these are here */
/* #undef HAVE_CLOCK_GETTIME */
/* #undef HAVE_CLOCK_GETRES */

/* Define if mpid/<device> provided an mpich-mpid.h header file (needed for
   mpid.h) */
#define HAVE_MPICH_MPID_H 1

/* Define if you have catclose.  */
#define HAVE_CATCLOSE 1

/* Define if you have catgets.  */
#define HAVE_CATGETS 1

/* Define if you have catopen.  */
#define HAVE_CATOPEN 1

/* Define if you have gethostname.  */
/* #undef HAVE_GETHOSTNAME */

/* Define if you have gethrtime.  */
/* #undef HAVE_GETHRTIME */

/* Define if you have mmap.  */
/* #undef HAVE_MMAP */

/* Define if you have nice.  */
#define HAVE_NICE 1

/* Define if you have sigaction.  */
#define HAVE_SIGACTION 1

/* Define if you have strdup.  */
#define HAVE_STRDUP 1

/* Define if you have sysinfo.  */
/* #undef HAVE_SYSINFO */

/* Define if you have system.  */
#define HAVE_SYSTEM 1

/* Define if you have the <memory.h> header file.  */
#define HAVE_MEMORY_H 1

/* Define if you have the <netdb.h> header file.  */
#define HAVE_NETDB_H 1

/* Define if you have the <nl_types.h> header file.  */
#define HAVE_NL_TYPES_H 1

/* Define if you have the <signal.h> header file.  */
#define HAVE_SIGNAL_H 1

/* Define if you have the <stdlib.h> header file.  */
#define HAVE_STDLIB_H 1

/* Define if you have the <string.h> header file.  */
#define HAVE_STRING_H 1

/* Define if you have the <sys/ioctl.h> header file.  */
#define HAVE_SYS_IOCTL_H 1

/* Define if you have the <sys/sockio.h> header file.  */
/* #undef HAVE_SYS_SOCKIO_H */

/* Define if you have the <sys/systeminfo.h> header file.  */
/* #undef HAVE_SYS_SYSTEMINFO_H */

/* Define if you have the <unistd.h> header file.  */
#define HAVE_UNISTD_H 1

/* Define if you have the nsl library (-lnsl).  */
/* #undef HAVE_LIBNSL */
