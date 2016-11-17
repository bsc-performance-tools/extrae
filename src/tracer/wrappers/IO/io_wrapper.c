/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                   Extrae                                  *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *     ___     This library is free software; you can redistribute it and/or *
 *    /  __         modify it under the terms of the GNU LGPL as published   *
 *   /  /  _____    by the Free Software Foundation; either version 2.1      *
 *  /  /  /     \   of the License, or (at your option) any later version.   *
 * (  (  ( B S C )                                                           *
 *  \  \  \_____/   This library is distributed in hope that it will be      *
 *   \  \__         useful but WITHOUT ANY WARRANTY; without even the        *
 *    \___          implied warranty of MERCHANTABILITY or FITNESS FOR A     *
 *                  PARTICULAR PURPOSE. See the GNU LGPL for more details.   *
 *                                                                           *
 * You should have received a copy of the GNU Lesser General Public License  *
 * along with this library; if not, write to the Free Software Foundation,   *
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA          *
 * The GNU LEsser General Public License is contained in the file COPYING.   *
 *                                 ---------                                 *
 *   Barcelona Supercomputing Center - Centro Nacional de Supercomputacion   *
\*****************************************************************************/

#include "common.h"

/*
 * __USE_FILE_OFFSET64
 *
 * Extrae is compiled by default with this flag to support large files.
 * When defined, some I/O calls such as preadv, pwritev... are renamed
 * automatically by the compiler to their 64-bit versions preadv64, pwritev64...
 *
 * This file needs to be compiled without this flag in order to be able to
 * write both wrappers without the compiler changing their names automatically
 * from the 32-bit version into the 64-bit.
 *
 */
#ifdef __USE_FILE_OFFSET64
# undef __USE_FILE_OFFSET64
#endif

#if HAVE_STDIO_H
# include <stdio.h>
#endif
#if HAVE_DLFCN_H
# define __USE_GNU
# include <dlfcn.h>
# undef __USE_GNU
#endif
#if HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_SYS_UIO_H
# include <sys/uio.h>
#endif
#ifdef HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif
#ifdef HAVE_SYS_STAT_H
# include <sys/stat.h>
#endif
#ifdef HAVE_FCNTL_H
# include <fcntl.h>
#endif
#ifdef HAVE_STDARG_H
# include <stdarg.h>
#endif
#ifdef HAVE_ERRNO_H
/*
 * We need errno to save its value before entering the IO wrappers and restore
 * it before leaving, as it may change during the execution of the wrapper,
 * which causes conflicts with applications that check it (e.g. NetCDF)
 */
# include <errno.h>
#endif

#include "io_wrapper.h"
#include "io_probe.h"
#include "wrapper.h"

#if defined(INSTRUMENT_IO)

//#define DEBUG

/***************************************************************************************\
 * This file contains wrappers to instrument the I/O system calls (read, write, etc.). *
 * The interposition of these wrappers require a shared library (-DPIC).               *
 * Currently, there's a known issue: An incompatibility with the GNU OpenMP runtime.   *
 * This is just a guess, but the problem is likely to be that the OpenMP runtime calls *
 * read() or some other I/O routine before the runtime itself is initialized, and when *
 * we try to emit the events and request to identify the thread through                *
 * omp_get_thread_num, that results in a crash or returns some invalid value. Just a   *
 * guess, we'll have to check when we see this problem again.                          *
\***************************************************************************************/

/* Global pointers to the real implementation of the OS I/O calls */
static int     (*real_open)(const char *pathname, int flags, ...)                               = NULL;
static int     (*real_open64)(const char *pathname, int flags, ...)                             = NULL;
static FILE *  (*real_fopen)(const char *path, const char *mode)                                = NULL;
static FILE *  (*real_fopen64)(const char *path, const char *mode)                              = NULL;

static ssize_t (*real_read)(int fd, void *buf, size_t count)                                    = NULL;
static ssize_t (*real_write)(int fd, const void *buf, size_t count)                             = NULL;

static size_t  (*real_fread)(void *ptr, size_t size, size_t nmemb, FILE *stream)                = NULL;
static size_t  (*real_fwrite)(const void *ptr, size_t size, size_t nmemb, FILE *stream)         = NULL;

static ssize_t (*real_pread)(int fd, void *buf, size_t count, off_t offset)                     = NULL;
static ssize_t (*real_pwrite)(int fd, const void *buf, size_t count, off_t offset)              = NULL;

static ssize_t (*real_readv)(int fd, const struct iovec *iov, int iovcnt)                       = NULL;
static ssize_t (*real_writev)(int fd, const struct iovec *iov, int iovcnt)                      = NULL;
static ssize_t (*real_preadv)(int fd, const struct iovec *iov, int iovcnt, off_t offset)        = NULL;
static ssize_t (*real_preadv64)(int fd, const struct iovec *iov, int iovcnt, off_t offset)      = NULL;
static ssize_t (*real_pwritev)(int fd, const struct iovec *iov, int iovcnt, off_t offset)       = NULL;
static ssize_t (*real_pwritev64)(int fd, const struct iovec *iov, int iovcnt, __off64_t offset) = NULL;

/**
 * Extrae_iotrace_init
 *
 * Initialization routine for the I/O tracing module. Performs a discovery of the
 * address of the real implementation of the I/O calls through dlsym. The initialization
 * is deferred until any of the instrumented symbols is used for the first time.
 */
void Extrae_iotrace_init (void)
{
# if defined(PIC) /* Only available for .so libraries */

  /*
   * Find the first implementation of the I/O calls in the default library search order
   * after the current library. Not finding any of the symbols doesn't throw an error
   * unless the application tries to use it later.
   */
  real_open      = (int(*)(const char *, int, ...)) dlsym(RTLD_NEXT, "open");
  real_open64    = (int(*)(const char *, int, ...)) dlsym(RTLD_NEXT, "open64");
  real_fopen     = (FILE *(*)(const char *, const char *)) dlsym(RTLD_NEXT, "fopen");
  real_fopen64   = (FILE *(*)(const char *, const char *)) dlsym(RTLD_NEXT, "fopen64");

  real_read      = (ssize_t(*)(int, void*, size_t)) dlsym (RTLD_NEXT, "read");
  real_write     = (ssize_t(*)(int, const void*, size_t)) dlsym (RTLD_NEXT, "write");

  real_fread     = (size_t(*)(void *, size_t, size_t, FILE *)) dlsym (RTLD_NEXT, "fread");
  real_fwrite    = (size_t(*)(const void *, size_t, size_t, FILE *)) dlsym (RTLD_NEXT, "fwrite");

  real_pread     = (ssize_t(*)(int fd, void *buf, size_t count, off_t offset)) dlsym (RTLD_NEXT, "pread");
  real_pwrite    = (ssize_t(*)(int fd, const void *buf, size_t count, off_t offset)) dlsym (RTLD_NEXT, "pwrite");

  real_readv     = (ssize_t(*)(int, const struct iovec *, int)) dlsym (RTLD_NEXT, "readv");
  real_writev    = (ssize_t(*)(int, const struct iovec *, int)) dlsym (RTLD_NEXT, "writev");
  real_preadv    = (ssize_t(*)(int, const struct iovec *, int, off_t)) dlsym (RTLD_NEXT, "preadv");
  real_preadv64  = (ssize_t(*)(int, const struct iovec *, int, __off64_t)) dlsym (RTLD_NEXT, "preadv64");
  real_pwritev   = (ssize_t(*)(int, const struct iovec *, int, off_t)) dlsym (RTLD_NEXT, "pwritev");
  real_pwritev64 = (ssize_t(*)(int, const struct iovec *, int, __off64_t)) dlsym (RTLD_NEXT, "pwritev64");

#  if defined(DEBUG)
  fprintf(stderr, "[DEBUG] Extrae installed the following I/O hooks:\n");
  fprintf(stderr, "[DEBUG] open hooked at %p\n", real_open);
  fprintf(stderr, "[DEBUG] open64 hooked at %p\n", real_open64);
  fprintf(stderr, "[DEBUG] fopen hooked at %p\n", real_fopen);
  fprintf(stderr, "[DEBUG] fopen64 hooked at %p\n", real_fopen64);
  fprintf(stderr, "[DEBUG] read hooked at %p\n", real_read);
  fprintf(stderr, "[DEBUG] write hooked at %p\n", real_write);
  fprintf(stderr, "[DEBUG] fread hooked at %p\n", real_fread);
  fprintf(stderr, "[DEBUG] fwrite hooked at %p\n", real_fwrite);
  fprintf(stderr, "[DEBUG] pread hooked at %p\n", real_pread);
  fprintf(stderr, "[DEBUG] pwrite hooked at %p\n", real_pwrite);
  fprintf(stderr, "[DEBUG] readv hooked at %p\n", real_readv);
  fprintf(stderr, "[DEBUG] writev hooked at %p\n", real_writev);
  fprintf(stderr, "[DEBUG] preadv hooked at %p\n", real_preadv);
  fprintf(stderr, "[DEBUG] preadv64 hooked at %p\n", real_preadv64);
  fprintf(stderr, "[DEBUG] pwritev hooked at %p\n", real_pwritev);
  fprintf(stderr, "[DEBUG] pwritev64 hooked at %p\n", real_pwritev64);
#  endif

# else

  fprintf (stderr, PACKAGE_NAME": Warning! I/O instrumentation requires linking with shared library!\n");

# endif
}

# if defined(PIC) /* Only available for .so libraries */

/**
 * open
 *
 * Wrapper for the system call 'open'
 */
int open(const char *pathname, int flags, ...)
{
#ifdef HAVE_ERRNO_H
  int errno_real = errno;
#endif
  int mode = 0;
  int fd = -1;

  /* Check whether IO instrumentation is enabled */
  int canInstrument = EXTRAE_INITIALIZED()                 &&
                      mpitrace_on                          &&
                      Extrae_get_trace_io();

  /* Can't be evaluated before because the compiler optimizes the if's clauses,
	 * and THREADID calls a null callback if Extrae is not yet initialized */
	if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

  if (flags & O_CREAT)
  {
    va_list arg;
    va_start (arg, flags);
    mode = va_arg (arg, int);
    va_end (arg);
  }

  /* Initialize the module if the pointer to the real call is not yet set */
  if (real_open == NULL)
  {
    Extrae_iotrace_init();
  }

#if defined(DEBUG)
  if (canInstrument)
  {
    fprintf (stderr, PACKAGE_NAME": open is at %p\n", real_open);
    fprintf (stderr, PACKAGE_NAME": open params %s %d\n", pathname, flags);
  }
#endif

  if (real_open != NULL && canInstrument)
  {
    /* Instrumentation is enabled, emit events and invoke the real call */
    Backend_Enter_Instrumentation (4+Caller_Count[CALLER_IO]);
#ifdef HAVE_ERRNO_H                                                             
	  errno = errno_real;                                                         
#endif                                                                          
		fd = real_open (pathname, flags, mode);
#ifdef HAVE_ERRNO_H
  	errno_real = errno;
#endif
    Probe_IO_open_Entry (fd, pathname);
    TRACE_IO_CALLER(LAST_READ_TIME, 3);

    Probe_IO_open_Exit ();
    Backend_Leave_Instrumentation ();
#ifdef HAVE_ERRNO_H
  	errno = errno_real;
#endif
  }
  else if (real_open != NULL && !canInstrument)
  {
    /* Instrumentation is not enabled, bypass to the real call */
    fd = real_open (pathname, flags, mode);
  }
  else
  {
    /*
     * An error is thrown if the application uses this symbol but during the initialization
     * we couldn't find the real implementation. This kind of situation could happen in the
     * very strange case where, by the time this symbol is first called, the libc (where the
     * real implementation is) has not been loaded yet. One suggestion if we see this error
     * is to try to prepend the libc.so to the LD_PRELOAD to force to load it first.
     */
    fprintf (stderr, PACKAGE_NAME": open is not hooked! exiting!!\n");
    abort();
  }
  return fd;
}

/**
 * open64
 *
 * Wrapper for the system call 'open64'
 */
int open64(const char *pathname, int flags, ...)
{
#ifdef HAVE_ERRNO_H                                                             
	int errno_real = errno;                                                       
#endif
  int mode = 0;
  int fd = -1;

  /* Check whether IO instrumentation is enabled */
  int canInstrument = EXTRAE_INITIALIZED()                 &&
                      mpitrace_on                          &&
                      Extrae_get_trace_io();

  /* Can't be evaluated before because the compiler optimizes the if's clauses,
	 * and THREADID calls a null callback if Extrae is not yet initialized */
  if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

  if (flags & O_CREAT)
  {
    va_list arg;
    va_start (arg, flags);
    mode = va_arg (arg, int);
    va_end (arg);
  }

  /* Initialize the module if the pointer to the real call is not yet set */
  if (real_open64 == NULL)
  {
    Extrae_iotrace_init();
  }

#if defined(DEBUG)
  if (canInstrument)
  {
    fprintf (stderr, PACKAGE_NAME": open64 is at %p\n", real_open64);
    fprintf (stderr, PACKAGE_NAME": open64 params %s %d\n", pathname, flags);
  }
#endif

  if (real_open64 != NULL && canInstrument)
  {
    /* Instrumentation is enabled, emit events and invoke the real call */
    Backend_Enter_Instrumentation (4+Caller_Count[CALLER_IO]);
#ifdef HAVE_ERRNO_H
  	errno = errno_real;
#endif
    fd = real_open64 (pathname, flags, mode);
#ifdef HAVE_ERRNO_H
  	errno_real = errno;
#endif
    Probe_IO_open_Entry (fd, pathname);
    TRACE_IO_CALLER(LAST_READ_TIME, 3);

    Probe_IO_open_Exit ();
    Backend_Leave_Instrumentation ();
#ifdef HAVE_ERRNO_H
  	errno = errno_real;
#endif
  }
  else if (real_open64 != NULL && !canInstrument)
  {
    /* Instrumentation is not enabled, bypass to the real call */
    fd = real_open64 (pathname, flags, mode);
  }
  else
  {
    /*
     * An error is thrown if the application uses this symbol but during the initialization
     * we couldn't find the real implementation. This kind of situation could happen in the
     * very strange case where, by the time this symbol is first called, the libc (where the
     * real implementation is) has not been loaded yet. One suggestion if we see this error
     * is to try to prepend the libc.so to the LD_PRELOAD to force to load it first.
     */
    fprintf (stderr, PACKAGE_NAME": open64 is not hooked! exiting!!\n");
    abort();
  }
  return fd;
}

/*
 * fopen
 *
 * Wrapper for the system call 'fopen'
 */
FILE * fopen(const char *path, const char *mode)
{
#ifdef HAVE_ERRNO_H
 	int errno_real = errno;
#endif
  FILE *f = NULL;

  /* Check whether IO instrumentation is enabled */
  int canInstrument = EXTRAE_INITIALIZED()                 &&
                      mpitrace_on                          &&
                      Extrae_get_trace_io();

  /* Can't be evaluated before because the compiler optimizes the if's clauses,
	 * and THREADID calls a null callback if Extrae is not yet initialized */
	if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

  /* Initialize the module if the pointer to the real call is not yet set */
  if (real_fopen == NULL)
  {
    Extrae_iotrace_init();
  }

#if defined(DEBUG)
  if (canInstrument)
  {
    fprintf (stderr, PACKAGE_NAME": fopen is at %p\n", real_fopen);
    fprintf (stderr, PACKAGE_NAME": fopen params %s %s\n", path, mode);
  }
#endif

  if (real_fopen != NULL && canInstrument)
  {
    /* Instrumentation is enabled, emit events and invoke the real call */
		int fd = -1;
    Backend_Enter_Instrumentation (4+Caller_Count[CALLER_IO]);
#ifdef HAVE_ERRNO_H
  	errno = errno_real;
#endif
    f = real_fopen (path, mode);
#ifdef HAVE_ERRNO_H
   	errno_real = errno;
#endif
		if (f != NULL)
		{
			fd = fileno(f);
		}

		Probe_IO_fopen_Entry (fd, path);
		TRACE_IO_CALLER(LAST_READ_TIME, 3);

		Probe_IO_fopen_Exit ();
    Backend_Leave_Instrumentation ();
#ifdef HAVE_ERRNO_H
  	errno = errno_real;
#endif
  }
  else if (real_fopen != NULL && !canInstrument)
  {
    /* Instrumentation is not enabled, bypass to the real call */
    f = real_fopen (path, mode);
  }
  else
  {
    /*
     * An error is thrown if the application uses this symbol but during the initialization
     * we couldn't find the real implementation. This kind of situation could happen in the
     * very strange case where, by the time this symbol is first called, the libc (where the
     * real implementation is) has not been loaded yet. One suggestion if we see this error
     * is to try to prepend the libc.so to the LD_PRELOAD to force to load it first.
     */
    fprintf (stderr, PACKAGE_NAME": fopen is not hooked! exiting!!\n");
    abort();
  }
  return f;
}

/*
 * fopen64
 *
 * Wrapper for the system call 'fopen64'
 */
FILE * fopen64(const char *path, const char *mode)
{
#ifdef HAVE_ERRNO_H
	int errno_real = errno;
#endif
  FILE *f = NULL;

  /* Check whether IO instrumentation is enabled */
  int canInstrument = EXTRAE_INITIALIZED()                 &&
                      mpitrace_on                          &&
                      Extrae_get_trace_io();

  /* Can't be evaluated before because the compiler optimizes the if's clauses,
	 * and THREADID calls a null callback if Extrae is not yet initialized */
	if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

  /* Initialize the module if the pointer to the real call is not yet set */
  if (real_fopen64 == NULL)
  {
    Extrae_iotrace_init();
  }

#if defined(DEBUG)
  if (canInstrument)
  {
    fprintf (stderr, PACKAGE_NAME": fopen64 is at %p\n", real_fopen64);
    fprintf (stderr, PACKAGE_NAME": fopen64 params %s %s\n", path, mode);
  }
#endif

  if (real_fopen64 != NULL && canInstrument)
  {
    /* Instrumentation is enabled, emit events and invoke the real call */
		int fd = -1;
    Backend_Enter_Instrumentation (4+Caller_Count[CALLER_IO]);
#ifdef HAVE_ERRNO_H
	  errno = errno_real;
#endif
    f = real_fopen64 (path, mode);
#ifdef HAVE_ERRNO_H
    errno_real = errno;
#endif
		if (f != NULL)
		{
			fd = fileno(f);
		}
		Probe_IO_fopen_Entry (fd, path);
		TRACE_IO_CALLER(LAST_READ_TIME, 3);

    Probe_IO_fopen_Exit ();
    Backend_Leave_Instrumentation ();
#ifdef HAVE_ERRNO_H
	  errno = errno_real;
#endif
  }
  else if (real_fopen64 != NULL && !canInstrument)
  {
    /* Instrumentation is not enabled, bypass to the real call */
    f = real_fopen64 (path, mode);
  }
  else
  {
    /*
     * An error is thrown if the application uses this symbol but during the initialization
     * we couldn't find the real implementation. This kind of situation could happen in the
     * very strange case where, by the time this symbol is first called, the libc (where the
     * real implementation is) has not been loaded yet. One suggestion if we see this error
     * is to try to prepend the libc.so to the LD_PRELOAD to force to load it first.
     */
    fprintf (stderr, PACKAGE_NAME": fopen64 is not hooked! exiting!!\n");
    abort();
  }
  return f;
}


/**
 * read
 *
 * Wrapper for the system call 'read'
 */
ssize_t read (int fd, void *buf, size_t count)
{
#ifdef HAVE_ERRNO_H
	int errno_real = errno;
#endif
  /* Check whether IO instrumentation is enabled */
  int canInstrument = EXTRAE_INITIALIZED()                 &&
                      mpitrace_on                          &&
                      Extrae_get_trace_io();
  ssize_t res;

  /* Can't be evaluated before because the compiler optimizes the if's clauses,
	 * and THREADID calls a null callback if Extrae is not yet initialized */
  if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

  /* Initialize the module if the pointer to the real call is not yet set */
  if (real_read == NULL)
  {
    Extrae_iotrace_init();
  }

#if defined(DEBUG)
  if (canInstrument)
  {
    fprintf (stderr, PACKAGE_NAME": read is at %p\n", real_read);
    fprintf (stderr, PACKAGE_NAME": read params %d %p %lu\n", fd, buf, count);
  }
#endif

  if (real_read != NULL && canInstrument)
  {
    /* Instrumentation is enabled, emit events and invoke the real call */
    Backend_Enter_Instrumentation (4+Caller_Count[CALLER_IO]);
    Probe_IO_read_Entry (fd, count);
    TRACE_IO_CALLER(LAST_READ_TIME, 3);
#ifdef HAVE_ERRNO_H
		errno = errno_real;
#endif
    res = real_read (fd, buf, count);
#ifdef HAVE_ERRNO_H
		errno_real = errno;
#endif
    Probe_IO_read_Exit ();
    Backend_Leave_Instrumentation ();
#ifdef HAVE_ERRNO_H
		errno = errno_real;
#endif
  }
  else if (real_read != NULL && !canInstrument)
  {
    /* Instrumentation is not enabled, bypass to the real call */
    res = real_read (fd, buf, count);
  }
  else
  {
    /*
     * An error is thrown if the application uses this symbol but during the initialization
     * we couldn't find the real implementation. This kind of situation could happen in the
     * very strange case where, by the time this symbol is first called, the libc (where the
     * real implementation is) has not been loaded yet. One suggestion if we see this error
     * is to try to prepend the libc.so to the LD_PRELOAD to force to load it first.
     */
    fprintf (stderr, PACKAGE_NAME": read is not hooked! exiting!!\n");
    abort();
  }
  return res;
}

/**
 * write
 *
 * Wrapper for the system call 'write'
 */
ssize_t write (int fd, const void *buf, size_t count)
{
#ifdef HAVE_ERRNO_H
	int errno_real = errno;
#endif
  /* Check whether IO instrumentation is enabled */
  int canInstrument = EXTRAE_INITIALIZED()                 &&
                      mpitrace_on &&
                      Extrae_get_trace_io();
  ssize_t res;

  /* Can't be evaluated before because the compiler optimizes the if's clauses,
	 * and THREADID calls a null callback if Extrae is not yet initialized */
  if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

  /* Initialize the module if the pointer to the real call is not yet set */
  if (real_write == NULL)
  {
    Extrae_iotrace_init();
  }

#if defined(DEBUG)
  if (canInstrument)
  {
    fprintf (stderr, PACKAGE_NAME": write is at %p\n", real_write);
    fprintf (stderr, PACKAGE_NAME": write params %d %p %lu\n", fd, buf, count);
  }
#endif

  if (real_write != NULL && canInstrument)
  {
    /* Instrumentation is enabled, emit events and invoke the real call */
    Backend_Enter_Instrumentation (4+Caller_Count[CALLER_IO]);
    Probe_IO_write_Entry (fd, count);
    TRACE_IO_CALLER(LAST_READ_TIME, 3);
#ifdef HAVE_ERRNO_H
		errno = errno_real;
#endif
    res = real_write (fd, buf, count);
#ifdef HAVE_ERRNO_H
		errno_real = errno;
#endif
    Probe_IO_write_Exit ();
    Backend_Leave_Instrumentation ();
#ifdef HAVE_ERRNO_H
		errno = errno_real;
#endif
  }
  else if (real_write != NULL && !canInstrument)
  {
    /* Instrumentation is not enabled, bypass to the real call */
    res = real_write (fd, buf, count);
  }
  else
  {
    /*
     * An error is thrown if the application uses this symbol but during the initialization
     * we couldn't find the real implementation. This kind of situation could happen in the
     * very strange case where, by the time this symbol is first called, the libc (where the
     * real implementation is) has not been loaded yet. One suggestion if we see this error
     * is to try to prepend the libc.so to the LD_PRELOAD to force to load it first.
     */
    fprintf (stderr, PACKAGE_NAME": write is not hooked! exiting!!\n");
    abort();
  }
  return res;
}

/**
 * fread
 *
 * Wrapper for the system call 'fread'
 */
size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream)
{
#ifdef HAVE_ERRNO_H                                                             
	int errno_real = errno;                                                       
#endif                                                                          
  /* Check whether IO instrumentation is enabled */
  int canInstrument = EXTRAE_INITIALIZED()                 &&
                      mpitrace_on                          &&
                      Extrae_get_trace_io();
  size_t res;

  /* Can't be evaluated before because the compiler optimizes the if's clauses,
	 * and THREADID calls a null callback if Extrae is not yet initialized */
  if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

  /* Initialize the module if the pointer to the real call is not yet set */
  if (real_fread == NULL)
  {
    Extrae_iotrace_init();
  }

#if defined(DEBUG)
  if (canInstrument)
  {
    fprintf (stderr, PACKAGE_NAME": fread is at %p\n", real_fread);
    fprintf (stderr, PACKAGE_NAME": fread params %p %ld %ld %p\n", ptr, size, nmemb, stream);
  }
#endif

  if (real_fread != NULL && canInstrument)
  {
    /* Instrumentation is enabled, emit events and invoke the real call */
    Backend_Enter_Instrumentation (4+Caller_Count[CALLER_IO]);
    Probe_IO_fread_Entry (fileno(stream), size * nmemb);
    TRACE_IO_CALLER(LAST_READ_TIME, 3);
#ifdef HAVE_ERRNO_H
		errno = errno_real;
#endif
    res = real_fread (ptr, size, nmemb, stream);
#ifdef HAVE_ERRNO_H
		errno_real = errno;
#endif
    Probe_IO_fread_Exit ();
    Backend_Leave_Instrumentation ();
#ifdef HAVE_ERRNO_H
		errno = errno_real;
#endif
  }
  else if (real_fread != NULL && !canInstrument)
  {
    /* Instrumentation is not enabled, bypass to the real call */
    res = real_fread (ptr, size, nmemb, stream);
  }
  else
  {
    /*
     * An error is thrown if the application uses this symbol but during the initialization
     * we couldn't find the real implementation. This kind of situation could happen in the
     * very strange case where, by the time this symbol is first called, the libc (where the
     * real implementation is) has not been loaded yet. One suggestion if we see this error
     * is to try to prepend the libc.so to the LD_PRELOAD to force to load it first.
     */
    fprintf (stderr, PACKAGE_NAME": fread is not hooked! exiting!!\n");
    abort();
  }
  return res;
}

/**
 * fwrite
 *
 * Wrapper for the system call 'fwrite'
 */
size_t fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream)
{
#ifdef HAVE_ERRNO_H                                                             
  int errno_real = errno;                                                       
#endif                                                                          
  /* Check whether IO instrumentation is enabled */
  int canInstrument = EXTRAE_INITIALIZED()                 &&
                      mpitrace_on                          &&
                      Extrae_get_trace_io();
  size_t res;

  /* Can't be evaluated before because the compiler optimizes the if's clauses,
	 * and THREADID calls a null callback if Extrae is not yet initialized */
	if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

  if (real_fwrite == NULL)
  {
    Extrae_iotrace_init();
  }

#if defined(DEBUG)
  if (canInstrument)
  {
    fprintf (stderr, PACKAGE_NAME": fwrite is at %p\n", real_fwrite);
    fprintf (stderr, PACKAGE_NAME": fwrite params %p %ld %ld %p\n", ptr, size, nmemb, stream);
  }
#endif

  if (real_fwrite != NULL && canInstrument)
  {
    /* Instrumentation is enabled, emit events and invoke the real call */
    Backend_Enter_Instrumentation (4+Caller_Count[CALLER_IO]);
    Probe_IO_fwrite_Entry (fileno(stream), size * nmemb);
    TRACE_IO_CALLER(LAST_READ_TIME, 3);
#ifdef HAVE_ERRNO_H
		errno = errno_real;
#endif
    res = real_fwrite (ptr, size, nmemb, stream);
#ifdef HAVE_ERRNO_H
		errno_real = errno;
#endif
    Probe_IO_fwrite_Exit ();
    Backend_Leave_Instrumentation ();
#ifdef HAVE_ERRNO_H
		errno = errno_real;
#endif
  }
  else if (real_fwrite != NULL && !canInstrument)
  {
    /* Instrumentation is not enabled, bypass to the real call */
    res = real_fwrite (ptr, size, nmemb, stream);
  }
  else
  {
    /*
     * An error is thrown if the application uses this symbol but during the initialization
     * we couldn't find the real implementation. This kind of situation could happen in the
     * very strange case where, by the time this symbol is first called, the libc (where the
     * real implementation is) has not been loaded yet. One suggestion if we see this error
     * is to try to prepend the libc.so to the LD_PRELOAD to force to load it first.
     */
    fprintf (stderr, PACKAGE_NAME": fwrite is not hooked! exiting!!\n");
    abort();
  }
  return res;
}

/**
 * pread
 *
 * Wrapper for the system call 'pread'
 */
ssize_t pread(int fd, void *buf, size_t count, off_t offset)
{
#ifdef HAVE_ERRNO_H                                                             
  int errno_real = errno;                                                       
#endif                                                                          
  /* Check whether IO instrumentation is enabled */
  int canInstrument = EXTRAE_INITIALIZED()                 &&
                      mpitrace_on                          &&
                      Extrae_get_trace_io();
  ssize_t res;

  /* Can't be evaluated before because the compiler optimizes the if's clauses,
	 * and THREADID calls a null callback if Extrae is not yet initialized */
	if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

  /* Initialize the module if the pointer to the real call is not yet set */
  if (real_pread == NULL)
  {
    Extrae_iotrace_init();
  }

#if defined(DEBUG)
  if (canInstrument)
  {
    fprintf (stderr, PACKAGE_NAME": pread is at %p\n", real_pread);
    fprintf (stderr, PACKAGE_NAME": pread params %d %p %ld %ld\n", fd, buf, count, offset);
  }
#endif

  if (real_pread != NULL && canInstrument)
  {
    /* Instrumentation is enabled, emit events and invoke the real call */
    Backend_Enter_Instrumentation (4+Caller_Count[CALLER_IO]);
    Probe_IO_pread_Entry (fd, count);
    TRACE_IO_CALLER(LAST_READ_TIME, 3);
#ifdef HAVE_ERRNO_H
		errno = errno_real;
#endif
    res = real_pread (fd, buf, count, offset);
#ifdef HAVE_ERRNO_H
		errno_real = errno;
#endif
    Probe_IO_pread_Exit ();
    Backend_Leave_Instrumentation ();
#ifdef HAVE_ERRNO_H
		errno = errno_real;
#endif
  }
  else if (real_pread != NULL && !canInstrument)
  {
    /* Instrumentation is not enabled, bypass to the real call */
    res = real_pread (fd, buf, count, offset);
  }
  else
  {
    /*
     * An error is thrown if the application uses this symbol but during the initialization
     * we couldn't find the real implementation. This kind of situation could happen in the
     * very strange case where, by the time this symbol is first called, the libc (where the
     * real implementation is) has not been loaded yet. One suggestion if we see this error
     * is to try to prepend the libc.so to the LD_PRELOAD to force to load it first.
     */
    fprintf (stderr, PACKAGE_NAME": pread is not hooked! exiting!!\n");
    abort();
  }
  return res;
}

/**
 * pwrite
 *
 * Wrapper for the system call 'pwrite'
 */
ssize_t pwrite(int fd, const void *buf, size_t count, off_t offset)
{
#ifdef HAVE_ERRNO_H                                                             
	int errno_real = errno;                                                       
#endif                                                                          
  /* Check whether IO instrumentation is enabled */
  int canInstrument = EXTRAE_INITIALIZED()                 &&
                      mpitrace_on                          &&
                      Extrae_get_trace_io();
  ssize_t res;

  /* Can't be evaluated before because the compiler optimizes the if's clauses,
	 * and THREADID calls a null callback if Extrae is not yet initialized */
	if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

  /* Initialize the module if the pointer to the real call is not yet set */
  if (real_pwrite == NULL)
  {
    Extrae_iotrace_init();
  }

#if defined(DEBUG)
  if (canInstrument)
  {
    fprintf (stderr, PACKAGE_NAME": pwrite is at %p\n", real_pwrite);
    fprintf (stderr, PACKAGE_NAME": pwrite params %d %p %ld %ld\n", fd, buf, count, offset);
  }
#endif

  if (real_pwrite != NULL && canInstrument)
  {
    /* Instrumentation is enabled, emit events and invoke the real call */
    Backend_Enter_Instrumentation (4+Caller_Count[CALLER_IO]);
    Probe_IO_pwrite_Entry (fd, count);
    TRACE_IO_CALLER(LAST_READ_TIME, 3);
#ifdef HAVE_ERRNO_H
		errno = errno_real;
#endif
    res = real_pwrite (fd, buf, count, offset);
#ifdef HAVE_ERRNO_H
		errno_real = errno;
#endif
    Probe_IO_pwrite_Exit ();
    Backend_Leave_Instrumentation ();
#ifdef HAVE_ERRNO_H
		errno = errno_real;
#endif
  }
  else if (real_pwrite != NULL && !canInstrument)
  {
    /* Instrumentation is not enabled, bypass to the real call */
    res = real_pwrite (fd, buf, count, offset);
  }
  else
  {
    /*
     * An error is thrown if the application uses this symbol but during the initialization
     * we couldn't find the real implementation. This kind of situation could happen in the
     * very strange case where, by the time this symbol is first called, the libc (where the
     * real implementation is) has not been loaded yet. One suggestion if we see this error
     * is to try to prepend the libc.so to the LD_PRELOAD to force to load it first.
     */
    fprintf (stderr, PACKAGE_NAME": pwrite is not hooked! exiting!!\n");
    abort();
  }
  return res;
}

/**
 * readv
 *
 * Wrapper for the system call 'readv'
 */
ssize_t readv (int fd, const struct iovec *iov, int iovcnt)
{
#ifdef HAVE_ERRNO_H                                                             
	int errno_real = errno;                                                       
#endif                                                                          
  /* Check whether IO instrumentation is enabled */
  int canInstrument = EXTRAE_INITIALIZED()                 &&
                      mpitrace_on                          &&
                      Extrae_get_trace_io();
  ssize_t res;

  /* Can't be evaluated before because the compiler optimizes the if's clauses,
	 * and THREADID calls a null callback if Extrae is not yet initialized */
	if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

  /* Initialize the module if the pointer to the real call is not yet set */
  if (real_readv == NULL)
  {
    Extrae_iotrace_init();
  }

#if defined(DEBUG)
  if (canInstrument)
  {
    fprintf (stderr, PACKAGE_NAME": readv is at %p\n", real_readv);
    fprintf (stderr, PACKAGE_NAME": readv params %d %p %d\n", fd, iov, iovcnt);
  }
#endif

  if (real_readv != NULL && canInstrument)
  {
    /* Instrumentation is enabled, emit events and invoke the real call */
    int i;
    ssize_t size = 0;

    Backend_Enter_Instrumentation (4+Caller_Count[CALLER_IO]);

    for (i=0; i<iovcnt; i++)
    {
      size += iov[i].iov_len;
    }

    Probe_IO_readv_Entry (fd, size);
    TRACE_IO_CALLER(LAST_READ_TIME, 3);
#ifdef HAVE_ERRNO_H
		errno = errno_real;
#endif
    res = real_readv (fd, iov, iovcnt);
#ifdef HAVE_ERRNO_H
		errno_real = errno;
#endif
    Probe_IO_readv_Exit ();
    Backend_Leave_Instrumentation ();
#ifdef HAVE_ERRNO_H
		errno = errno_real;
#endif
  }
  else if (real_readv != NULL && !canInstrument)
  {
    /* Instrumentation is not enabled, bypass to the real call */
    res = real_readv (fd, iov, iovcnt);
  }
  else
  {
    /*
     * An error is thrown if the application uses this symbol but during the initialization
     * we couldn't find the real implementation. This kind of situation could happen in the
     * very strange case where, by the time this symbol is first called, the libc (where the
     * real implementation is) has not been loaded yet. One suggestion if we see this error
     * is to try to prepend the libc.so to the LD_PRELOAD to force to load it first.
     */
    fprintf (stderr, PACKAGE_NAME": readv is not hooked! exiting!!\n");
    abort();
  }
  return res;
}

/**
 * writev
 *
 * Wrapper for the system call 'writev'
 */
ssize_t writev(int fd, const struct iovec *iov, int iovcnt)
{
#ifdef HAVE_ERRNO_H                                                             
	int errno_real = errno;                                                       
#endif                                                                          
  /* Check whether IO instrumentation is enabled */
  int canInstrument = EXTRAE_INITIALIZED()                 &&
                      mpitrace_on                          &&
                      Extrae_get_trace_io();
  ssize_t res;

  /* Can't be evaluated before because the compiler optimizes the if's clauses,
	 * and THREADID calls a null callback if Extrae is not yet initialized */
	if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

  /* Initialize the module if the pointer to the real call is not yet set */
  if (real_writev == NULL)
  {
    Extrae_iotrace_init();
  }

#if defined(DEBUG)
  if (canInstrument)
  {
    fprintf (stderr, PACKAGE_NAME": writev is at %p\n", real_writev);
    fprintf (stderr, PACKAGE_NAME": writev params %d %p %d\n", fd, iov, iovcnt);
  }
#endif

  if (real_writev != NULL && canInstrument)
  {
    /* Instrumentation is enabled, emit events and invoke the real call */
    int i;
    ssize_t size = 0;

    Backend_Enter_Instrumentation (4+Caller_Count[CALLER_IO]);

    for (i=0; i<iovcnt; i++)
    {
      size += iov[i].iov_len;
    }

    Probe_IO_writev_Entry (fd, size);
    TRACE_IO_CALLER(LAST_READ_TIME, 3);
#ifdef HAVE_ERRNO_H
		errno = errno_real;
#endif
    res = real_writev (fd, iov, iovcnt);
#ifdef HAVE_ERRNO_H
		errno_real = errno;
#endif
    Probe_IO_writev_Exit ();
    Backend_Leave_Instrumentation ();
#ifdef HAVE_ERRNO_H
		errno = errno_real;
#endif
  }
  else if (real_writev != NULL && !canInstrument)
  {
    /* Instrumentation is not enabled, bypass to the real call */
    res = real_writev (fd, iov, iovcnt);
  }
  else
  {
    /*
     * An error is thrown if the application uses this symbol but during the initialization
     * we couldn't find the real implementation. This kind of situation could happen in the
     * very strange case where, by the time this symbol is first called, the libc (where the
     * real implementation is) has not been loaded yet. One suggestion if we see this error
     * is to try to prepend the libc.so to the LD_PRELOAD to force to load it first.
     */
    fprintf (stderr, PACKAGE_NAME": writev is not hooked! exiting!!\n");
    abort();
  }
  return res;
}

/**
 * preadv
 *
 * Wrapper for the system call 'preadv'
 */
ssize_t preadv(int fd, const struct iovec *iov, int iovcnt, off_t offset)
{
#ifdef HAVE_ERRNO_H                                                             
	int errno_real = errno;                                                       
#endif                                                                          
  /* Check whether IO instrumentation is enabled */
  int canInstrument = EXTRAE_INITIALIZED()                 &&
                      mpitrace_on                          &&
                      Extrae_get_trace_io();
  ssize_t res;

  /* Can't be evaluated before because the compiler optimizes the if's clauses,
	 * and THREADID calls a null callback if Extrae is not yet initialized */
  if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

  /* Initialize the module if the pointer to the real call is not yet set */
  if (real_preadv == NULL)
  {
    Extrae_iotrace_init();
  }

#if defined(DEBUG)
  if (canInstrument)
  {
    fprintf (stderr, PACKAGE_NAME": preadv is at %p\n", real_preadv);
    fprintf (stderr, PACKAGE_NAME": preadv params %d %p %d %ld\n", fd, iov, iovcnt, offset);
  }
#endif

  if (real_preadv != NULL && canInstrument)
  {
    /* Instrumentation is enabled, emit events and invoke the real call */
    int i;
    ssize_t size = 0;

    Backend_Enter_Instrumentation (4+Caller_Count[CALLER_IO]);

    for (i=0; i<iovcnt; i++)
    {
      size += iov[i].iov_len;
    }

    Probe_IO_preadv_Entry (fd, size);
    TRACE_IO_CALLER(LAST_READ_TIME, 3);
#ifdef HAVE_ERRNO_H
		errno = errno_real;
#endif
    res = real_preadv (fd, iov, iovcnt, offset);
#ifdef HAVE_ERRNO_H
		errno_real = errno;
#endif
    Probe_IO_preadv_Exit ();
    Backend_Leave_Instrumentation ();
#ifdef HAVE_ERRNO_H
		errno = errno_real;
#endif
  }
  else if (real_preadv != NULL && !canInstrument)
  {
    /* Instrumentation is not enabled, bypass to the real call */
    res = real_preadv (fd, iov, iovcnt, offset);
  }
  else
  {
    /*
     * An error is thrown if the application uses this symbol but during the initialization
     * we couldn't find the real implementation. This kind of situation could happen in the
     * very strange case where, by the time this symbol is first called, the libc (where the
     * real implementation is) has not been loaded yet. One suggestion if we see this error
     * is to try to prepend the libc.so to the LD_PRELOAD to force to load it first.
     */
    fprintf (stderr, PACKAGE_NAME": preadv is not hooked! exiting!!\n");
    abort();
  }
  return res;
}

/**
 * preadv64
 *
 * Wrapper for the system call 'preadv64'
 */
ssize_t preadv64(int fd, const struct iovec *iov, int iovcnt, __off64_t offset)
{
#ifdef HAVE_ERRNO_H                                                             
	int errno_real = errno;                                                       
#endif                                                                          
  /* Check whether IO instrumentation is enabled */
  int canInstrument = EXTRAE_INITIALIZED()                 &&
                      mpitrace_on                          &&
                      Extrae_get_trace_io();
  ssize_t res;

  /* Can't be evaluated before because the compiler optimizes the if's clauses,
	 * and THREADID calls a null callback if Extrae is not yet initialized */
	if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

  /* Initialize the module if the pointer to the real call is not yet set */
  if (real_preadv64 == NULL)
  {
    Extrae_iotrace_init();
  }

#if defined(DEBUG)
  if (canInstrument)
  {
    fprintf (stderr, PACKAGE_NAME": preadv64 is at %p\n", real_preadv64);
    fprintf (stderr, PACKAGE_NAME": preadv64 params %d %p %d %ld\n", fd, iov, iovcnt, offset);
  }
#endif

  if (real_preadv64 != NULL && canInstrument)
  {
    /* Instrumentation is enabled, emit events and invoke the real call */
    int i;
    ssize_t size = 0;

    Backend_Enter_Instrumentation (4+Caller_Count[CALLER_IO]);

    for (i=0; i<iovcnt; i++)
    {
      size += iov[i].iov_len;
    }

    Probe_IO_preadv_Entry (fd, size);
    TRACE_IO_CALLER(LAST_READ_TIME, 3);
#ifdef HAVE_ERRNO_H
		errno = errno_real;
#endif
    res = real_preadv64 (fd, iov, iovcnt, offset);
#ifdef HAVE_ERRNO_H
		errno_real = errno;
#endif
    Probe_IO_preadv_Exit ();
    Backend_Leave_Instrumentation ();
#ifdef HAVE_ERRNO_H
		errno = errno_real;
#endif
  }
  else if (real_preadv64 != NULL && !canInstrument)
  {
    /* Instrumentation is not enabled, bypass to the real call */
    res = real_preadv64 (fd, iov, iovcnt, offset);
  }
  else
  {
    /*
     * An error is thrown if the application uses this symbol but during the initialization
     * we couldn't find the real implementation. This kind of situation could happen in the
     * very strange case where, by the time this symbol is first called, the libc (where the
     * real implementation is) has not been loaded yet. One suggestion if we see this error
     * is to try to prepend the libc.so to the LD_PRELOAD to force to load it first.
     */
    fprintf (stderr, PACKAGE_NAME": preadv64 is not hooked! exiting!!\n");
    abort();
  }
  return res;
}

/**
 * pwritev
 *
 * Wrapper for the system call 'pwritev'
 */
ssize_t pwritev(int fd, const struct iovec *iov, int iovcnt, off_t offset)
{
#ifdef HAVE_ERRNO_H                                                             
	int errno_real = errno;                                                       
#endif                                                                          
  /* Check whether IO instrumentation is enabled */
  int canInstrument = EXTRAE_INITIALIZED()                 &&
                      mpitrace_on                          &&
                      Extrae_get_trace_io();
  ssize_t res;

  /* Can't be evaluated before because the compiler optimizes the if's clauses,
	 * and THREADID calls a null callback if Extrae is not yet initialized */
	if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

  /* Initialize the module if the pointer to the real call is not yet set */
  if (real_pwritev == NULL)
  {
    Extrae_iotrace_init();
  }

#if defined(DEBUG)
  if (canInstrument)
  {
    fprintf (stderr, PACKAGE_NAME": pwritev is at %p\n", real_pwritev);
    fprintf (stderr, PACKAGE_NAME": pwritev params %d %p %d %ld\n", fd, iov, iovcnt, offset);
  }
#endif

  if (real_pwritev != NULL && canInstrument)
  {
    /* Instrumentation is enabled, emit events and invoke the real call */
    int i;
    ssize_t size = 0;

    Backend_Enter_Instrumentation (4+Caller_Count[CALLER_IO]);

    for (i=0; i<iovcnt; i++)
    {
      size += iov[i].iov_len;
    }

    Probe_IO_pwritev_Entry (fd, size);
    TRACE_IO_CALLER(LAST_READ_TIME, 3);
#ifdef HAVE_ERRNO_H
		errno = errno_real;
#endif
    res = real_pwritev (fd, iov, iovcnt, offset);
#ifdef HAVE_ERRNO_H
		errno_real = errno;
#endif
    Probe_IO_pwritev_Exit ();
    Backend_Leave_Instrumentation ();
#ifdef HAVE_ERRNO_H
		errno = errno_real;
#endif
  }
  else if (real_pwritev != NULL && !canInstrument)
  {
    /* Instrumentation is not enabled, bypass to the real call */
    res = real_pwritev (fd, iov, iovcnt, offset);
  }
  else
  {
    /*
     * An error is thrown if the application uses this symbol but during the initialization
     * we couldn't find the real implementation. This kind of situation could happen in the
     * very strange case where, by the time this symbol is first called, the libc (where the
     * real implementation is) has not been loaded yet. One suggestion if we see this error
     * is to try to prepend the libc.so to the LD_PRELOAD to force to load it first.
     */
    fprintf (stderr, PACKAGE_NAME": pwritev is not hooked! exiting!!\n");
    abort();
  }
  return res;
}

/**
 * pwritev64
 *
 * Wrapper for the system call 'pwritev64'
 */
ssize_t pwritev64(int fd, const struct iovec *iov, int iovcnt, __off64_t offset)
{
#ifdef HAVE_ERRNO_H                                                             
	int errno_real = errno;                                                       
#endif                                                                          
  /* Check whether IO instrumentation is enabled */
  int canInstrument = EXTRAE_INITIALIZED()                 &&
                      mpitrace_on                          &&
                      Extrae_get_trace_io();
  ssize_t res;

  /* Can't be evaluated before because the compiler optimizes the if's clauses,
	 * and THREADID calls a null callback if Extrae is not yet initialized */
	if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

  /* Initialize the module if the pointer to the real call is not yet set */
  if (real_pwritev64 == NULL)
  {
    Extrae_iotrace_init();
  }

#if defined(DEBUG)
  if (canInstrument)
  {
    fprintf (stderr, PACKAGE_NAME": pwritev64 is at %p\n", real_pwritev64);
    fprintf (stderr, PACKAGE_NAME": pwritev64 params %d %p %d %ld\n", fd, iov, iovcnt, offset);
  }
#endif

  if (real_pwritev64 != NULL && canInstrument)
  {
    /* Instrumentation is enabled, emit events and invoke the real call */
    int i;
    ssize_t size = 0;

    Backend_Enter_Instrumentation (4+Caller_Count[CALLER_IO]);

    for (i=0; i<iovcnt; i++)
    {
      size += iov[i].iov_len;
    }

    Probe_IO_pwritev_Entry (fd, size);
    TRACE_IO_CALLER(LAST_READ_TIME, 3);
#ifdef HAVE_ERRNO_H
		errno = errno_real;
#endif
    res = real_pwritev64 (fd, iov, iovcnt, offset);
#ifdef HAVE_ERRNO_H
		errno_real = errno;
#endif
    Probe_IO_pwritev_Exit ();
    Backend_Leave_Instrumentation ();
#ifdef HAVE_ERRNO_H
		errno = errno_real;
#endif
  }
  else if (real_pwritev64 != NULL && !canInstrument)
  {
    /* Instrumentation is not enabled, bypass to the real call */
    res = real_pwritev64 (fd, iov, iovcnt, offset);
  }
  else
  {
    /*
     * An error is thrown if the application uses this symbol but during the initialization
     * we couldn't find the real implementation. This kind of situation could happen in the
     * very strange case where, by the time this symbol is first called, the libc (where the
     * real implementation is) has not been loaded yet. One suggestion if we see this error
     * is to try to prepend the libc.so to the LD_PRELOAD to force to load it first.
     */
    fprintf (stderr, PACKAGE_NAME": pwritev64 is not hooked! exiting!!\n");
    abort();
  }
  return res;
}

# endif /* -DPIC */

#endif /* -DINSTRUMENT_IO */
