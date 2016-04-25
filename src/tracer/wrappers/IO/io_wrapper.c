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
static ssize_t (*real_read)(int fd, void *buf, size_t count)                              = NULL;
static ssize_t (*real_write)(int fd, const void *buf, size_t count)                       = NULL;

static size_t  (*real_fread)(void *ptr, size_t size, size_t nmemb, FILE *stream)          = NULL;
static size_t  (*real_fwrite)(const void *ptr, size_t size, size_t nmemb, FILE *stream)   = NULL;

static ssize_t (*real_pread)(int fd, void *buf, size_t count, off_t offset)               = NULL;
static ssize_t (*real_pwrite)(int fd, const void *buf, size_t count, off_t offset)        = NULL;

static ssize_t (*real_readv)(int fd, const struct iovec *iov, int iovcnt)                 = NULL;
static ssize_t (*real_writev)(int fd, const struct iovec *iov, int iovcnt)                = NULL;
static ssize_t (*real_preadv)(int fd, const struct iovec *iov, int iovcnt, off_t offset)  = NULL;
static ssize_t (*real_pwritev)(int fd, const struct iovec *iov, int iovcnt, off_t offset) = NULL;

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
  real_read    = (ssize_t(*)(int, void*, size_t)) dlsym (RTLD_NEXT, "read");
  real_write   = (ssize_t(*)(int, const void*, size_t)) dlsym (RTLD_NEXT, "write");

  real_fread   = (size_t(*)(void *, size_t, size_t, FILE *)) dlsym (RTLD_NEXT, "fread");
  real_fwrite  = (size_t(*)(const void *, size_t, size_t, FILE *)) dlsym (RTLD_NEXT, "fwrite");

  real_pread   = (ssize_t(*)(int fd, void *buf, size_t count, off_t offset)) dlsym (RTLD_NEXT, "pread");
  real_pwrite  = (ssize_t(*)(int fd, const void *buf, size_t count, off_t offset)) dlsym (RTLD_NEXT, "pwrite");

  real_readv   = (ssize_t(*)(int, const struct iovec *, int)) dlsym (RTLD_NEXT, "readv");
  real_writev  = (ssize_t(*)(int, const struct iovec *, int)) dlsym (RTLD_NEXT, "writev");
  real_preadv  = (ssize_t(*)(int, const struct iovec *, int, off_t)) dlsym (RTLD_NEXT, "preadv");
  real_pwritev = (ssize_t(*)(int, const struct iovec *, int, off_t)) dlsym (RTLD_NEXT, "pwritev");

# else

  fprintf (stderr, PACKAGE_NAME": Warning! I/O instrumentation requires linking with shared library!\n");

# endif
}

# if defined(PIC) /* Only available for .so libraries */

/**
 * read
 *
 * Wrapper for the system call 'read'
 */
ssize_t read (int fd, void *buf, size_t count)
{
  /* Check whether IO instrumentation is enabled */
  int canInstrument = !Backend_inInstrumentation(THREADID) && 
                      mpitrace_on                          &&
                      Extrae_get_trace_io();
  ssize_t res;

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
    Backend_Enter_Instrumentation (2);
    Probe_IO_read_Entry (fd, count);
    TRACE_IO_CALLER(LAST_READ_TIME, 3);
    res = real_read (fd, buf, count);
    Probe_IO_read_Exit ();
    Backend_Leave_Instrumentation ();
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
  /* Check whether IO instrumentation is enabled */
  int canInstrument = !Backend_inInstrumentation(THREADID) && 
                      mpitrace_on &&
                      Extrae_get_trace_io();
  ssize_t res;

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
    Backend_Enter_Instrumentation (2);
    Probe_IO_write_Entry (fd, count);
    TRACE_IO_CALLER(LAST_READ_TIME, 3);
    res = real_write (fd, buf, count);
    Probe_IO_write_Exit ();
    Backend_Leave_Instrumentation ();
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
  /* Check whether IO instrumentation is enabled */
  int canInstrument = !Backend_inInstrumentation(THREADID) &&
                      mpitrace_on                          &&
                      Extrae_get_trace_io();
  size_t res;

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
    Backend_Enter_Instrumentation (2);
    Probe_IO_fread_Entry (fileno(stream), size * nmemb);
    TRACE_IO_CALLER(LAST_READ_TIME, 3);
    res = real_fread (ptr, size, nmemb, stream);
    Probe_IO_fread_Exit ();
    Backend_Leave_Instrumentation ();
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
  /* Check whether IO instrumentation is enabled */
  int canInstrument = !Backend_inInstrumentation(THREADID) &&
                      mpitrace_on                          &&
                      Extrae_get_trace_io();
  size_t res;

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
    Backend_Enter_Instrumentation (2);
    Probe_IO_fwrite_Entry (fileno(stream), size * nmemb);
    TRACE_IO_CALLER(LAST_READ_TIME, 3);
    res = real_fwrite (ptr, size, nmemb, stream);
    Probe_IO_fwrite_Exit ();
    Backend_Leave_Instrumentation ();
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
  /* Check whether IO instrumentation is enabled */
  int canInstrument = !Backend_inInstrumentation(THREADID) &&
                      mpitrace_on                          &&
                      Extrae_get_trace_io();
  ssize_t res;

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
    Backend_Enter_Instrumentation (2);
    Probe_IO_pread_Entry (fd, count);
    TRACE_IO_CALLER(LAST_READ_TIME, 3);
    res = real_pread (fd, buf, count, offset);
    Probe_IO_pread_Exit ();
    Backend_Leave_Instrumentation ();
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
  /* Check whether IO instrumentation is enabled */
  int canInstrument = !Backend_inInstrumentation(THREADID) &&
                      mpitrace_on                          &&
                      Extrae_get_trace_io();
  ssize_t res;

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
    Backend_Enter_Instrumentation (2);
    Probe_IO_pwrite_Entry (fd, count);
    TRACE_IO_CALLER(LAST_READ_TIME, 3);
    res = real_pwrite (fd, buf, count, offset);
    Probe_IO_pwrite_Exit ();
    Backend_Leave_Instrumentation ();
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
  /* Check whether IO instrumentation is enabled */
  int canInstrument = !Backend_inInstrumentation(THREADID) &&
                      mpitrace_on                          &&
                      Extrae_get_trace_io();
  ssize_t res;

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

    Backend_Enter_Instrumentation (2);

    for (i=0; i<iovcnt; i++)
    {
      size += iov[i].iov_len;
    }

    Probe_IO_readv_Entry (fd, size);
    TRACE_IO_CALLER(LAST_READ_TIME, 3);
    res = real_readv (fd, iov, iovcnt);
    Probe_IO_readv_Exit ();
    Backend_Leave_Instrumentation ();
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
  /* Check whether IO instrumentation is enabled */
  int canInstrument = !Backend_inInstrumentation(THREADID) &&
                      mpitrace_on                          &&
                      Extrae_get_trace_io();
  ssize_t res;

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

    Backend_Enter_Instrumentation (2);

    for (i=0; i<iovcnt; i++)
    {
      size += iov[i].iov_len;
    }

    Probe_IO_writev_Entry (fd, size);
    TRACE_IO_CALLER(LAST_READ_TIME, 3);
    res = real_writev (fd, iov, iovcnt);
    Probe_IO_writev_Exit ();
    Backend_Leave_Instrumentation ();
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
  /* Check whether IO instrumentation is enabled */
  int canInstrument = !Backend_inInstrumentation(THREADID) &&
                      mpitrace_on                          &&
                      Extrae_get_trace_io();
  ssize_t res;
  
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

    Backend_Enter_Instrumentation (2);

    for (i=0; i<iovcnt; i++)
    {
      size += iov[i].iov_len;
    }

    Probe_IO_preadv_Entry (fd, size);
    TRACE_IO_CALLER(LAST_READ_TIME, 3);
    res = real_preadv (fd, iov, iovcnt, offset);
    Probe_IO_preadv_Exit ();
    Backend_Leave_Instrumentation ();
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
 * pwritev
 * 
 * Wrapper for the system call 'pwritev' 
 */
ssize_t pwritev(int fd, const struct iovec *iov, int iovcnt, off_t offset)
{
  /* Check whether IO instrumentation is enabled */
  int canInstrument = !Backend_inInstrumentation(THREADID) &&
                      mpitrace_on                          &&
                      Extrae_get_trace_io();
  ssize_t res;
  
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

    Backend_Enter_Instrumentation (2);

    for (i=0; i<iovcnt; i++)
    {
      size += iov[i].iov_len;
    }

    Probe_IO_pwritev_Entry (fd, size);
    TRACE_IO_CALLER(LAST_READ_TIME, 3);
    res = real_pwritev (fd, iov, iovcnt, offset);
    Probe_IO_pwritev_Exit ();
    Backend_Leave_Instrumentation ();
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

# endif /* -DPIC */

#endif /* -DINSTRUMENT_IO */
