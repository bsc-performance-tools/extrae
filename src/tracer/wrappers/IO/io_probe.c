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

#if HAVE_UNISTD_H
# include <unistd.h>
#endif
#if HAVE_SYS_STAT_H
# include <sys/stat.h>
#endif
#if HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif
#if HAVE_PTHREAD_H
# include <pthread.h>
#endif

#include "threadid.h"
#include "wrapper.h"
#include "trace_macros.h"
#include "io_probe.h"

/***********************************************************************************************\
 * This file contains the probes to record the begin/end events for instrumented I/O routines. *
 * As usual, the probes are separate for begin and end events to be able to be injected before *
 * and after the call site or at the entry and exit points of the instrumented routine with    *
 * Dyninst. Despite the support, the I/O calls are not intercepted with Dyninst at the moment. *
\***********************************************************************************************/

/* Global variable to control whether the tracing for I/O calls is enabled */
static int trace_io_enabled = FALSE;

/* Mutex to synchronize multiple threads recording in the *.SYM file the name of the opened files */
pthread_mutex_t record_open_file_in_sym;

/* We keep a global counter of the opened files to assign an unique id to each open() call. 
 * This id is then associated with the name of the opened file in the *.SYM. 
 * In this way we don't have to keep track of the close() calls and deal with the reused fd's.
 */
static int open_counter = 0;

/** 
 * Extrae_set_trace_io
 * 
 * \param enable Set the tracing for I/O calls enabled or disabled.
 */
void Extrae_set_trace_io (int enable)
{
  trace_io_enabled = enable; 
}

/** 
 * Extrae_get_trace_io_status
 *
 * \return true if the tracing for I/O calls is enabled; false otherwise.
 */
int Extrae_get_trace_io (void)
{ 
  return trace_io_enabled; 
}

/** 
 * Extrae_get_descriptor_type
 * 
 * \param fd A file descriptor
 * \return whether fd is an open file descriptor referring to a terminal, a regular file, a socket or a FIFO pipe.
 */
static unsigned Extrae_get_descriptor_type (int fd)
{
  if (isatty(fd))
  {
    return DESCRIPTOR_TYPE_ATTY;
  }
  else
  {
    struct stat buf;

    fstat (fd, &buf);
    if (S_ISREG(buf.st_mode))
    {
      return DESCRIPTOR_TYPE_REGULARFILE;
    }
    else if (S_ISSOCK(buf.st_mode))
    {
      return DESCRIPTOR_TYPE_SOCKET;
    }
    else if (S_ISFIFO(buf.st_mode))
    {
      return DESCRIPTOR_TYPE_FIFO_PIPE;
    }
    else
    {
      return DESCRIPTOR_TYPE_UNKNOWN; 
    }
  }
}

/**
 * Probe_IO_open_Entry
 *
 * Probe injected at the beginning of the I/O call 'open' 
 * \param fd A file descriptor
 * \param pathname Name of the opened file 
 */
void Probe_IO_open_Entry (int fd, const char *pathname)
{
  if (mpitrace_on && trace_io_enabled)
  {
    unsigned type = Extrae_get_descriptor_type (fd);

    TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, OPEN_EV, EVT_BEGIN, fd);
    TRACE_MISCEVENT(LAST_READ_TIME, OPEN_EV, EVT_BEGIN+2, type);

    pthread_mutex_lock(&record_open_file_in_sym);
    /* 
     * Register the current open id and file name in the *.SYM inside a mutex because
     * there might be multiple threads trying to do this and the *.SYM is per task.
     */
    open_counter ++;
    Extrae_AddTypeValuesEntryToLocalSYM ('F', open_counter, (char *)pathname, (char )0, 0, NULL, NULL);
    TRACE_MISCEVENT(LAST_READ_TIME, OPEN_EV, EVT_BEGIN+3, open_counter);

    pthread_mutex_unlock(&record_open_file_in_sym);
  }
}

/**
 * Probe_IO_open_Exit
 *
 * Probe injected at the end of the I/O call 'open' 
 */
void Probe_IO_open_Exit ()
{
  if (mpitrace_on && trace_io_enabled)
  {
    TRACE_MISCEVENTANDCOUNTERS(TIME, OPEN_EV, EVT_END, EMPTY);
  }
}

/**
 * Probe_IO_fopen_Entry
 *
 * Probe injected at the beginning of the I/O call 'fopen' 
 * \param fd A file descriptor
 * \param pathname Name of the opened file 
 */
void Probe_IO_fopen_Entry (int fd, const char *pathname)
{
  if (mpitrace_on && trace_io_enabled)
  {
    unsigned type = Extrae_get_descriptor_type (fd);

    TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, FOPEN_EV, EVT_BEGIN, fd);
    TRACE_MISCEVENT(LAST_READ_TIME, FOPEN_EV, EVT_BEGIN+2, type);

    pthread_mutex_lock(&record_open_file_in_sym);
    /* 
     * Register the current open id and file name in the *.SYM inside a mutex because
     * there might be multiple threads trying to do this and the *.SYM is per task.
     */
    open_counter ++;
    Extrae_AddTypeValuesEntryToLocalSYM ('F', open_counter, (char *)pathname, (char )0, 0, NULL, NULL);
    TRACE_MISCEVENT(LAST_READ_TIME, FOPEN_EV, EVT_BEGIN+3, open_counter);

    pthread_mutex_unlock(&record_open_file_in_sym);
  }
}

/**
 * Probe_IO_fopen_Exit
 *
 * Probe injected at the end of the I/O call 'fopen' 
 */
void Probe_IO_fopen_Exit ()
{
  if (mpitrace_on && trace_io_enabled)
  {
    TRACE_MISCEVENTANDCOUNTERS(TIME, FOPEN_EV, EVT_END, EMPTY);
  }
}

/**
 * Probe_IO_read_Entry
 *
 * Probe injected at the beginning of the I/O call 'read' 
 * \param fd A file descriptor
 * \param size The number of bytes read
 */
void Probe_IO_read_Entry (int fd, ssize_t size)
{
  if (mpitrace_on && trace_io_enabled)
  {
    unsigned type = Extrae_get_descriptor_type (fd);
    TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, READ_EV, EVT_BEGIN, fd);
    TRACE_MISCEVENT(LAST_READ_TIME, READ_EV, EVT_BEGIN+1, size);
    TRACE_MISCEVENT(LAST_READ_TIME, READ_EV, EVT_BEGIN+2, type);
  }
}

/**
 * Probe_IO_read_Exit
 *
 * Probe injected at the end of the I/O call 'read' 
 */
void Probe_IO_read_Exit (void)
{
  if (mpitrace_on && trace_io_enabled)
  {
    TRACE_MISCEVENTANDCOUNTERS(TIME, READ_EV, EVT_END, EMPTY);
  }
}

/**
 * Probe_IO_write_Entry
 *
 * Probe injected at the beginning of the I/O call 'write' 
 * \param fd A file descriptor
 * \param size The number of bytes written
 */
void Probe_IO_write_Entry (int fd, ssize_t size)
{
  if (mpitrace_on && trace_io_enabled)
  {
    unsigned type = Extrae_get_descriptor_type (fd);
    TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, WRITE_EV, EVT_BEGIN, fd);
    TRACE_MISCEVENT(LAST_READ_TIME, WRITE_EV, EVT_BEGIN+1, size);
    TRACE_MISCEVENT(LAST_READ_TIME, WRITE_EV, EVT_BEGIN+2, type);
  }
}

/**
 * Probe_IO_write_Exit
 *
 * Probe injected at the end of the I/O call 'write' 
 */
void Probe_IO_write_Exit (void)
{
  if (mpitrace_on && trace_io_enabled)
  {
    TRACE_MISCEVENTANDCOUNTERS(TIME, WRITE_EV, EVT_END, EMPTY);
  }
}

/**
 * Probe_IO_fread_Entry 
 * 
 * Probe injected at the beginning of the I/O call 'fread'
 * \param fd A file descriptor
 * \param size The number of bytes read
 */
void Probe_IO_fread_Entry (int fd, size_t size)
{
  if (mpitrace_on && trace_io_enabled)
  {
    unsigned type = Extrae_get_descriptor_type (fd);
    TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, FREAD_EV, EVT_BEGIN, fd);
    TRACE_MISCEVENT(LAST_READ_TIME, FREAD_EV, EVT_BEGIN+1, size);
    TRACE_MISCEVENT(LAST_READ_TIME, FREAD_EV, EVT_BEGIN+2, type);
  }
}

/**
 * Probe_IO_fread_Exit
 *
 * Probe injected at the end of the I/O call 'fread' 
 */
void Probe_IO_fread_Exit (void)
{
  if (mpitrace_on && trace_io_enabled)
  {
    TRACE_MISCEVENTANDCOUNTERS(TIME, FREAD_EV, EVT_END, EMPTY);
  }
}

/**
 * Probe_IO_fwrite_Entry
 *
 * Probe injected at the beginning of the I/O call 'fwrite' 
 * \param fd A file descriptor
 * \param size The number of bytes written
 */
void Probe_IO_fwrite_Entry (int fd, size_t size)
{
  if (mpitrace_on && trace_io_enabled)
  {
    unsigned type = Extrae_get_descriptor_type (fd);
    TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, FWRITE_EV, EVT_BEGIN, fd);
    TRACE_MISCEVENT(LAST_READ_TIME, FWRITE_EV, EVT_BEGIN+1, size);
    TRACE_MISCEVENT(LAST_READ_TIME, FWRITE_EV, EVT_BEGIN+2, type);
  }
}

/**
 * Probe_IO_fwrite_Exit
 *
 * Probe injected at the end of the I/O call 'fwrite' 
 */
void Probe_IO_fwrite_Exit (void)
{
  if (mpitrace_on && trace_io_enabled)
  {
    TRACE_MISCEVENTANDCOUNTERS(TIME, FWRITE_EV, EVT_END, EMPTY);
  }
}

/**
 * Probe_IO_pread_Entry 
 * 
 * Probe injected at the beginning of the I/O call 'pread'
 * \param fd A file descriptor
 * \param size The number of bytes read
 */
void Probe_IO_pread_Entry (int fd, ssize_t size)
{
  if (mpitrace_on && trace_io_enabled)
  {
    unsigned type = Extrae_get_descriptor_type (fd);
    TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, PREAD_EV, EVT_BEGIN, fd);
    TRACE_MISCEVENT(LAST_READ_TIME, PREAD_EV, EVT_BEGIN+1, size);
    TRACE_MISCEVENT(LAST_READ_TIME, PREAD_EV, EVT_BEGIN+2, type);
  }
}

/**
 * Probe_IO_pread_Exit
 *
 * Probe injected at the end of the I/O call 'pread' 
 */
void Probe_IO_pread_Exit (void)
{
  if (mpitrace_on && trace_io_enabled)
  {
    TRACE_MISCEVENTANDCOUNTERS(TIME, PREAD_EV, EVT_END, EMPTY);
  }
}

/**
 * Probe_IO_pwrite_Entry
 *
 * Probe injected at the beginning of the I/O call 'pwrite' 
 * \param fd A file descriptor
 * \param size The number of bytes written
 */
void Probe_IO_pwrite_Entry (int fd, ssize_t size)
{
  if (mpitrace_on && trace_io_enabled)
  {
    unsigned type = Extrae_get_descriptor_type (fd);
    TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, PWRITE_EV, EVT_BEGIN, fd);
    TRACE_MISCEVENT(LAST_READ_TIME, PWRITE_EV, EVT_BEGIN+1, size);
    TRACE_MISCEVENT(LAST_READ_TIME, PWRITE_EV, EVT_BEGIN+2, type);
  }
}

/**
 * Probe_IO_pwrite_Exit
 *
 * Probe injected at the end of the I/O call 'pwrite' 
 */
void Probe_IO_pwrite_Exit (void)
{
  if (mpitrace_on && trace_io_enabled)
  {
    TRACE_MISCEVENTANDCOUNTERS(TIME, PWRITE_EV, EVT_END, EMPTY);
  }
}

/**
 * Probe_IO_readv_Entry 
 * 
 * Probe injected at the beginning of the I/O call 'readv'
 * \param fd A file descriptor
 * \param size The number of bytes read
 */
void Probe_IO_readv_Entry (int fd, ssize_t size)
{
  if (mpitrace_on && trace_io_enabled)
  {
    unsigned type = Extrae_get_descriptor_type (fd);
    TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, READV_EV, EVT_BEGIN, fd);
    TRACE_MISCEVENT(LAST_READ_TIME, READV_EV, EVT_BEGIN+1, size);
    TRACE_MISCEVENT(LAST_READ_TIME, READV_EV, EVT_BEGIN+2, type);
  }
}

/**
 * Probe_IO_readv_Exit
 *
 * Probe injected at the end of the I/O call 'readv' 
 */
void Probe_IO_readv_Exit (void)
{
  if (mpitrace_on && trace_io_enabled)
  {
    TRACE_MISCEVENTANDCOUNTERS(TIME, READV_EV, EVT_END, EMPTY);
  }
}

/**
 * Probe_IO_writev_Entry
 *
 * Probe injected at the beginning of the I/O call 'writev' 
 * \param fd A file descriptor
 * \param size The number of bytes written
 */
void Probe_IO_writev_Entry (int fd, ssize_t size)
{
  if (mpitrace_on && trace_io_enabled)
  {
    unsigned type = Extrae_get_descriptor_type (fd);
    TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, WRITEV_EV, EVT_BEGIN, fd);
    TRACE_MISCEVENT(LAST_READ_TIME, WRITEV_EV, EVT_BEGIN+1, size);
    TRACE_MISCEVENT(LAST_READ_TIME, WRITEV_EV, EVT_BEGIN+2, type);
  }
}

/**
 * Probe_IO_writev_Exit
 *
 * Probe injected at the end of the I/O call 'writev' 
 */
void Probe_IO_writev_Exit (void)
{
  if (mpitrace_on && trace_io_enabled)
  {
    TRACE_MISCEVENTANDCOUNTERS(TIME, WRITEV_EV, EVT_END, EMPTY);
  }
}

/**
 * Probe_IO_preadv_Entry 
 * 
 * Probe injected at the beginning of the I/O call 'preadv'
 * \param fd A file descriptor
 * \param size The number of bytes read
 */
void Probe_IO_preadv_Entry (int fd, ssize_t size)
{
  if (mpitrace_on && trace_io_enabled)
  {
    unsigned type = Extrae_get_descriptor_type (fd);
    TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, PREAD_EV, EVT_BEGIN, fd);
    TRACE_MISCEVENT(LAST_READ_TIME, PREAD_EV, EVT_BEGIN+1, size);
    TRACE_MISCEVENT(LAST_READ_TIME, PREAD_EV, EVT_BEGIN+2, type);
  }
}

/**
 * Probe_IO_preadv_Exit
 *
 * Probe injected at the end of the I/O call 'preadv' 
 */
void Probe_IO_preadv_Exit (void)
{
  if (mpitrace_on && trace_io_enabled)
  {
    TRACE_MISCEVENTANDCOUNTERS(TIME, PREAD_EV, EVT_END, EMPTY);
  }
}

/**
 * Probe_IO_pwritev_Entry
 *
 * Probe injected at the beginning of the I/O call 'pwritev' 
 * \param fd A file descriptor
 * \param size The number of bytes written
 */
void Probe_IO_pwritev_Entry (int fd, ssize_t size)
{
  if (mpitrace_on && trace_io_enabled)
  {
    unsigned type = Extrae_get_descriptor_type (fd);
    TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, PWRITEV_EV, EVT_BEGIN, fd);
    TRACE_MISCEVENT(LAST_READ_TIME, PWRITEV_EV, EVT_BEGIN+1, size);
    TRACE_MISCEVENT(LAST_READ_TIME, PWRITEV_EV, EVT_BEGIN+2, type);
  }
}

/**
 * Probe_IO_pwritev_Exit
 *
 * Probe injected at the end of the I/O call 'pwritev' 
 */
void Probe_IO_pwritev_Exit (void)
{
  if (mpitrace_on && trace_io_enabled)
  {
    TRACE_MISCEVENTANDCOUNTERS(TIME, PWRITEV_EV, EVT_END, EMPTY);
  }
}

