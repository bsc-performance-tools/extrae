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

#ifndef __IO_PROBE_H__
#define __IO_PROBE_H__

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif

void Extrae_set_trace_io (int enable);
int  Extrae_get_trace_io (void);

void Probe_IO_open_Entry (int fd, const char *pathname);
void Probe_IO_open_Exit  (void);

void Probe_IO_fopen_Entry (int fd, const char *pathname);
void Probe_IO_fopen_Exit  (void);

void Probe_IO_read_Entry (int fd, ssize_t size);
void Probe_IO_read_Exit  (void);

void Probe_IO_write_Entry (int fd, ssize_t size);
void Probe_IO_write_Exit  (void);

void Probe_IO_fread_Entry (int fd, size_t size);
void Probe_IO_fread_Exit  (void);

void Probe_IO_fwrite_Entry (int fd, size_t size);
void Probe_IO_fwrite_Exit  (void);

void Probe_IO_pread_Entry (int fd, ssize_t size);
void Probe_IO_pread_Exit  (void);

void Probe_IO_pwrite_Entry (int fd, ssize_t size);
void Probe_IO_pwrite_Exit  (void);

void Probe_IO_readv_Entry (int fd, ssize_t size);
void Probe_IO_readv_Exit  (void);

void Probe_IO_writev_Entry (int fd, ssize_t size);
void Probe_IO_writev_Exit  (void);

void Probe_IO_preadv_Entry (int fd, ssize_t size);
void Probe_IO_preadv_Exit  (void);

void Probe_IO_pwritev_Entry (int fd, ssize_t size);
void Probe_IO_pwritev_Exit  (void);

#endif /* __IO_PROBE_H__ */
