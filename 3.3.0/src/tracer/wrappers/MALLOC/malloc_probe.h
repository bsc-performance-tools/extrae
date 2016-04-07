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

#ifndef PROBE_MALLOC_H_INCLUDED
#define PROBE_MALLOC_H_INCLUDED

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif

void Extrae_set_trace_malloc (int b);
int Extrae_get_trace_malloc (void);
void Extrae_set_trace_malloc_allocate (int b);
int Extrae_get_trace_malloc_allocate (void);
void Extrae_set_trace_malloc_allocate_threshold (unsigned long long t);
unsigned long long Extrae_get_trace_malloc_allocate_threshold (void);
void Extrae_set_trace_malloc_free (int b);
int Extrae_get_trace_malloc_free (void);

void Probe_Malloc_Entry (size_t s);
void Probe_Malloc_Exit (void *p);

void Probe_Free_Entry (void *p);
void Probe_Free_Exit (void);

void Probe_Calloc_Entry (size_t s1, size_t s2);
void Probe_Calloc_Exit (void *p);

void Probe_Realloc_Entry (void *p, size_t s);
void Probe_Realloc_Exit (void *p);

#endif /* PROBE_MALLOC_H_INCLUDED */
