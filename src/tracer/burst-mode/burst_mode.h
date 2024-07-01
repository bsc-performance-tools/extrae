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

#ifndef BURST_MOD_DEFINED
#define BURST_MOD_DEFINED

#include "stats_module.h"

typedef void (*func_ptr_t) (void);

void xtr_burst_init ( void );

void xtr_burst_finalize (void);

void xtr_burst_begin ( void );

int xtr_burst_end ( void );

void xtr_burst_parallel_OL_entry (void * function_address) __attribute__((weak));

void xtr_burst_parallel_OL_exit ( void ) __attribute__((weak));

void xtr_burst_realloc (int old_num_threads, int new_num_threads);


#endif /* End of BURST_MOD_DEFINED */
