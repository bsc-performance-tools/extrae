/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL$
 | @last_commit: $Date$
 | @version:     $Revision$
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef _CELL_WRAPPER_H_INCLUDED_
#define _CELL_WRAPPER_H_INCLUDED_

#include <config.h>

#if defined(IS_CELL_MACHINE)

#if CELL_SDK == 1
# include <libspe.h>
#elif CELL_SDK == 2
# include <libspe2.h>
#endif

#include "spu/defaults.h"

#if CELL_SDK == 1
int CELLtrace_init (int spus, speid_t * spe_id);
int Extrae_CELL_init (int spus, speid_t * spe_id);
#elif CELL_SDK == 2
int CELLtrace_init (int spus, spe_context_ptr_t * spe_id);
int Extrae_CELL_init (int spus, spe_context_ptr_t * spe_id);
#endif
int CELLtrace_fini (void);
int Extrae_CELL_fini (void);

int prepare_CELLTrace_init (int nthreads);

extern unsigned int cell_tracing_enabled;
extern unsigned int spu_dma_channel;
extern unsigned int spu_buffer_size;
extern unsigned int spu_file_size;

#endif /* IS_CELL_MACHINE */

#endif /* _CELL_WRAPPER_H_INCLUDED_ */

