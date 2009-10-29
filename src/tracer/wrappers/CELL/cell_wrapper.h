/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *                                                             ___           *
 *   +---------+     http:// www.cepba.upc.edu/tools_i.htm    /  __          *
 *   |    o//o |     http:// www.bsc.es                      /  /  _____     *
 *   |   o//o  |                                            /  /  /     \    *
 *   |  o//o   |     E-mail: cepbatools@cepba.upc.edu      (  (  ( B S C )   *
 *   | o//o    |     Phone:          +34-93-401 71 78       \  \  \_____/    *
 *   +---------+     Fax:            +34-93-401 25 77        \  \__          *
 *    C E P B A                                               \___           *
 *                                                                           *
 * This software is subject to the terms of the CEPBA/BSC license agreement. *
 *      You must accept the terms of this license to use this software.      *
 *                                 ---------                                 *
 *                European Center for Parallelism of Barcelona               *
 *                      Barcelona Supercomputing Center                      *
\*****************************************************************************/

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL$
 | 
 | @last_commit: $Date$
 | @version:     $Revision$
 | 
 | History:
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
#elif CELL_SDK == 2
int CELLtrace_init (int spus, spe_context_ptr_t * spe_id);
#endif
int CELLtrace_fini (void);

int prepare_CELLTrace_init (int nthreads);

extern unsigned int cell_tracing_enabled;
extern unsigned int spu_dma_channel;
extern unsigned int spu_buffer_size;
extern unsigned int spu_file_size;

#endif /* IS_CELL_MACHINE */

#endif /* _CELL_WRAPPER_H_INCLUDED_ */

