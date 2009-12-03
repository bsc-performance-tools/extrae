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

#ifndef PPU_INCLUDED_H
#define PPU_INCLUDED_H

#include "mpitrace_user_events.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Special Cell tracing from the PPU side - initialization and finalization */
int CELLtrace_init (int spus, speid_t * spe_id);
void CELLtrace_fini (void);

/* Additional symbols for PPU side - synonims for the MPI_* standard calls */
void PPUtrace_init (void);
void PPUtrace_fini (void);
void PPUtrace_user_function (unsigned enter);
void PPUtrace_event (unsigned int type, unsigned int value);
void PPUtrace_Nevent (unsigned int count, unsigned int *types,
	unsigned int *values);
void PPUtrace_shutdown (void);
void PPUtrace_restart (void);
void PPUtrace_counters (void);
void PPUtrace_previous_hwc_set (void);
void PPUtrace_next_hwc_set (void);
void PPUtrace_eventandcounters (int type, int value);
void PPUItrace_Neventandcounters (unsigned int count, unsigned int *types,
	unsigned int *values);
void PPUtrace_set_options (int options);

#ifdef __cplusplus
}
#endif

#endif /* PPU_INCLUDED_H */
