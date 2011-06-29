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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL$
 | @last_commit: $Date$
 | @version:     $Revision$
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef PPU_INCLUDED_H
#define PPU_INCLUDED_H

#include "mpitrace_user_events.h"
#include "extrae_version.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Special Cell tracing from the PPU side - initialization and finalization */
int Extrae_CELL_init (int spus, speid_t * spe_id);
int CELLtrace_init (int spus, speid_t * spe_id);

void Extrae_CELL_fini (void);
void CELLtrace_fini (void);

/* Additional symbols for PPU side - synonims for the MPI_* standard calls */
void Extrae_init (void);
void PPUtrace_init (void);

void Extrae_fini (void);
void PPUtrace_fini (void);

void Extrae_user_function (unsigned enter);
void PPUtrace_user_function (unsigned enter);

void Extrae_event (unsigned int type, unsigned int value);
void PPUtrace_event (unsigned int type, unsigned int value);

void Extrae_nevent (unsigned int count, unsigned int *types,
	unsigned int *values);
void PPUtrace_nevent (unsigned int count, unsigned int *types,
	unsigned int *values);

void Extrae_shutdown (void);
void PPUtrace_shutdown (void);

void Extrae_restart (void);
void PPUtrace_restart (void);

void Extrae_counters (void);
void PPUtrace_counters (void);

void Extrae_previous_hwc_set (void);
void PPUtrace_previous_hwc_set (void);

void Extrae_next_hwc_set (void);
void PPUtrace_next_hwc_set (void);

void Extrae_eventandcounters (int type, int value);
void PPUtrace_eventandcounters (int type, int value);

void Extrae_neventandcounters (unsigned int count, unsigned int *types,
	unsigned int *values);
void PPUtrace_neventandcounters (unsigned int count, unsigned int *types,
	unsigned int *values);

void Extrae_set_options (int options);
void PPUtrace_set_options (int options);

#ifdef __cplusplus
}
#endif

#endif /* PPU_INCLUDED_H */
