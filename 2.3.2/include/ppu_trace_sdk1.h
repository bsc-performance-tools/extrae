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

#include "extrae_version.h"

#ifdef __cplusplus
extern "C" {
#endif

void Extrae_init(void);
void MPItrace_init(void);
void Extrae_fini(void);
void MPItrace_fini(void);

int  Extrae_CELL_init (int spus, speid_t * spe_id);
int  CELLtrace_init (int spus, speid_t * spe_id);

void Extrae_CELL_fini (void);
void CELLtrace_fini (void);

void Extrae_event (unsigned int type, unsigned int value);
void MPItrace_event (unsigned int type, unsigned int value);
void PPUtrace_event (unsigned int type, unsigned int value);

void Extrae_nevent (unsigned int count, unsigned int *types, unsigned int *values);
void MPItrace_nevent (unsigned int count, unsigned int *types, unsigned int *values);
void PPUtrace_nevent (unsigned int count, unsigned int *types, unsigned int *values);

#ifdef __cplusplus
}
#endif

#endif /* PPU_INCLUDED_H */
