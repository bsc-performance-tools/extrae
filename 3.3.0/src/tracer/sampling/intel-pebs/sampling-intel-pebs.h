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

/* Additional credis to Vince Weaver, vincent.weaver _at_ maine.edu 
   For further reference see https://github.com/deater/perf_event_tests */

#ifndef EXTRAE_INTEL_PEBS_H_INCLUDED
#define EXTRAE_INTEL_PEBS_H_INCLUDED

#include <intel-pebs-types.h>

int Extrae_IntelPEBS_enable (int loads);
void Extrae_IntelPEBS_disable (void); 

void Extrae_IntelPEBS_setLoadPeriod (int period);
void Extrae_IntelPEBS_setStorePeriod (int period);
void Extrae_IntelPEBS_setLoadSampling (int enabled);
void Extrae_IntelPEBS_setMinimumLoadLatency (int numcycles);
void Extrae_IntelPEBS_setStoreSampling (int enabled);

void Extrae_IntelPEBS_nextSampling (void);

#endif
