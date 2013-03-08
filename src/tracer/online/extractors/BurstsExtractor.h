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
 | @file: $HeadURL: https://svn.bsc.es/repos/ptools/extrae/trunk/src/tracer/xml-parse.c $
 | @last_commit: $Date: 2013-01-25 15:56:47 +0100 (Fri, 25 Jan 2013) $
 | @version:     $Revision: 1464 $
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef __BURSTS_EXTRACTOR_H__
#define __BURSTS_EXTRACTOR_H__

#include "BufferExtractor.h"
#include "Bursts.h"

#if USE_HARDWARE_COUNTERS

# include "num_hwc.h"

# define BURST_HWC_CLEAR(accum) \
   for (int k=0; k<MAX_HWC; k++) { accum[k]  = 0; }
# define BURST_HWC_DIFF(accum, evt_end, evt_ini) \
   for (int k=0; k<MAX_HWC; k++) { accum[k] = (Get_EvHWCVal(evt_end))[k] - (Get_EvHWCVal(evt_ini))[k]; }

#endif /* USE_HARDWARE_COUNTERS */


class BurstsExtractor : public BufferExtractor
{
  public:
    BurstsExtractor(unsigned long long min_duration);

    void ProcessEvent(event_t *evt);
    bool isBurstBegin(event_t *evt);
    bool isBurstEnd  (event_t *evt);

    Bursts * GetBursts();

  private:
    Bursts            *ExtractedBursts;
    event_t           *LastBegin;
    unsigned long long DurationFilter;
#if USE_HARDWARE_COUNTERS
    long long          OngoingBurstHWCs[MAX_HWC];
#endif /* USE_HARDWARE_COUNTERS */
};

#endif /* __BURSTS_EXTRACTOR_H__ */
