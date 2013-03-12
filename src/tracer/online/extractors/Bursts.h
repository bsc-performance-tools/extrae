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

#ifndef __BURSTS_H__
#define __BURSTS_H__

#include <MRNet_wrappers.h>

#define BURSTS_CHUNK 100

class Bursts 
{
  public:
    Bursts();
    ~Bursts();

    void Insert(unsigned long long timestamp, unsigned long long duration);

    int GetNumberOfBursts();
    unsigned long long GetBurstTime(int burst_id);
    unsigned long long GetBurstDuration(int burst_id);

#if USE_HARDWARE_COUNTERS
    void Insert(unsigned long long timestamp, unsigned long long duration, int hwc_set, long long *hwcs);

    int GetBurstCountersSet(int burst_id);
    int GetBurstCountersValues(int burst_id, long long *& hwcs);
#endif /* USE_HARDWARE_COUNTERS */

  private:
    unsigned long long *Timestamps;
    unsigned long long *Durations;
#if USE_HARDWARE_COUNTERS
    long long          *HWCValues;
    int                *HWCSets;
#endif /* USE_HARDWARE_COUNTERS */

    int NumberOfBursts;
    int MaxBursts;
};


#endif /* __BURSTS__ */
