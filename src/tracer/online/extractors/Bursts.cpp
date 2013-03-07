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

#include "common.h"

static char UNUSED rcsid[] = "$Id: threadid.c 1311 2012-10-25 11:05:07Z harald $";

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#include "Bursts.h"
#include "num_hwc.h"
#include "tags.h"

Bursts::Bursts()
{
  NumberOfBursts = 0;
  MaxBursts      = 0;

  Timestamps = NULL;
  Durations  = NULL;
  HWCValues  = NULL;
  HWCSets    = NULL;
}

Bursts::~Bursts()
{
  if (NumberOfBursts > 0)
  {
    free(Timestamps);
    free(Durations);
    free(HWCValues);
    free(HWCSets);
  }
}

/**
 * Store a new burst.
 * \param timestamp The timestamp of the burst.
 * \param duration  The duration of the burst.
 * \param hwc_set   The set of counters read in this burst.
 * \param hwcs      The value of the counters.
 */
void Bursts::Insert(unsigned long long timestamp, unsigned long long duration, int hwc_set, long long *hwcs)
{
  if (NumberOfBursts == MaxBursts)
  {
    MaxBursts += BURSTS_CHUNK;
    Timestamps = (unsigned long long *)realloc(Timestamps, MaxBursts * sizeof(unsigned long long));
    Durations  = (unsigned long long *)realloc(Durations, MaxBursts * sizeof(unsigned long long));
    HWCValues  = (long long *)realloc(HWCValues, MaxBursts * MAX_HWC * sizeof(long long));
    HWCSets    = (int *)realloc(HWCSets, MaxBursts * sizeof(int));
  }
  Timestamps[ NumberOfBursts ] = timestamp;
  Durations [ NumberOfBursts ] = duration;
  HWCSets   [ NumberOfBursts ] = hwc_set;
  for (int i=0; i<MAX_HWC; i++)
  {
    int idx = (NumberOfBursts * MAX_HWC) + i;
    HWCValues[ idx ] = hwcs[i];
  }
  NumberOfBursts ++;
}

/**
 * Returns the number of bursts in the container.
 * \return The total count of bursts.
 */
int Bursts::GetNumberOfBursts()
{
  return NumberOfBursts;
}

/**
 * Returns the time of the specified burst.
 * \return The timestamp of the burst.
 */
unsigned long long Bursts::GetBurstTime(int burst_id)
{
  if (burst_id >= NumberOfBursts)
    return 0;
  else
    return Timestamps[burst_id];
}

/**
 * Returns the duration of the specified burst.
 * \return The duration of the burst.
 */
unsigned long long Bursts::GetBurstDuration(int burst_id)
{
  if (burst_id >= NumberOfBursts)
    return 0;
  else
    return Durations[burst_id];
}

/**
 * Returns the set of counters read for the specified burst.
 * \return The counters set identifier.
 */
int Bursts::GetBurstCountersSet(int burst_id)
{
  if (burst_id >= NumberOfBursts)
    return 0;
  else
    return HWCSets[burst_id];
}

/**
 * Returns the value of the counters read for the specified burst.
 * \return The counters values by reference, and the number of counters by value.
 */
int Bursts::GetBurstCountersValues(int burst_id, long long *& hwcs)
{
  hwcs = (long long *)malloc(sizeof(long long) * MAX_HWC);
  for (int i=0; i<MAX_HWC; i++)
  {
    int idx = (burst_id * MAX_HWC) + i;
    hwcs[i] = HWCValues[idx];
  }
  return MAX_HWC;
}

