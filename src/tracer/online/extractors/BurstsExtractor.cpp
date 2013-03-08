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

#include "common.h"

static char UNUSED rcsid[] = "$Id$";

#include "BurstsExtractor.h"
#include "events.h"
#include "timesync.h"
#include "taskid.h"
#include "hwc.h"

/**
 * Constructor. Allocates a new container to store the bursts extracted from the buffer.
 *
 * @param min_duration The minimum duration of a CPU burst to consider.
 */
BurstsExtractor::BurstsExtractor(unsigned long long min_duration)
{
  LastBegin       = NULL;
  DurationFilter  = min_duration;
  ExtractedBursts = new Bursts();
}

/**
 * Callback for every event parsed from the buffer when the super-class method Extract() is called.
 * Here, we look for entries and exits of MPI's to delimit the CPU bursts and store them in 
 * the ExtractedBursts container.
 *
 * @param evt The event being parsed.
 */
void BurstsExtractor::ProcessEvent(event_t *evt)
{
  if (isBurstBegin(evt))
  {
    /* Here begins a burst */
    LastBegin = evt;
#if USE_HARDWARE_COUNTERS
    BURST_HWC_CLEAR(OngoingBurstHWCs);
#endif /* USE_HARDWARE_COUNTERS */
  }
  else if ((isBurstEnd(evt)) && (LastBegin != NULL))
  {
    /* Here ends a burst */
    unsigned long long ts_ini      = Get_EvTime(LastBegin);
    unsigned long long ts_ini_sync = TIMESYNC(0, TASKID, ts_ini);
    unsigned long long ts_end      = Get_EvTime(evt);
    unsigned long long duration    = ts_end - ts_ini;

    /* Only store those bursts that are longer than the given threshold */
    if (duration >= DurationFilter)
    {
      /* The counters are always accumulating, so make the diff 
       * between the end and the start of the burst to have the 
       * counters values for the region.
       */
#if USE_HARDWARE_COUNTERS
      int hwc_set = Get_EvHWCSet(evt);
      BURST_HWC_DIFF(OngoingBurstHWCs, evt, LastBegin);

      ExtractedBursts->Insert(ts_ini_sync, duration, hwc_set, OngoingBurstHWCs);
#else
      ExtractedBursts->Insert(ts_ini_sync, duration);
#endif /* USE_HARDWARE_COUNTERS */
    }
  }
  else
  {
    if (Get_EvEvent(evt) == HWC_CHANGE_EV)
    {
      /* If counters changed in the middle of the burst, discard it */
      LastBegin = NULL;
    }
  }
}

/**
 * Returns true if the given event marks the start of a burst.
 *
 * @param evt A buffer event.
 * @return true if delimits the start of a burst; false otherwise.
 */
bool BurstsExtractor::isBurstBegin(event_t *evt)
{
  int type  = Get_EvEvent(evt);
  int value = Get_EvValue(evt);

  return (((IsMPI(type)) && (value == EVT_END)) || ((IsBurst(type)) && (value == EVT_BEGIN)));
}

/**
 * Returns true if the given event marks the end of a burst.
 *
 * @param evt A buffer event.
 * @return true if delimits the end of a burst; false otherwise.
 */
bool BurstsExtractor::isBurstEnd(event_t *evt)
{
  int type  = Get_EvEvent (evt);
  int value = Get_EvValue (evt);

  return (((IsMPI(type)) && (value == EVT_BEGIN)) || ((IsBurst(type)) && (value == EVT_END)));
}

/**
 * @return the extracted bursts.
 */
Bursts * BurstsExtractor::GetBursts()
{
  return ExtractedBursts;
}
