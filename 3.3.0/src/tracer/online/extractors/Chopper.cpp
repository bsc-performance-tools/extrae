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

#include "common.h"

#if defined(HAVE_STDLIB_H)
# include <stdlib.h>
#endif
#include "Chopper.h"
#include <stdio.h>

Chopper::Chopper()
{
  ChopAt    = NULL;
  ChopTime  = 0;
  ChopType  = KEEP_ONGOING_STATE;
}

event_t * Chopper::FindCloserRunning(int thread_id, unsigned long long time_to_chop, int chop_type, bool use_checkpoint)
{
  ChopAt   = NULL;
  ChopTime = time_to_chop;
  ChopType = chop_type;

  PrevRunningBeginEv = PrevRunningEndEv = LastRunningBeginEv = LastRunningEndEv = NULL;
 
  ParseBuffer(thread_id, use_checkpoint);

  return ChopAt;
}

unsigned long long Chopper::FindCloserRunningTime(int thread_id, unsigned long long time_to_chop, int chop_type, bool use_checkpoint)
{
  FindCloserRunning( thread_id, time_to_chop, chop_type, use_checkpoint );

  if (ChopAt == NULL)
  {
    return time_to_chop;
  }
  else
  {
    return Get_EvTime(ChopAt);
  }
}

int Chopper::ParseEvent(int thread_id, event_t *current_event)
{
  unsigned long long current_time = Get_EvTime(current_event);

  if (isRunningBegin(thread_id, current_event))
  {
    if (LastRunningEndEv != NULL)
    {
      PrevRunningBeginEv = LastRunningBeginEv;
      PrevRunningEndEv   = LastRunningEndEv;
    }
    LastRunningBeginEv = current_event;
    LastRunningEndEv = NULL;
  }
  else if (isRunningEnd(thread_id, current_event) && LastRunningBeginEv != NULL)
  {
    LastRunningEndEv = current_event;

    if (Get_EvTime( LastRunningEndEv ) >= ChopTime)
    {
      if (ChopType == KEEP_ONGOING_STATE)
      {
        ChopAt = LastRunningBeginEv;
      }
      else if (ChopType == DISCARD_ONGOING_STATE)
      {
        ChopAt = PrevRunningEndEv;
      }
      return STOP_PARSING;
    }
  }
  return CONTINUE_PARSING;
}

#if 0
int Chopper::ParseEvent(int thread_id, event_t *current_event)
{
  unsigned long long current_time = Get_EvTime(current_event);

  if (((thread_id == 0) && (isBurstEnd(current_event))) ||
      ((thread_id > 0)  && (isBurstEndInAuxThreads(current_event))))
  {
    ChopAt = current_event; 
  }

  if (current_time >= ChopTime)
  {
    return STOP_PARSING;
  }
  return CONTINUE_PARSING;
}
#endif

void Chopper::MaskAll()
{
  for (int i=0; i<GetNumberOfThreads(); i++)
  {
    Buffer_t *buffer = GetBuffer(i);

    Mask_SetRegion (buffer, Buffer_GetHead(buffer), Buffer_GetTail(buffer), MASK_NOFLUSH);
  }
}

void Chopper::UnmaskAll(unsigned long long from_time, unsigned long long to_time)
{
  for (int i=0; i<GetNumberOfThreads(); i++)
  {
    Buffer_t *buffer    = GetBuffer(i);
    event_t  *ChopFrom  = NULL;
    event_t  *ChopUntil = NULL;

    ChopFrom  = FindCloserRunning(i, from_time, KEEP_ONGOING_STATE, false);
    ChopUntil = FindCloserRunning(i, to_time, DISCARD_ONGOING_STATE, true);

    if (ChopFrom != NULL && ChopUntil != NULL && ChopFrom != ChopUntil)
    {
      Mask_UnsetRegion (buffer, ChopFrom, ChopUntil, MASK_NOFLUSH);
    }
  }
}

