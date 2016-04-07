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

#ifndef __CHOPPER_H__
#define __CHOPPER_H__

#include "BufferParser.h"

#define KEEP_ONGOING_STATE    0
#define DISCARD_ONGOING_STATE 1

class Chopper : public BufferParser
{
  public:
    Chopper();

    event_t * FindCloserRunning(int thread_id, unsigned long long time_to_chop, int chop_type = KEEP_ONGOING_STATE, bool use_checkpoint = false);
    unsigned long long FindCloserRunningTime(int thread_id, unsigned long long time_to_chop, int chop_type = KEEP_ONGOING_STATE, bool use_checkpoint = false);

    void MaskAll();
    void UnmaskAll(unsigned long long from_time, unsigned long long to_time);

  private:
    unsigned long long ChopTime;
    int                ChopType;
    event_t           *ChopAt;
    event_t           *LastRunningBeginEv, *LastRunningEndEv;
    event_t           *PrevRunningBeginEv, *PrevRunningEndEv;

    int ParseEvent(int thread_id, event_t *current_event);



#if 0
  public:
    Chopper();
    ~Chopper();

    void Chop(
      unsigned long long fromTime, 
      unsigned long long toTime,
      event_t *&firstEv,
      event_t *&lastEv);

    BufferIterator_t * FindCloserRunning(unsigned long long time_to_find);
    unsigned long long TimeCloserRunning(unsigned long long time_to_find);
    event_t *EventCloserRunning( unsigned long long time_to_find );

    BufferIterator_t * DontBreakStates(unsigned long long time_to_find, bool inclusively);
    BufferIterator_t * RemoveLastState(unsigned long long time_to_find);
/*
    unsigned long long GetDontBreakStates(unsigned long long time_to_find, bool inclusively);
    unsigned long long RemoveLastState(unsigned long long time_to_find);
*/
#endif
};

#endif /* __CHOPPER_H__ */
