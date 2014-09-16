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

#if defined(HAVE_STDLIB_H)
# include <stdlib.h>
#endif
#include "Chopper.h"
#include "events.h"

/**
 * Constructor. Creates a new forward iterator for the extraction buffer.
 */
Chopper::Chopper()
{
  ExtractionIterator = BIT_NewForward(ExtractionBuffer);
}


/**
 * Destructor. Frees the iterator.
 */
Chopper::~Chopper()
{
  BIT_Free(ExtractionIterator);
}

/**
 * Selects the event closest to the specified timestamp. 
 *
 * \param time_to_find  The timestamp we're looking for.
 * \param inclusively   If there's no event in that exact timestamp and set to true, 
 *                      returns the previous event before the given time; and the next otherwise.
 * \return The iterator pointing to the event that is closer to the given timestamp.
 */
BufferIterator_t * Chopper::DontBreakStates(unsigned long long time_to_find, bool inclusively)
{
  BufferIterator_t *found=NULL, *prev=NULL;
  event_t *curEvt = NULL;
  
  while ((!BIT_OutOfBounds(ExtractionIterator)) && (found == NULL))
  {
    curEvt = BIT_GetEvent(ExtractionIterator);

    /* Select the begin time inclusively (don't break states) */
    if (Get_EvTime(curEvt) < time_to_find)
    {
      if ((prev == NULL) || (Get_EvTime(curEvt) > Get_EvTime(BIT_GetEvent(prev))))
      {
        BIT_Free(prev);
        prev = BIT_Copy(ExtractionIterator);
      }
      BIT_Next(ExtractionIterator);
    }
    else if (Get_EvTime(curEvt) == time_to_find) found = BIT_Copy(ExtractionIterator);
    else
    {
      if (prev != NULL)
      {
#if 0
        /* Decide whether we move to the next or previous state, depending on which is closer */
        if ((Get_EvTime(curEvt) - time_to_find) < (time_to_find - Get_EvTime(BIT_GetEvent(prev))))
          found = BIT_Copy(ExtractionIterator);
        else
          found = prev;
#endif
        if (inclusively)
          found = prev;
        else
          found = BIT_Copy(ExtractionIterator);
      }
      else found = BIT_Copy(ExtractionIterator);
    }
  }
  return found;
}

/*
unsigned long long Chopper::DontBreakStates(unsigned long long time_to_find, bool inclusively)
{
  BufferIterator_t *it = DontBreakStates(time_to_find, inclusively);
  if (!BIT_OutOfBounds(it))
  {
    return Get_EvTime( BIT_GetEvent(it) );  
  }
  else return time_to_find;
}
*/

/**
 * Select the event before the specified timestamp. 
 * \param time_to_find  The timestamp we're looking for.
 * \return The iterator pointing to the event that is closer to the given timestamp.
 */
BufferIterator_t * Chopper::RemoveLastState(unsigned long long time_to_find)
{
  BufferIterator_t *found=NULL, *prev=NULL;
  event_t *curEvt=NULL;

  while ((!BIT_OutOfBounds(ExtractionIterator)) && (found == NULL))
  {
    curEvt = BIT_GetEvent(ExtractionIterator);

    /* Select the end time exclusively (remove last state) */
    if (Get_EvTime(curEvt) <= time_to_find) BIT_Next(ExtractionIterator);
    else found = prev;

    BIT_Free(prev);
    prev = BIT_Copy(ExtractionIterator);
  }
  return found;
}

BufferIterator_t * Chopper::FindCloserRunning(unsigned long long time_to_find)
{
  BufferIterator_t *last_running_start = NULL, *next_running_start = NULL;

  while (!BIT_OutOfBounds(ExtractionIterator))
  {
    event_t *current_event = BIT_GetEvent( ExtractionIterator );
    unsigned long long current_time = Get_EvTime(current_event);

    if (isBurstEnd(current_event)) 
    {
      BIT_Free( last_running_start );
      last_running_start = BIT_Copy( ExtractionIterator );
    }

    if (current_time >= time_to_find)
    {
      return last_running_start;
    }

    BIT_Next( ExtractionIterator );
  }
  return NULL;
}

event_t *Chopper::EventCloserRunning( unsigned long long time_to_find )
{
  BufferIterator_t *it = FindCloserRunning( time_to_find );
  if (!BIT_OutOfBounds( it ))
  {
    return BIT_GetEvent( it );
  }
  else return NULL;
}

unsigned long long Chopper::TimeCloserRunning( unsigned long long time_to_find )
{
  BufferIterator_t *it = FindCloserRunning( time_to_find );
  if (!BIT_OutOfBounds( it ))
  {
    return Get_EvTime( BIT_GetEvent( it ) );
  }
  else return time_to_find;
}


/*
unsigned long long Chopper::RemoveLastState(unsigned long long time_to_find)
{
  BufferIterator_t *it = RemoveLastState(time_to_find);
  if (!BIT_OutOfBounds(it))
  {
    return Get_EvTime( BIT_GetEvent(it) );
  }
  else return time_to_find;
}
*/

#if 0

Rewind()
{
  pon el iter al principio
}

CHOP(from, to)
{
  busca la primera salida de mpi
  y coge de ahi hasta la ultima salida de mpi

  retorna los eventos de esos puntos

  conforme avanzas por los eventos, que pasa con los contadores???

}

#endif


/** 
 * Select a region in the buffer between two given times. The iterator is never rewound, so 
 * subsequent calls to Chop() with progressive, non-overlapping time intervals, do not 
 * start parsing the buffer from the beginning. 
 *
 * \param fromTime The starting timestamp.
 * \param toTime   The ending timestamp.
 *
 * @return firstEv The event at the starting timestamp.
 * @return lastEv  The event at the ending timestamp.
 */
void Chopper::Chop(unsigned long long fromTime, unsigned long long toTime, event_t *&firstEv, event_t *&lastEv)
{
  BufferIterator_t *firstIt = NULL;
  BufferIterator_t *lastIt  = NULL;

  /* Find the first event of the region */
  firstIt = DontBreakStates(fromTime, true);
  if (!BIT_OutOfBounds(firstIt))
  {
    firstEv = BIT_GetEvent(firstIt);

#if USE_HARDWARE_COUNTERS
    /* Look for the first event that emits counters in the chopped region */
    event_t *first_with_hwcs = firstEv;
    while ((!Get_EvHWCRead(first_with_hwcs)) && (!BIT_OutOfBounds(firstIt)))
    {
      BIT_Next(firstIt);
      first_with_hwcs = BIT_GetEvent(firstIt);
    }
    Reset_EvHWCs(first_with_hwcs); /* The first event that reads counters after the chop sets them to 0 */
#endif

    /* Find the last event of the region */
    lastIt = RemoveLastState(toTime);
//    lastIt = DontBreakStates(toTime, true);
    lastEv = BIT_GetEvent(lastIt);
  
    BIT_Free(firstIt);
    BIT_Free(lastIt);
  }
}

