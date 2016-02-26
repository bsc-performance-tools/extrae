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

#include <cstdlib>

#include "BufferParser.h"
#include "trace_buffers.h"
#include "threadid.h"
#include "clock.h"

/* Oh the ugliness! Should include wrapper.h, but that brings a lot of unnecessary deps.
 * Instead, remove that routine from wrapper.c and put it in a separate module that makes actual sense.
 */
extern "C" {
  extern unsigned Backend_getMaximumOfThreads();
}

BufferParser::BufferParser()
{
  TotalThreads = Backend_getMaximumOfThreads();
  Checkpoint   = NULL;
}

BufferParser::~BufferParser()
{
  TotalThreads = 0;
  Clear_Checkpoint();
}

Buffer_t * BufferParser::GetBuffer()
{
  return GetBuffer(0);
}

int BufferParser::GetNumberOfThreads()
{
  return TotalThreads;
}

Buffer_t * BufferParser::GetBuffer(int thread_id)
{
  if ((thread_id >= 0) && (thread_id < TotalThreads))
  {
    return TRACING_BUFFER(thread_id);
  }
  else
  {
    return NULL;
  }
}

BufferIterator_t * BufferParser::Get_Checkpoint()
{
  return Checkpoint;
}

void BufferParser::Clear_Checkpoint()
{
  BIT_Free(Checkpoint);
  Checkpoint = NULL;
}

void BufferParser::Set_Checkpoint(BufferIterator_t *it)
{
  Checkpoint = BIT_Copy(it);
}

/**
 * Parses all the events of the master buffer. 
 * For each event, calls the method ProcessEvent, which is defined in the derived class.
 */
void BufferParser::ParseBuffer(bool continue_from_checkpoint)
{
  ParseBuffer(0, 0, TIME, continue_from_checkpoint);
}

/**
 * Parses all the events of the buffer specified. 
 * For each event, calls the method ProcessEvent, which is defined in the derived class.
 */
void BufferParser::ParseBuffer(int thread_id, bool continue_from_checkpoint)
{
  ParseBuffer(thread_id, 0, TIME, continue_from_checkpoint);
}

/**
 * Parses the specified buffer in the time range between from and to.
 * For each event, calls the method ParseEvent, which is defined in the derived
 * class. If this method returns -1, the parsing is interrupted.
 *
 * @param from The starting timestamp.
 * @param to   The ending timestamp.
 */
#include <stdio.h>
void BufferParser::ParseBuffer(int thread_id, unsigned long long from, unsigned long long to, bool continue_from_checkpoint)
{
  int               rc          = 0;
  Buffer_t         *buffer      = GetBuffer(thread_id);
  int               buffer_size = Buffer_GetFillCount(buffer);
  BufferIterator_t *it          = NULL;

  if (buffer_size <= 0) return;

  if (continue_from_checkpoint)
  {
    it = Get_Checkpoint();
  }
  
  if ((it == NULL) || (!continue_from_checkpoint))
  {
    Clear_Checkpoint();
    it = BIT_NewRange(buffer, from, to);
  }
  
  while ((!BIT_OutOfBounds(it)) && (rc != -1))
  {
    event_t *current_evt = BIT_GetEvent(it);

    rc = ParseEvent(thread_id, current_evt);

    BIT_Next(it);

    Set_Checkpoint( it );
  }

  BIT_Free(it);
}

/**
 * Phony implementation of the ProcessEvent method that does nothing.
 */
int BufferParser::ParseEvent(UNUSED int thread_id, UNUSED event_t *current_event)
{
  return 0;
}

/**
 * Returns true if the given event marks the start of a burst.
 *
 * @param evt A buffer event.
 * @return true if delimits the start of a burst; false otherwise.
 */
bool BufferParser::isBurstBegin(event_t *evt)
{
  int type  = Get_EvEvent(evt);
  int value = Get_EvValue(evt);

  return (((IsMPI(type)) && (type != MPI_IRECVED_EV) && (type != MPI_PERSIST_REQ_EV) && (type != MPI_TEST_COUNTER_EV) && (type != MPI_IPROBE_COUNTER_EV) && (value == EVT_END)) ||
          ((IsBurst(type)) && (value == EVT_BEGIN)));
}

/**
 * Returns true if the given event marks the end of a burst.
 *
 * @param evt A buffer event.
 * @return true if delimits the end of a burst; false otherwise.
 */
bool BufferParser::isBurstEnd(event_t *evt)
{
  int type  = Get_EvEvent (evt);
  int value = Get_EvValue (evt);

  return (((IsMPI(type)) && (type != MPI_IRECVED_EV) && (type != MPI_PERSIST_REQ_EV) && (type != MPI_TEST_COUNTER_EV) && (type != MPI_IPROBE_COUNTER_EV) && (value == EVT_BEGIN)) ||
          ((IsBurst(type)) && (value == EVT_END)));
}

bool BufferParser::isRunningBegin(int thread_id, event_t *evt)
{
  if (thread_id == 0)
  {
    return isBurstBegin(evt);
  }
  else
  {
    int type  = Get_EvEvent (evt);
    int value = Get_EvValue (evt);

    return (((type == OMPFUNC_EV) || (type == TASKFUNC_EV)) && (value != EVT_END));
  }
}

bool BufferParser::isRunningEnd(int thread_id, event_t *evt)
{
  if (thread_id == 0)
  {
    return isBurstEnd(evt);
  }
  else
  {
    int type  = Get_EvEvent (evt);
    int value = Get_EvValue (evt);

    return (((type == OMPFUNC_EV) || (type == TASKFUNC_EV)) && (value == EVT_END));
  }
}

