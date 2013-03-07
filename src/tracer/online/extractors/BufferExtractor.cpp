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

#include "BufferExtractor.h"
#include "trace_buffers.h"
#include "threadid.h"
#include "clock.h"

BufferExtractor::BufferExtractor()
{
  /* By default data is retrieved from thread 0 */
  ExtractionBuffer = TRACING_BUFFER(0); 
}

/**
 * Parses the whole extraction buffer. For each event, calls the method ProcessEvent, 
 * which is defined in the derived class.
 */
void BufferExtractor::ExtractAll()
{
  Extract(0, TIME);
}

/**
 * Parses the extraction buffer in the time range specified between from and to.
 * For each event, calls the method ProcessEvent, which is defined in the derived
 * class.
 *
 * @param from The starting timestamp.
 * @param to   The ending timestamp.
 */
void BufferExtractor::Extract(unsigned long long from, unsigned long long to)
{
  int buf_size = Buffer_GetFillCount(ExtractionBuffer);

  if (buf_size <= 0) return;

  ExtractionIterator = BIT_NewRange(ExtractionBuffer, from, to);
  
  while (!BIT_OutOfBounds(ExtractionIterator))
  {
    event_t *current_evt = BIT_GetEvent(ExtractionIterator);

    ProcessEvent(current_evt);

    BIT_Next(ExtractionIterator);
  }

  BIT_Free(ExtractionIterator);
}

/**
 * Phony implementation of the ProcessEvent method that does nothing.
 */
void BufferExtractor::ProcessEvent(UNUSED event_t *current_evt)
{
}


