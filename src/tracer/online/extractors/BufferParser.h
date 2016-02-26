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

#ifndef __BUFFER_PARSER_H__
#define __BUFFER_PARSER_H__

#include <vector>

#include "events.h"
#include "record.h"
#include "buffers.h"

using std::vector;

#define CONTINUE_PARSING 0
#define STOP_PARSING    -1

class BufferParser
{
  public:
    BufferParser();
    ~BufferParser();

    int       GetNumberOfThreads();
    Buffer_t *GetBuffer( void );
    Buffer_t *GetBuffer( int thread_id );

    void         ParseBuffer( bool continue_from_checkpoint = false );
    void         ParseBuffer( int thread_id, bool continue_from_checkpoint = false );
    virtual void ParseBuffer( int thread_id, unsigned long long from, unsigned long long to, bool continue_from_checkpoint = false );

    virtual int  ParseEvent ( int thread_id, event_t *evt );

    bool isBurstBegin( event_t *evt );
    bool isBurstEnd  ( event_t *evt );
    bool isRunningBegin(int thread_id, event_t *evt);
    bool isRunningEnd  (int thread_id, event_t *evt);


  private:
    int               TotalThreads;
    BufferIterator_t *Checkpoint;

    BufferIterator_t *Get_Checkpoint();
    void Set_Checkpoint(BufferIterator_t *);
    void Clear_Checkpoint();
};

#endif /* __BUFFER_PARSER_H__ */
