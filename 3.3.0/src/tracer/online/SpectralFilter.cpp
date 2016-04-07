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

#include <iostream>
#include <vector>

using std::vector;

#include <MRNet_wrappers.h>

#include "SpectralFilter.h"
#include "Signal.h"
#include "tags.h"


extern "C" {

const char *filterOnlineSpectral_format_string = "";

void filterOnlineSpectral( 
  std::vector< PacketPtr >& packets_in,
  std::vector< PacketPtr >& packets_out,
  std::vector< PacketPtr >& /* packets_out_reverse */,
  void ** /* client data */,
  UNUSED PacketPtr& params,
  const TopologyLocalInfo& top_info)
{
  int tag = packets_in[0]->get_Tag();

  /* Bypass the implicit filter in the back-ends, there's nothing to merge at this level! */
  if (BOTTOM_FILTER(top_info))
  {
    for (unsigned int i=0; i<packets_in.size(); i++)
    {
      packets_out.push_back(packets_in[i]);
    }
    return;
  }

  /* Process the packets crossing the filter */
  switch(tag)
  {
    case REDUCE_SIGNAL:
    {
      /* Load N signals up to a maximum chunk size and add them, until all signals are processed */
      unsigned int nextSignal = 0;
      Signal *SumSignal = new Signal();

      while (nextSignal < packets_in.size())
      {
        vector<Signal *> ChildrenSignals;
        int ChunkSize = SumSignal->GetSize();
        do
        {
          /* Unpack next child's signal */
          PACKET_PTR cur_packet = packets_in[nextSignal];
          Signal *child_Signal = new Signal( cur_packet );

          /* Save the signal in a list */
          ChildrenSignals.push_back( child_Signal );
          
          ChunkSize  += child_Signal->GetSize();
          nextSignal ++;
        } while ((ChunkSize < MAX_SIGNAL_CHUNK_SIZE) && (nextSignal < packets_in.size()));
 
        /* Sum the signals unpacked so far */
        SumSignal->Sum(ChildrenSignals);
      }

      /* Send the summed signal to the upper level of the network */
      packets_out.push_back( SumSignal->Serialize( packets_in[0]->get_StreamId() ) );

      break;
    }
    default:
    {
      /* Bypass all other messages to the parent node */
      for (unsigned int i=0; i<packets_in.size(); i++)
      {
        packets_out.push_back( packets_in[i] );
      }
      break;
    }
  }
}

} /* extern "C" */
