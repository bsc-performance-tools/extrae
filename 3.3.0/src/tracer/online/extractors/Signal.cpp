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

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#include <iostream>
#include <spectral-api.h>
#include "Signal.h"
#include "tags.h"

/**
 * Empty signal constructor.
 */
Signal::Signal()
{
  SpectralSignal = NULL;
}

/**
 * Signal constructor from a file.
 */
Signal::Signal(string file)
{
  SpectralSignal = Spectral_LoadSignal((char *)file.c_str());
}

/**
 * Signal constructor from the given bursts information.
 *
 * @param bursts A container of bursts.
 */
Signal::Signal(Bursts *bursts)
{
  /* Allocate the signal */
  SpectralSignal = Spectral_AllocateSignal( bursts->GetNumberOfBursts() );

  /* Insert each burst as a signal point */
  for (int i=0; i<bursts->GetNumberOfBursts(); i++)
  {
    Spectral_AddPoint3(
      SpectralSignal,
      bursts->GetBurstTime(i),
      bursts->GetBurstDuration(i),
      bursts->GetBurstDuration(i) / 1000000.0);
  }
}

/**
 * Signal constructor from an MRNet message.
 *
 * @param InputPacket The packet that contains the signal serialized.
 */
Signal::Signal(PACKET_PTR InputPacket)
{
  SpectralSignal = NULL;
  Unpack(InputPacket);
}

/**
 * Destructor. Frees the signal.
 */
Signal::~Signal()
{
  Spectral_FreeSignal( SpectralSignal );
}

/**
 * @return the signal.
 */
signal_t * Signal::GetSignal()
{
  return SpectralSignal;
}

/**
 * @return the size of the signal.
 */
int Signal::GetSize()
{
  return Spectral_GetSignalSize(SpectralSignal);
}

/**
 * Adds the current signal to the given list of signals.
 *
 * @param SignalsList The list of signals to add to this signal.
 */
void Signal::Sum(vector<Signal *> SignalsList)
{ 
  int i           = 0;
  int num_signals = 0;
  signal_t **all_signals = NULL;
  signal_t  *SumSignal   = NULL;
  
  num_signals = SignalsList.size();
  all_signals = (signal_t **)malloc( (num_signals + 1) * sizeof (signal_t *));
  for (i=0; i<num_signals; i++)
  {
    all_signals[i] = SignalsList[i]->GetSignal();
  }

  if (SpectralSignal != NULL) 
  { 
    /* This signal is summed with all those in the list */
    all_signals[i] = this->GetSignal();
    num_signals ++;
  }

  SumSignal = Spectral_AddSortedN(num_signals, all_signals);

  free(all_signals);
  Spectral_FreeSignal( SpectralSignal );
  SpectralSignal = SumSignal;
}

/**
 * Serializes the signal object through the given MRNet stream. Used in the BEs.
 *
 * @param OutputStream The stream used to send a message that contains the signal serialized.
 */
void Signal::Serialize(STREAM *OutputStream)
{
  spectral_time_t  *Times  = NULL;
  spectral_time_t  *Deltas = NULL;
  spectral_value_t *Values = NULL;
  int SignalSize = 0;

  Spectral_CompressSignal( &SpectralSignal, 10000 );

  SignalSize = Spectral_SerializeSignal( SpectralSignal, &Times, &Deltas, &Values );

  MRN_STREAM_SEND(OutputStream, REDUCE_SIGNAL, SIGNAL_XMIT_FORMAT,
    Times,  SignalSize,
    Deltas, SignalSize,
    Values, SignalSize);

  free(Times);
  free(Deltas);
  free(Values);
}

/**
 * Serializes the signal object through the given MRNet stream. Used in the CPs.
 *
 * @param OutputStreamID The stream identifier.
 * @return An MRNet packet that contains the signal serialized.
 */
PACKET_PTR Signal::Serialize(int OutputStreamID )
{
  spectral_time_t  *Times  = NULL;
  spectral_time_t  *Deltas = NULL;
  spectral_value_t *Values = NULL;
  int SignalSize = 0;

  Spectral_CompressSignal( &SpectralSignal, 10000 );

  SignalSize = Spectral_SerializeSignal( SpectralSignal, &Times, &Deltas, &Values);

  PACKET_PTR new_packet( new Packet( OutputStreamID, REDUCE_SIGNAL, SIGNAL_XMIT_FORMAT,
    Times,  SignalSize,
    Deltas, SignalSize,
    Values, SignalSize ) );
  new_packet->set_DestroyData(true);

  return new_packet;
}

/**
 * De-serializes the signal contained in an MRNet package.
 *
 * @param InputPacket The MRNet packet that contains the signal serialized.
 */
void Signal::Unpack(PACKET_PTR InputPacket)
{
  spectral_time_t  *Times  = NULL;
  spectral_time_t  *Deltas = NULL;
  spectral_value_t *Values = NULL;
  int SignalSize = 0;

  PACKET_unpack(InputPacket, SIGNAL_XMIT_FORMAT,
   &Times,  &SignalSize,
   &Deltas, &SignalSize,
   &Values, &SignalSize);

  SpectralSignal = Spectral_AssembleSignal(SignalSize, Times, Deltas, Values);

  free(Times);
  free(Deltas);
  free(Values);
}

