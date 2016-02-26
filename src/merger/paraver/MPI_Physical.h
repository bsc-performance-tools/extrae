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

#ifndef _MPI_PHYSICAL_HEADER_
#define  _MPI_PHYSICAL_HEADER_

#ifdef MPI_PHYSICAL_COMM

#include "MPI_EventEncoding.h"

#ifndef TRUE
#define TRUE (1==1)
#endif

#ifndef FALSE
#define FALSE (0==1)
#endif

typedef struct
{
  iotimer_t Temps;
  iotimer_t Temps_entrada;
  int Valid;
}
InformacioFisica;

extern int *num_phys_Sends, *num_phys_Receives;
extern InformacioFisica **phys_Sends, **phys_Receives;

/* Event que identifica quan hi ha el send fisic */
/* Un send fisic sempre estara enmig d'un event d'inici rutina de send i 
   de fi de rutina de send, per la qual cosa no hi ha gaire problema per
   establir un vincle entre send fisic i send logic */
#define           TAG_SND_FISIC   4000

/* Event que identifica un receive logic */
#define           TAG_RCV_FISIC   4001

/* Event que permet fer matching entre recv fisic i recv logic */
/* En teoria, s'ha de fer un matching entre un  TAG_RCV_FISIC i un TAG_RCV_F_L
   per tal de saber quina parella d'events formen la comunicacio fisica i
   logica */
#define           TAG_RCV_F_L     4002

#endif

#endif
