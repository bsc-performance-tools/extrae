/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *                                                             ___           *
 *   +---------+     http:// www.cepba.upc.edu/tools_i.htm    /  __          *
 *   |    o//o |     http:// www.bsc.es                      /  /  _____     *
 *   |   o//o  |                                            /  /  /     \    *
 *   |  o//o   |     E-mail: cepbatools@cepba.upc.edu      (  (  ( B S C )   *
 *   | o//o    |     Phone:          +34-93-401 71 78       \  \  \_____/    *
 *   +---------+     Fax:            +34-93-401 25 77        \  \__          *
 *    C E P B A                                               \___           *
 *                                                                           *
 * This software is subject to the terms of the CEPBA/BSC license agreement. *
 *      You must accept the terms of this license to use this software.      *
 *                                 ---------                                 *
 *                European Center for Parallelism of Barcelona               *
 *                      Barcelona Supercomputing Center                      *
\*****************************************************************************/

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL$
 | 
 | @last_commit: $Date$
 | @version:     $Revision$
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

/*************************************************************************
 ** MPI_Physical_Info.h
 ** Per HSG. -> informació basica per tenir comunicacions fisiques reals
 ** a la trasa de Paraver en el cas d'emprar MPICH!
 *************************************************************************/

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
