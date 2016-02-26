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

#ifndef MPI_PRV_EVENTS_H
#define MPI_PRV_EVENTS_H

#include <config.h>

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif

#include "common.h"

/*************************************************************************
 * S'utilitza el format nou dels .prv, que genera diferents tipus i estan
 * definits en un fitxer de constants.
 *************************************************************************/

void Enable_MPI_Soft_Counter(unsigned int EvType);

/* S'afegeix el fitxer on hi ha totes les constants */
#include "MPI_EventEncoding.h"

#define MPITYPE_FLAG_COLOR 9

#define NUM_MPI_BLOCK_GROUPS  8     /* Dels 12, de moment nomes 8 son diferents */

void SoftCountersEvent_WriteEnabled_MPI_Operations(FILE *fd);
void MPITEvent_WriteEnabled_MPI_Operations (FILE * fd);
void Enable_MPI_Operation (int tmpit);

#if defined(PARALLEL_MERGE)
void Share_MPI_Softcounter_Operations (void);
void Share_MPI_Operations (void);
#endif

void Translate_MPI_MPIT2PRV (int typempit, UINT64 valuempit, int *typeprv, UINT64 *valueprv);

#endif
