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

#ifndef MPI_COMUNICADORS
#define MPI_COMUNICADORS

#include "object_tree.h"

#define MPI_COMM_WORLD_ALIAS 1
#define MPI_COMM_SELF_ALIAS  2
#define MPI_NEW_INTERCOMM_ALIAS  3

typedef struct
{
  uintptr_t id;
  unsigned int num_tasks;
  int *tasks;
}
TipusComunicador;

void initialize_comunicadors (int n_ptasks);
void afegir_comunicador (TipusComunicador * comm, int ptask, int task);
int primer_comunicador (TipusComunicador * comm);
int seguent_comunicador (TipusComunicador * comm);
uintptr_t alies_comunicador (uintptr_t comid, int ptask, int task);
int numero_comunicadors (void);

void addInterCommunicator (uintptr_t InterCommID,
	uintptr_t CommID1, int leader1, uintptr_t CommID2, int leader2,
	int ptask, int task);
int getInterCommunicatorInfo (unsigned pos, uintptr_t *AliasInterComm,
	uintptr_t *AliasIntraComm1, int *leader1,
	uintptr_t *AliasIntraComm2, int *leader2);

#endif
