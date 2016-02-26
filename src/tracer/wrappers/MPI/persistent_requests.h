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

#ifndef _PERSISTENT_REQUESTS

#define _PERSISTENT_REQUESTS
#include "queue.h"

#if !defined(MPI_SUPPORT) /* This shouldn't be compiled if MPI are not used */
# error "This should not be compiled outside MPI bounds"
#endif

#ifdef HAVE_MPI_H
# include <mpi.h>
#endif

typedef struct
{
  MPI_Request req;       /* Identificador */
  MPI_Datatype datatype; /* datatype of these elements */
  MPI_Comm comm;         /* Communicator identifier */
  int tipus;             /* Tipus d'operacio: ISEND/IBSEND/ISSEND/IRSEND/IRECV */
  int count;             /* num of elements in the transmission */
  int task;              /* source/destination */
  int tag;               /* Tag identifier */
} persistent_req_t;


typedef struct PR_Queue_t
{
  persistent_req_t *request;
  struct PR_Queue_t *next;
  struct PR_Queue_t *prev;
} PR_Queue_t;


persistent_req_t *PR_Busca_request (PR_Queue_t * cua, MPI_Request* reqid);

void PR_Elimina_request (PR_Queue_t * cua, MPI_Request* reqid);

void PR_NewRequest (int tipus, int count, MPI_Datatype datatype,
  int task, int tag, MPI_Comm comm, MPI_Request req, PR_Queue_t * cua);

void PR_queue_init (PR_Queue_t * cua);

#endif
