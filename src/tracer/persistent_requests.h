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

#ifndef _PERSISTENT_REQUESTS

#define _PERSISTENT_REQUESTS
#include "queue.h"

#if !defined(MPI_SUPPORT) && !defined(PACX_SUPPORT) /* This shouldn't be compiled if MPI or PACX are not used */
# error "This should not be compiled outside MPI/PACX bounds"
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

void PR_NewRequest (int tipus, void *buf, int count, MPI_Datatype datatype,
  int task, int tag, MPI_Comm comm, MPI_Request req, PR_Queue_t * cua);

void PR_queue_init (PR_Queue_t * cua);

#endif
