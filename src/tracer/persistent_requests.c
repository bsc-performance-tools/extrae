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
#include "common.h"

static char UNUSED rcsid[] = "$Id$";

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#include "persistent_requests.h"
#include "wrapper.h"

#if !defined(MPI_SUPPORT) /* This shouldn't be compiled if MPI is not used */
# error "This should not be compiled outside MPI bounds"
#endif

void PR_queue_init (PR_Queue_t * cua)
{
  INIT_QUEUE (cua);
}

PR_Queue_t *PR_QueueSearch (PR_Queue_t * queue, MPI_Request* reqid)
{
  PR_Queue_t *ptmp;

  for (ptmp = (queue)->next; ptmp != (queue); ptmp = ptmp->next)
    if (ptmp->request->req == *reqid)
      return (ptmp);
  return (NULL);
}

persistent_req_t *PR_Busca_request (PR_Queue_t * cua, MPI_Request* reqid)
{
  PR_Queue_t *element_cua;

  element_cua = PR_QueueSearch (cua, reqid);
  if (element_cua == NULL)
    return (NULL);
  return (element_cua->request);
}

void PR_Elimina_request (PR_Queue_t * cua, MPI_Request* reqid)
{
  PR_Queue_t *element_cua;

  element_cua = PR_QueueSearch (cua, reqid);
  if (element_cua == NULL)
    return;
  free (element_cua->request);
  REMOVE_ITEM (element_cua);
  free (element_cua);
}

void PR_NewRequest (int tipus, void *buf, int count, MPI_Datatype datatype, int task,
	int tag, MPI_Comm comm, MPI_Request req, PR_Queue_t* cua)
{
  persistent_req_t *nova_pr;
  PR_Queue_t *nou_element_cua;

  /*
   * Es reserva memoria per la nova request 
   */
  nova_pr = (persistent_req_t *) malloc (sizeof (persistent_req_t));

	if (nova_pr == NULL)
	{
		fprintf (stderr, "mpitrace: ERROR! Cannot allocate memory for a new persistent request!\n");
		return;
	}

  /*
   * Se li assignen les dades donades 
   */
  nova_pr->req = req;
  nova_pr->tipus = tipus;
  nova_pr->count = count;
  nova_pr->datatype = datatype;
  nova_pr->task = task;
  nova_pr->tag = tag;
  nova_pr->comm = comm;

  /*
   * S'afegeix la request a la col.lecció 
   */
	nou_element_cua = (PR_Queue_t *) malloc (sizeof (PR_Queue_t));
	if (nou_element_cua == NULL)
	{
		fprintf (stderr, "mpitrace: ERROR! Cannot add a new persistent request to the queue of requests!\n");
		return;
	}
  nou_element_cua->request = nova_pr;
  INSERT_ITEM_INCREASING (cua, nou_element_cua, PR_Queue_t, request->req);
}
