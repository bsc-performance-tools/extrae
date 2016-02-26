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
#if defined(MPI_HAS_INIT_THREAD_C) || defined(MPI_HAS_INIT_THREAD_F)
# ifdef HAVE_PTHREAD_H
#  include <pthread.h>
# endif
#endif

#include "persistent_requests.h"
#include "wrapper.h"

#if !defined(MPI_SUPPORT) /* This shouldn't be compiled if MPI is not used */
# error "This should not be compiled outside MPI bounds"
#endif

#if defined(MPI_HAS_INIT_THREAD_C) || defined(MPI_HAS_INIT_THREAD_F)
pthread_mutex_t pr_lock = PTHREAD_MUTEX_INITIALIZER;
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

#if defined(MPI_HAS_INIT_THREAD_C) || defined(MPI_HAS_INIT_THREAD_F)
  pthread_mutex_lock(&pr_lock);
#endif
  element_cua = PR_QueueSearch (cua, reqid);
  if (element_cua == NULL)
  {
#if defined(MPI_HAS_INIT_THREAD_C) || defined(MPI_HAS_INIT_THREAD_F)
  pthread_mutex_unlock(&pr_lock);
#endif
    return;
  }
  free (element_cua->request);
  REMOVE_ITEM (element_cua);
  free (element_cua);
#if defined(MPI_HAS_INIT_THREAD_C) || defined(MPI_HAS_INIT_THREAD_F)
  pthread_mutex_unlock(&pr_lock);
#endif
}

void PR_NewRequest (int tipus, int count, MPI_Datatype datatype, int task,
	int tag, MPI_Comm comm, MPI_Request req, PR_Queue_t* cua)
{
  persistent_req_t *nova_pr;
  PR_Queue_t *nou_element_cua;

#if defined(MPI_HAS_INIT_THREAD_C) || defined(MPI_HAS_INIT_THREAD_F)
  pthread_mutex_lock(&pr_lock);
#endif
  /*
   * Es reserva memoria per la nova request 
   */
  nova_pr = (persistent_req_t *) malloc (sizeof (persistent_req_t));

	if (nova_pr == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": ERROR! Cannot allocate memory for a new persistent request!\n");
#if defined(MPI_HAS_INIT_THREAD_C) || defined(MPI_HAS_INIT_THREAD_F)
        pthread_mutex_unlock(&pr_lock);
#endif
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
		fprintf (stderr, PACKAGE_NAME": ERROR! Cannot add a new persistent request to the queue of requests!\n");
#if defined(MPI_HAS_INIT_THREAD_C) || defined(MPI_HAS_INIT_THREAD_F)
        pthread_mutex_unlock(&pr_lock);
#endif
		return;
	}
  nou_element_cua->request = nova_pr;
  INSERT_ITEM_INCREASING (cua, nou_element_cua, PR_Queue_t, request->req);
#if defined(MPI_HAS_INIT_THREAD_C) || defined(MPI_HAS_INIT_THREAD_F)
  pthread_mutex_unlock(&pr_lock);
#endif
}
