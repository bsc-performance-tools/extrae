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

#ifdef HAVE_STRING_H
# include <string.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif

#include "debug.h"
#include "queue.h"
#include "mpi_comunicadors.h"
#include "trace_to_prv.h"

typedef struct
{
	uintptr_t comms[2];
	int leaders[2];
	uintptr_t commid;
	uintptr_t alias;
} InterCommInfo_t;
static InterCommInfo_t *InterComm_global = NULL;
static unsigned num_InterCommunicators = 0;

typedef struct
{
	uintptr_t commid;
	uintptr_t alias;
} InterCommInfoAlias_t;
static InterCommInfoAlias_t ***Intercomm_ptask_task = NULL;
static unsigned **num_InterCommunicatorAlias = NULL;

static unsigned int num_comunicadors = 0;

#define ID_MINIM 1

typedef struct _CommInfo_t
{
  struct _CommInfo_t *next, *prev;

  TipusComunicador info;
} CommInfo_t;


typedef struct _CommAliasInfo_t
{
  struct _CommAliasInfo_t *next, *prev;
  uintptr_t commid_de_la_task;
  int alies;
} CommAliasInfo_t;


static CommInfo_t comunicadors; /* La llista de communicadors */
static CommAliasInfo_t **alies_comunicadors;    /* Llista alies per cada ptask-task */
static CommInfo_t *comm_actual = NULL;

static void afegir_alies (TipusComunicador * comm, CommInfo_t * info_com, int ptask, int task);

/*******************************************************************
 * initialize_comunicadors
 * -----------------
 *******************************************************************/

// #define DEBUG_COMMUNICATORS

void initialize_comunicadors (int n_ptasks)
{
	int ii;
	unsigned jj;

	/* Init intra-communicator */
	INIT_QUEUE (&comunicadors);

#if defined(DEBUG_COMMUNICATORS)
	fprintf (stderr, "DEBUG: Initializing communicators\n");
#endif

	alies_comunicadors = (CommAliasInfo_t **) malloc (n_ptasks * sizeof (CommAliasInfo_t *));
	ASSERT(alies_comunicadors!=NULL, "Not enough memory for intra-communicators alias");

	for (ii = 0; ii < n_ptasks; ii++)
	{
		ptask_t *ptask_info = GET_PTASK_INFO(ii+1);
		alies_comunicadors[ii] = (CommAliasInfo_t *) malloc (ptask_info->ntasks * sizeof (CommAliasInfo_t));
		ASSERT(alies_comunicadors[ii]!=NULL, "Not enough memory for intra-communicators alias");
	}

	/* Init inter-communicator */
	Intercomm_ptask_task = (InterCommInfoAlias_t ***) malloc (n_ptasks * sizeof (InterCommInfoAlias_t **));
	ASSERT(Intercomm_ptask_task!=NULL, "Not enough memory for inter-communicators alias");
	num_InterCommunicatorAlias = (unsigned **) malloc (n_ptasks * sizeof (unsigned *));
	ASSERT(num_InterCommunicatorAlias!=NULL, "Not enough memory for inter-communicators alias");
	for (ii = 0; ii < n_ptasks; ii++)
	{
		ptask_t *ptask_info = GET_PTASK_INFO(ii+1);

		Intercomm_ptask_task[ii] = (InterCommInfoAlias_t**) malloc (ptask_info->ntasks * sizeof (InterCommInfoAlias_t *));
		ASSERT(Intercomm_ptask_task[ii]!=NULL, "Not enough memory for inter-communicators alias");
		memset (Intercomm_ptask_task[ii], 0, ptask_info->ntasks*sizeof(InterCommInfoAlias_t*));

		num_InterCommunicatorAlias[ii] = (unsigned*) malloc (ptask_info->ntasks * sizeof(unsigned));
		ASSERT(num_InterCommunicatorAlias[ii]!=NULL, "Not enough memory for inter-communicators alias");
		memset (num_InterCommunicatorAlias[ii], 0, ptask_info->ntasks*sizeof(unsigned));
	}

	for (ii = 0; ii < n_ptasks; ii++)
	{
		ptask_t *ptask_info = GET_PTASK_INFO(ii+1);
		for (jj = 0; jj < ptask_info->ntasks; jj++)
			INIT_QUEUE (&(alies_comunicadors[ii][jj]));
	}
}


/*******************************************************************
 * alies_comunicador
 * -----------------
 * Retorna l'alias corresponent al comunicador donat
 *******************************************************************/
uintptr_t alies_comunicador (uintptr_t comid, int ptask, int task)
{
	unsigned u;
	CommAliasInfo_t *info;

#if defined(DEBUG_COMMUNICATORS)
	fprintf (stderr, "[DEBUG] alies_comunicador (%lu, %d, %d)\n", comid, ptask, task);
#endif

	ptask--;
	task--;

	for (info = GET_HEAD_ITEM (&(alies_comunicadors[ptask][task]));
		info != NULL;
		info = GET_NEXT_ITEM (&(alies_comunicadors[ptask][task]), info))
	{
		if (info->commid_de_la_task == comid)
			return info->alies;
	}

	for (u = 0; u < num_InterCommunicatorAlias[ptask][task]; u++)
		if (Intercomm_ptask_task[ptask][task][u].commid == comid)
			return Intercomm_ptask_task[ptask][task][u].alias;

	printf ("mpi2prv: Error: Cannot find : comid = %lu, ptask = %d, task = %d\n", comid, ptask, task);

	return 0;
}



/*******************************************************************
 * compara_comunicadors
 * --------------------
 * Retorna 1 si els dos comunicadors son iguals i 0 en cas contrari.
 *******************************************************************/
int compara_comunicadors (TipusComunicador * comm1, TipusComunicador * comm2)
{
  unsigned i, iguals;

  if (comm1->num_tasks != comm2->num_tasks)
    return 0;

  iguals = 1;
  i = 0;
  while (i < comm1->num_tasks && iguals)
  {
    if (comm1->tasks[i] != comm2->tasks[i])
      iguals = 0;
    else
      i++;
  }

  return iguals;
}

static void addInterCommunicatorAlias (uintptr_t InterCommID, uintptr_t alias,
	int ptask, int task)
{
	int found = FALSE;
	unsigned u, found_pos;

	for (u = 0; u < num_InterCommunicatorAlias[ptask][task]; u++)
	{
		found = Intercomm_ptask_task[ptask][task][u].commid = InterCommID;
		if (found)
		{
			found_pos = u;
			break;
		}
	}

	if (!found)
	{
		found_pos = num_InterCommunicatorAlias[ptask][task];
		num_InterCommunicatorAlias[ptask][task]++;
		Intercomm_ptask_task[ptask][task] = (InterCommInfoAlias_t*) realloc (
		  Intercomm_ptask_task[ptask][task],
		  sizeof(InterCommInfoAlias_t)*num_InterCommunicatorAlias[ptask][task]);
		ASSERT(NULL != Intercomm_ptask_task[ptask][task],
		  "Not enough memory for inter-communicators alias");

		Intercomm_ptask_task[ptask][task][found_pos].commid = InterCommID;
	}

	Intercomm_ptask_task[ptask][task][found_pos].alias = alias;
}

void addInterCommunicator (uintptr_t InterCommID,
	uintptr_t CommID1, int leader1, uintptr_t CommID2, int leader2,
	int ptask, int task)
{
	unsigned u, found_pos;
	int found = FALSE;

	CommID1 = alies_comunicador (CommID1, ptask, task);
	CommID2 = alies_comunicador (CommID2, ptask, task);

	ptask--;
	task--;

	for (u = 0; u < num_InterCommunicators; ++u)
	{
		found =  (InterComm_global[u].comms[0] == CommID1 && InterComm_global[u].comms[1] == CommID2)
		      || (InterComm_global[u].comms[1] == CommID1 && InterComm_global[u].comms[0] == CommID2);
		if (found)
		{
			found_pos = u;
			break;
		}
	}

	if (!found) 
	{
		found_pos = num_InterCommunicators;
		num_InterCommunicators++;
		InterComm_global = (InterCommInfo_t *) realloc (InterComm_global,
		  sizeof(InterCommInfo_t)*num_InterCommunicators);
		ASSERT(NULL != InterComm_global, "Not enough memory for inter-communicators alias");
		InterComm_global[found_pos].comms[0] = CommID1;
		InterComm_global[found_pos].comms[1] = CommID2;
		InterComm_global[found_pos].leaders[0] = leader1;
		InterComm_global[found_pos].leaders[1] = leader2;
		InterComm_global[found_pos].commid = InterCommID;
		InterComm_global[found_pos].alias = num_comunicadors + ID_MINIM;
		num_comunicadors++;
	}

	addInterCommunicatorAlias (InterCommID, InterComm_global[found_pos].alias,
	  ptask, task);
}


/*******************************************************************
 * afegir_comunicador
 * --------------------
 * Afegeix el comunicador donat a la llista de comunicadors.
 *******************************************************************/
void afegir_comunicador (TipusComunicador * comm, int ptask, int task)
{
	unsigned i;
  int trobat;
  CommInfo_t *info_com;

  ptask--;                      /* Han de comenc,ar per 0 */
  task--;

#if defined (DEBUG_COMMUNICATORS)
  fprintf (stderr, "%d,%d: Adding com id %lu\n", ptask, task, comm->id);
#endif

  trobat = 0;
  for (info_com = GET_HEAD_ITEM (&comunicadors); info_com != NULL;
       info_com = GET_NEXT_ITEM (&comunicadors, info_com))
    if (compara_comunicadors (&(info_com->info), comm))
    {
      trobat = 1;
      break;
    }

  if (!trobat)
  {
    info_com = (CommInfo_t *) malloc (sizeof (CommInfo_t));
    if (info_com == NULL)
    {
      fprintf (stderr, "mpi2prv: Error: Not enough memory! (%s:%d)\n",
				__FILE__,__LINE__);
      exit (1);
    }
#if defined (DEBUG_COMMUNICATORS)
	fprintf (stderr, "%d,%d: info_comm = %p\n", ptask, task, info_com);
#endif

		info_com->info.num_tasks = comm->num_tasks;
		info_com->info.tasks = (int *) malloc (info_com->info.num_tasks*sizeof(int));
		if (NULL == info_com->info.tasks)
		{
			fprintf (stderr, "mpi2prv: Error! Cannot add communicator alias\n");
			fflush (stderr);
			exit (-1);
		}
		for (i = 0; i < info_com->info.num_tasks; i++)
			info_com->info.tasks[i] = comm->tasks[i];

    info_com->info.id = num_comunicadors + ID_MINIM;
    ENQUEUE_ITEM (&comunicadors, info_com);
    num_comunicadors++;
  }

  /*
   * En qualsevol cas hem de guardar un alias per aquest identificador 
   */
  afegir_alies (comm, info_com, ptask, task);
}


/*******************************************************************
 * afegir_alies
 * --------------------
 * Afegeix o modifica l'alies donat
 *******************************************************************/
static void afegir_alies (TipusComunicador * comm, CommInfo_t * info_com,
                          int ptask, int task)
{
  int trobat;
  CommAliasInfo_t *info_alies;

  /*
   * ptask i task se suposa que comencen per 0 
   */
#if defined(DEBUG_COMMUNICATORS)
  fprintf (stderr, "DEBUG ptask %d, task %d: Adding communicator alias\n",
		ptask, task);
#endif

  trobat = 0;
  for (info_alies = GET_HEAD_ITEM (&(alies_comunicadors[ptask][task]));
       info_alies != NULL;
       info_alies =
       GET_NEXT_ITEM (&(alies_comunicadors[ptask][task]), info_alies))
    if (info_alies->commid_de_la_task == comm->id)
    {
      trobat = 1;
      break;
    }

  if (trobat)
  {
    /*
     * Cal modificar aquest alies 
     */
    info_alies->alies = info_com->info.id;
  }
  else
  {
    /*
     * Cal crear un nou alies 
     */
    info_alies = (CommAliasInfo_t *) malloc (sizeof (CommAliasInfo_t));
    if (info_alies == NULL)
    {
      fprintf (stderr, "mpi2prv: Error: Not enough memory! (%s:%d)\n",
				__FILE__,__LINE__);
      exit (1);
    }
    info_alies->commid_de_la_task = comm->id;
    info_alies->alies = info_com->info.id;
    ENQUEUE_ITEM (&(alies_comunicadors[ptask][task]), info_alies);
  }
#if defined(DEBUG_COMMUNICATORS)
  fprintf (stderr, "      id %lu -> %d\n", comm->id, info_alies->alies);
#endif
}



/*******************************************************************
 * primer_comunicador
 * --------------------
 * Copia al punter donat el primer comunicador de la llista (si
 * n'hi ha algun) i retorna un enter que indica si n'hi havia algun.
 *******************************************************************/
int primer_comunicador (TipusComunicador * comm)
{
  comm_actual = GET_HEAD_ITEM (&comunicadors);
  if (comm_actual != NULL)
  {
    memcpy (comm, &(comm_actual->info), sizeof (TipusComunicador));
    return 0;
  }
  else
    return -1;
}


/*******************************************************************
 * seguent_comunicador
 * --------------------
 * Copia al punter donat el seguent comunicador de la llista
 * respecte l'ultim que s'havia retornat (si n'hi ha algun) i
 * retorna un enter que indica si n'hi havia algun.
 *******************************************************************/
int seguent_comunicador (TipusComunicador * comm)
{
  comm_actual = GET_NEXT_ITEM (&comunicadors, comm_actual);
  if (comm_actual != NULL)
  {
    memcpy (comm, &(comm_actual->info), sizeof (TipusComunicador));
    return 0;
  }
  return -1;
}

int getInterCommunicatorInfo (unsigned pos, uintptr_t *AliasInterComm,
	uintptr_t *AliasIntraComm1, int *leaderComm1,
	uintptr_t *AliasIntraComm2, int *leaderComm2)
{
	if (pos >= num_InterCommunicators)
		return 0;

	*AliasInterComm = InterComm_global[pos].alias;
	*AliasIntraComm1 = InterComm_global[pos].comms[0];
	*leaderComm1 = 1+InterComm_global[pos].leaders[0];
	*AliasIntraComm2 = InterComm_global[pos].comms[1];
	*leaderComm2 = 1+InterComm_global[pos].leaders[1];

	return 1;
}


/*******************************************************************
 * numero_comunicadors
 * -------------------
 * Retorna el numero de comunicadors de la llista
 *******************************************************************/
int numero_comunicadors (void)
{
  return num_comunicadors;
}
