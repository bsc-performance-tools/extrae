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

#ifdef HAVE_MPI_H
# include <mpi.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#include "Syncing.h"

/**
 * Passes the information from the connections file (read only by the root task) to 
 * the rest of tasks so that the back-ends know where to attach to start the MRNet.
 *
 * @param rank           This task's rank.
 * @param root           The root task that reads the connections file.
 * @param rdy_to_connect Only defined in the root task, tells if the file was read ok.
 * @param sendbuf        Only defined in the root task, contains the information to distribute.
 * @param sendcnts       Only defined in the root task, contains the information to distribute.
 * @param displs         Only defined in the root task, contains the information to distribute.
 * @param parHostname    Output parameter telling to which parent host this task has to connect.
 * @param parPort        Output parameter telling to which parent port this task has to connect.
 * @param parRank        Output parameter telling the rank of the parent to which this task has to connect.
 *
 * @return 0 if any task got errors; 1 otherwise 
 */
int SyncAttachments(
  int   rank,
  int   root,
  int   rdy_to_connect,
  char *sendbuf,
  int  *sendcnts,
  int  *displs,
  BE_data_t *BE_args)
{
  int i_went_ok = 0;

  /* Root task tells the rest if the file was read successfully */
  PMPI_Bcast(&rdy_to_connect, 1, MPI_INTEGER, root, MPI_COMM_WORLD);

  if (!rdy_to_connect)
  {
    if (rank == 0)
    {
      fprintf(stdout, "WARNING: There were problems initializing the on-line analysis!\n");
    }
    return 0;
  }

  char *ParentInfo = NULL;
  int   recvcnt    = 0;

  /* Distribute the number of chars that every process will receive */
  PMPI_Scatter(sendcnts, 1, MPI_INTEGER, &recvcnt, 1, MPI_INTEGER, root, MPI_COMM_WORLD);

  /* Distribute the connections information lines (1 per task) */
  ParentInfo = (char *)malloc(sizeof(char) * recvcnt);
  PMPI_Scatterv (sendbuf, sendcnts, displs, MPI_CHAR, ParentInfo, recvcnt, MPI_CHAR, root, MPI_COMM_WORLD);

  /* DEBUG 
  fprintf(stderr, "[BE %d] ParentInfo=%s\n", rank, ParentInfo); */

  if (recvcnt == 0)
  {
    fprintf(stderr, "ERROR: receiving connection information for task %d\n", rank);
  }
  else
  {
    /* Each task scans its line */
    int  matches = sscanf( ParentInfo, "%s %d %d %d", 
                           BE_args->parent_hostname, 
                           &(BE_args->parent_port), 
                           &(BE_args->parent_rank), 
                           &(BE_args->my_rank) );
    if ( matches != 4 ) 
    {
      fprintf(stderr, "ERROR: scanning connection information for task %d\n", rank);
    }
    else
    {
      i_went_ok = 1;
    }
  }

  /* Free pending connections info arrays */
  if (rank == 0)
  {
    free(sendcnts); free(displs); free(sendbuf);
  }
  free(ParentInfo);

  /* Check whether all tasks went ok */
  return SyncOk(i_went_ok);
}


/**
 * Perform a global barrier to synchronize all back-ends
 * @param this_be_ok Set to 0 if current back-end had errors; 0 otherwise.
 * @return 0 if any task got errors; 1 otherwise 
 */
int  SyncOk(int this_be_ok)
{
  int all_bes_ok = 0;

  PMPI_Allreduce(&this_be_ok, &all_bes_ok, 1, MPI_INTEGER, MPI_PROD, MPI_COMM_WORLD);

  return all_bes_ok;
}


/**
 * Perform a global barrier to synchronize all back-ends
 */
void SyncWaitAll()
{
  PMPI_Barrier(MPI_COMM_WORLD);
}

