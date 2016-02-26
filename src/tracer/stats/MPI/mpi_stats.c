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

#include "mpi_stats.h"
#include "utils.h"

#ifndef HAVE_STDIO_H
# include <stdio.h>
#endif

#include <mpi.h>

mpi_stats_t *global_mpi_stats = NULL;

mpi_stats_t * mpi_stats_init(int num_tasks)
{
    mpi_stats_t *mpi_stats = NULL;

    mpi_stats = (mpi_stats_t *)malloc(sizeof(mpi_stats_t));
    if (mpi_stats == NULL)
    { 
        fprintf (stderr, PACKAGE_NAME": Error! Unable to get memory for MPI Stats");
        exit(-1);
    }
    mpi_stats->ntasks = num_tasks;

    mpi_stats->P2P_Partner_In = (int *) malloc (mpi_stats->ntasks * sizeof(int));
    if (mpi_stats->P2P_Partner_In == NULL)
    {
        fprintf (stderr, PACKAGE_NAME": Error! Unable to get memory for MPI Stats");
        exit(-1);
    }
    mpi_stats->P2P_Partner_Out = (int *) malloc (mpi_stats->ntasks * sizeof(int));
    if (mpi_stats->P2P_Partner_Out == NULL)
    {
        fprintf (stderr, PACKAGE_NAME": Error! Unable to get memory for MPI Stats");
        exit(-1);
    }

    mpi_stats_reset(mpi_stats);

    return mpi_stats;
}

void mpi_stats_reset(mpi_stats_t * mpi_stats)
{
   int i;

   if (mpi_stats != NULL)
   {
      mpi_stats->P2P_Bytes_Sent = 0;
      mpi_stats->P2P_Bytes_Recv = 0;
      mpi_stats->COLLECTIVE_Bytes_Sent = 0;
      mpi_stats->COLLECTIVE_Bytes_Recv = 0;
      mpi_stats->P2P_Communications = 0;
      mpi_stats->COLLECTIVE_Communications = 0;
      mpi_stats->MPI_Others_count = 0;
      mpi_stats->Elapsed_Time_In_MPI = 0;

      mpi_stats->P2P_Communications_In = 0;
      mpi_stats->P2P_Communications_Out = 0;

      mpi_stats->Elapsed_Time_In_P2P_MPI = 0;
      mpi_stats->Elapsed_Time_In_COLLECTIVE_MPI = 0;

      for (i = 0; i < mpi_stats->ntasks; i++)
      {
         mpi_stats->P2P_Partner_In[i] = 0;
         mpi_stats->P2P_Partner_Out[i] = 0;
      }
   }
}

void mpi_stats_free(mpi_stats_t * mpi_stats)
{
  xfree(mpi_stats->P2P_Partner_In);
  xfree(mpi_stats->P2P_Partner_Out);
  xfree(mpi_stats);
}

void mpi_stats_sum(mpi_stats_t * base, mpi_stats_t * extra)
{
  int i = 0;

  if ((base != NULL) && (extra != NULL))
  {
    base->P2P_Bytes_Sent                 += extra->P2P_Bytes_Sent;
    base->P2P_Bytes_Recv                 += extra->P2P_Bytes_Recv;
    base->COLLECTIVE_Bytes_Sent          += extra->COLLECTIVE_Bytes_Sent;
    base->COLLECTIVE_Bytes_Recv          += extra->COLLECTIVE_Bytes_Recv;
    base->P2P_Communications             += extra->P2P_Communications;
    base->COLLECTIVE_Communications      += extra->COLLECTIVE_Communications;
    base->MPI_Others_count               += extra->MPI_Others_count;
    base->Elapsed_Time_In_MPI            += extra->Elapsed_Time_In_MPI;
    base->P2P_Communications_In          += extra->P2P_Communications_In;
    base->P2P_Communications_Out         += extra->P2P_Communications_Out;
    base->Elapsed_Time_In_P2P_MPI        += extra->Elapsed_Time_In_P2P_MPI;
    base->Elapsed_Time_In_COLLECTIVE_MPI += extra->Elapsed_Time_In_COLLECTIVE_MPI;
    for (i = 0; i < base->ntasks; i++)
    {
      base->P2P_Partner_In[i]  += extra->P2P_Partner_In[i];
      base->P2P_Partner_Out[i] += extra->P2P_Partner_Out[i];
    }
  }
}

void updateStats_P2P(mpi_stats_t * mpi_stats, int partner, int inputSize, int outputSize)
{
   /* Weird cases: MPI_Sendrecv_Fortran_Wrapper */
   if (mpi_stats != NULL)
   {
      mpi_stats->P2P_Communications ++;
      if (inputSize)
      {    
          mpi_stats->P2P_Bytes_Recv += inputSize;
          mpi_stats->P2P_Communications_In ++;
          if (partner != MPI_PROC_NULL && partner != MPI_ANY_SOURCE && 
              partner != MPI_UNDEFINED)
          {
              if (partner < mpi_stats->ntasks)
                  mpi_stats->P2P_Partner_In[partner] ++;
              else
                  fprintf(stderr, "[DEBUG] OUT_OF_RANGE partner=%d/%d\n",
                    partner, mpi_stats->ntasks);
          }
      }    
      if (outputSize)
      {    
          mpi_stats->P2P_Bytes_Sent += outputSize;
          mpi_stats->P2P_Communications_Out ++;
          if (partner != MPI_PROC_NULL && partner != MPI_ANY_SOURCE && 
              partner != MPI_UNDEFINED)
          {
              if (partner < mpi_stats->ntasks)
                  mpi_stats->P2P_Partner_Out[partner] ++;
              else
                  fprintf(stderr, "[DEBUG] OUT_OF_RANGE partner=%d/%d\n",
                    partner, mpi_stats->ntasks);
          }
      }    
   }
}

void updateStats_COLLECTIVE(mpi_stats_t * mpi_stats, int inputSize, int outputSize)
{
    mpi_stats->COLLECTIVE_Communications ++;
    if (inputSize)
    {
        mpi_stats->COLLECTIVE_Bytes_Recv += inputSize;
    }
    if (outputSize)
    {
        mpi_stats->COLLECTIVE_Bytes_Sent += outputSize;
    }
}

void updateStats_OTHER(mpi_stats_t * mpi_stats)
{
    mpi_stats->MPI_Others_count++;
}

int mpi_stats_get_num_partners(mpi_stats_t * mpi_stats, int * partners_vector)
{
    int i, num_partners = 0;
    for (i = 0; i < mpi_stats->ntasks; i++)
    {
        if (partners_vector[i] > 0) num_partners++;
    }
    return num_partners;
}

void mpi_stats_update_elapsed_time(mpi_stats_t * mpi_stats, unsigned EvtType, unsigned long long elapsedTime)
{
    mpi_stats->Elapsed_Time_In_MPI += elapsedTime;

    if(isMPI_P2P(EvtType))
        mpi_stats->Elapsed_Time_In_P2P_MPI += elapsedTime;
    else if(isMPI_Global(EvtType))
        mpi_stats->Elapsed_Time_In_COLLECTIVE_MPI += elapsedTime;
}
