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

#ifdef HAVE_STDIO_H
# ifdef HAVE_FOPEN64
#  define __USE_LARGEFILE64
# endif
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif

#if defined(PARALLEL_MERGE)
# include <mpi.h>
# include "mpi-aux.h"
#endif

#define NEW_DIMEMAS_TRACE

#define MAX_BUFFER_SIZE 1024

#include "events.h"
#include "dimemas_generator.h"
#include "object_tree.h"
#include "trace_to_prv.h"
#include "mpi_comunicadors.h"
#include "options.h"

/* ---------------------------------------------------- Constants -----------*/

/******************************************************************************
 **      Function name : Dimemas_WriteHeader
 **      
 **      Description : 
 ******************************************************************************/

#define PRVWRITECNTL(x) x

/******************************************************************************
 ***  Dimemas_WriteHeader
 ******************************************************************************/
int Dimemas_WriteHeader (unsigned num_appl, FILE *trf_fd,
	struct Pair_NodeCPU *info, char *outName)
{
	unsigned int threads, task, ptask;
#if defined (HAVE_MPI)  /* Sequential tracing does not use comunicators */
	TipusComunicador com;
	int i, final;
#endif

	UNREFERENCED_PARAMETER(info);
	                                /*  -- 18 0's --  */
	fprintf (trf_fd, "#DIMEMAS:%s:1,000000000000000000:",outName);

	/* DIMEMAS just supports 1 ptask */
	for (ptask = 0; ptask < num_appl; ptask++)
	{
		ptask_t *ptask_info = GET_PTASK_INFO(ptask+1);
		task_t *last_task_info = GET_TASK_INFO(ptask+1, ptask_info->ntasks);

		fprintf (trf_fd, "%d(", ptask_info->ntasks);

		for (task = 0; task < ptask_info->ntasks - 1; task++)
		{
			task_t *task_info = GET_TASK_INFO(ptask+1,task+1);
			fprintf (trf_fd, "%d,", task_info->threads);
		}
		threads = last_task_info->nthreads;
#if defined(HAVE_MPI)
		fprintf (trf_fd, "%d),%d", threads, numero_comunicadors());
#else
		fprintf (trf_fd, "%d),0", threads);
#endif
	}
	fprintf (trf_fd, "\n");


#if defined(HAVE_MPI)
	/* Write the communicator definition for every application */
	for (ptask = 0; ptask < num_appl; ptask++)
	{
		/* Write the communicators created manually by the application */
		final = (primer_comunicador (&com) < 0);
		while (!final)
		{
			/* Write this communicator */
			fprintf (trf_fd, "d:1:%d:%d", com.id, com.num_tasks);
			for (i = 0; i < com.num_tasks; i++)
			{
				fprintf (trf_fd, ":%d", com.tasks[i]);
			}
			fprintf (trf_fd, "\n");

			/* Get the next communicator */
			final = (seguent_comunicador (&com) < 0);
		}
	}
#endif
  return 0;
}

/******************************************************************************
 ***  Dimemas_WriteHeader
 ******************************************************************************/
int Dimemas_WriteOffsets (unsigned num_appl, FILE *trf_fd, char *outName,
	unsigned long long offset_position, unsigned int numfiles,
	unsigned long long *offsets)
{
	unsigned int ptask;
	int i;

	fflush (trf_fd);

	for (ptask = 0; ptask < num_appl; ptask++)
	{
		fprintf (trf_fd, "s");
		for (i = 0; i < numfiles; i++)
			fprintf (trf_fd,":%lld", offsets[i]);
	}
	fprintf (trf_fd, "\n");

	rewind (trf_fd);
	fprintf (trf_fd, "#DIMEMAS:%s:1,%018lld:",outName, offset_position);
	fflush (trf_fd);

	return 0;
}

/******************************************************************************
 **      Function name : Dimemas_NX_Generic_Send
 **      
 **      Description : 
 ******************************************************************************/

int Dimemas_NX_Generic_Send( FILE *fd,
                             int task, int thread,
                             int task_r, /* receiver */
                             int commid,
                             int size, UINT64 tag,
                             int synchronism )
{
#ifdef NEW_DIMEMAS_TRACE
# define NX_GENERIC_SEND_STRING "2:%d:%d:%d:%d:%lld:%d:%d\n"
#else
# define NX_GENERIC_SEND_STRING "\"NX send\" { %d, %d, %d, %d, %lld, %d, %d };;\n"
#endif

	return fprintf(fd,
		NX_GENERIC_SEND_STRING,
		task, thread, task_r, size, tag, commid, synchronism);
}

/******************************************************************************
 **      Function name : Dimemas_NX_Send
 **      
 **      Description : 
 ******************************************************************************/

int Dimemas_NX_Send( FILE *fd,
                     int task, int thread,
                     int task_r, /* receiver */
                     int commid,
                     int size, UINT64 tag )
{
  /* synchronism: NO immediate + NO rendezvous = 0 */
  
#ifdef NEW_DIMEMAS_TRACE
# define NX_SEND_STRING "2:%d:%d:%d:%d:%lld:%d:0\n"
#else
# define NX_SEND_STRING "\"NX send\" { %d, %d, %d, %d, %lld, %d, 0 };;\n"
#endif

	return fprintf(fd, NX_SEND_STRING, task, thread, task_r, size, tag, commid);
}



/******************************************************************************
 **      Function name : Dimemas_NX_Immediate_Send
 **      
 **      Description : 
 ******************************************************************************/

int Dimemas_NX_ImmediateSend( FILE *fd,
                              int task, int thread,
                              int task_r, /* receiver */
                              int commid,
                              int size, UINT64 tag)
{
  /* synchronism: immediate + NO rendezvous = 2 */
  
#ifdef NEW_DIMEMAS_TRACE
# define NX_ISEND_STRING "2:%d:%d:%d:%d:%lld:%d:2\n"
#else
# define NX_ISEND_STRING "\"NX send\" { %d, %d, %d, %d, %lld, %d, 2 };;\n"
#endif

	return fprintf (fd, NX_ISEND_STRING, task, thread, task_r, size, tag, commid);
}



/******************************************************************************
 **      Function name : Dimemas_NX_BlockingSend
 **      
 **      Description : 
 ******************************************************************************/

int Dimemas_NX_BlockingSend( FILE *fd,
                             int task, int thread,
                             int task_r, /* receiver */
                             int commid,
                             int size, UINT64 tag )
{
  /* synchronism: NO immediate + rendezvous = 1 */
  
#ifdef NEW_DIMEMAS_TRACE
# define NX_BSEND_STRING "2:%d:%d:%d:%d:%lld:%d:1\n"
#else
# define NX_BSEND_STRING "\"NX send\" { %d, %d, %d, %d, %lld, %d, 1 };;\n"
#endif

	return fprintf(fd, NX_BSEND_STRING, task, thread, task_r, size, tag, commid);
}

/******************************************************************************
 **      Function name : Dimemas_NX_Generic_Recv
 **      
 **      Description :   
 ******************************************************************************/

int Dimemas_NX_Generic_Recv( FILE *fd,
                             int task, int thread,
                             int task_s, /* source */
                             int commid,
                             int size, UINT64 tag,
                             int type )
{

#ifdef NEW_DIMEMAS_TRACE
# define NX_GENERIC_RECV_STRING "3:%d:%d:%d:%d:%lld:%d:%d\n"
#else
# define NX_GENERIC_RECV_STRING "\"NX recv\" { %d, %d, %d, %d, %lld, %d, %d };;\n"
#endif
 
	return fprintf(fd,
		NX_GENERIC_RECV_STRING,
		task, thread, task_s, size, tag, commid, type);
}

/******************************************************************************
 **      Function name : Dimemas_NX_Recv
 **      
 **      Description :   
 ******************************************************************************/

int Dimemas_NX_Recv( FILE *fd,
                     int task, int thread,
                     int task_s, /* source */
                     int commid,
                     int size, UINT64 tag )
{
  
#ifdef NEW_DIMEMAS_TRACE
# define NX_RECV_STRING "3:%d:%d:%d:%d:%lld:%d:0\n"
#else
# define NX_RECV_STRING "\"NX recv\" { %d, %d, %d, %d, %lld, %d, 0 };;\n"
#endif

	return fprintf (fd, NX_RECV_STRING, task, thread, task_s, size, tag, commid);
}

/******************************************************************************
 **      Function name : Dimemas_NX_Irecv
 **      
 **      Description :   
 ******************************************************************************/

int Dimemas_NX_Irecv( FILE *fd,
                      int task, int thread,
                      int task_s, /* source */
                      int commid,
                      int size, UINT64 tag )
{

#ifdef NEW_DIMEMAS_TRACE
# define NX_IRECV_STRING "3:%d:%d:%d:%d:%lld:%d:1\n"
#else
# define NX_IRECV_STRING "\"NX recv\" { %d, %d, %d, %d, %lld, %d, 1 };;\n"
#endif

	return fprintf (fd, NX_IRECV_STRING, task, thread, task_s, size, tag, commid);
}

/******************************************************************************
 **      Function name : Dimemas_NX_Wait
 **      
 **      Description :   
 ******************************************************************************/

int Dimemas_NX_Wait( FILE *fd,
                     int task, int thread,
                     int task_s, /* source */
                     int commid,
                     int size, UINT64 tag )
{
#ifdef NEW_DIMEMAS_TRACE
# define NX_WAIT_STRING "3:%d:%d:%d:%d:%lld:%d:2\n"
#else
# define NX_WAIT_STRING "\"NX recv\" { %d, %d, %d, %d, %lld, %d, 2 };;\n"
#endif

	return fprintf(fd, NX_WAIT_STRING, task, thread, task_s, size, tag, commid);
}

/******************************************************************************
 **      Function name : Dimemas_CPU_Burst
 **      
 **      Description : 
 ******************************************************************************/

int Dimemas_CPU_Burst( FILE *fd,
                       int task, int thread,
                       double burst_time )
{
#ifdef NEW_DIMEMAS_TRACE
# define CPU_BURST_STRING "1:%d:%d:%.6f\n"
#else
# define CPU_BURST_STRING "\"CPU burst\" { %d, %d, %.6f };;\n"
#endif

	return fprintf(fd, CPU_BURST_STRING, task, thread, burst_time);
}

#if defined(DEAD_CODE)
/******************************************************************************
 **      Function name : Dimemas_CPU_Burst
 **      
 **      Description : 
 ******************************************************************************/

int Dimemas_Communicator_Definition( FILE *fd,
                                     long long commid,
                                     int Ntasks,
                                     int *TaskList )
{
#if defined(DEAD_CODE)
  int ii;
  
#ifdef NEW_DIMEMAS_TRACE
  #define COMMUNICATOR_STRING "d:1:%lld:%d"
  
  if ( fprintf( fd, COMMUNICATOR_STRING, commid, Ntasks) < 0)
    return -1;
  
  for (ii= 0; ii< Ntasks; ii++)
  {
    if (fprintf( fd, ":%d", TaskList[ ii ] ) < 0)
      return -1;
  }
  
  if (fprintf(fd, "\n") < 0)
    return -1;
  
#else
  #define COMMUNICATOR_STRING "\"communicator definition\" { %lld, %d, [%d] { "
  
  if ( fprintf( fd, COMMUNICATOR_STRING, commid, Ntasks, Ntasks ) < 0)
    return -1;

  for (ii= 0; ii< Ntasks-1; ii++)
  {
    if (fprintf( fd, "%d,", TaskList[ ii ] ) < 0)
      return -1;
  }

  if (fprintf( fd, "%d }};;\n", TaskList[ Ntasks-1 ] ) < 0)
    return -1;
  
#endif

#endif /* DEAD_CODE */

  return 1;
}
#endif

/******************************************************************************
 **      Function name : Dimemas_User_Event
 **      
 **      Description : 
 ******************************************************************************/

int Dimemas_User_Event( FILE *fd,
                        int task, int thread,
                        UINT64 type, UINT64 value )
{
#ifdef NEW_DIMEMAS_TRACE
# define USER_EVENT_STRING "20:%d:%d:%lld:%lld\n"
#else
# define USER_EVENT_STRING "\"user event\" { %d, %d, %lld, %lld };;\n"
#endif

	return fprintf(fd, USER_EVENT_STRING, task, thread, type, value);
}

#if defined(DEAD_CODE)

/******************************************************************************
 **      Function name : Dimemas_User_EventType_Definition
 **      
 **      Description : 
 ******************************************************************************/

int Dimemas_User_EventType_Definition( FILE *fd,
                                       UINT64 type,
                                       char *Label,
                                       int color )
{
  #define USER_EVENT_TYPE_DEF_STRING "\"user event type definition\" { %lld, \"%s\", %d };;\n"
  ASSERT( Label != NULL );
  
  return fprintf(fd, USER_EVENT_TYPE_DEF_STRING, type, Label, color);
}

/******************************************************************************
 **      Function name : Dimemas_User_EventValue_Definition
 **      
 **      Description : 
 ******************************************************************************/

int Dimemas_User_EventValue_Definition( FILE *fd,
                                        UINT64 type,
                                        UINT64 value,
                                        char *Label )
{
  #define USER_EVENT_VALUE_DEF_STRING "\"user event value definition\" { %lld, %lld, \"%s\" };;\n"
  ASSERT( Label != NULL );
  
  return fprintf(fd, USER_EVENT_VALUE_DEF_STRING, type, value, Label);
}

/******************************************************************************
 **      Function name : Dimemas_Block_Definition
 **      
 **      Description : 
 ******************************************************************************/

int Dimemas_Block_Definition( FILE *fd,
                              UINT64 block,
                              char *Label )
{
  #define BLOCK_DEFINITION_STRING "\"block definition\" { %lld, \"%s\", \"\", 0, 0 };;\n"
  ASSERT( Label != NULL );
  
  return fprintf(fd, BLOCK_DEFINITION_STRING, block, Label);
}

/******************************************************************************
 **      Function name : Dimemas_Block_Begin
 **      
 **      Description : 
 ******************************************************************************/

int Dimemas_Block_Begin( FILE *fd,
                         int task, int thread,
                         UINT64 block )
{
  #define BLOCK_BEGIN_STRING "\"block begin\" { %d, %d, %lld };;\n"
  return fprintf(fd, BLOCK_BEGIN_STRING, task, thread, block);
}

/******************************************************************************
 **      Function name : Dimemas_Block_End
 **      
 **      Description : 
 ******************************************************************************/

int Dimemas_Block_End( FILE *fd,
                       int task, int thread,
                       UINT64 block )
{
  #define BLOCK_END_STRING "\"block end\" { %d, %d, %lld };;\n"
  return fprintf(fd, BLOCK_END_STRING, task, thread, block );
}

#endif /* DEAD_CODE */

/******************************************************************************
 **      Function name : Dimemas_Global_OP
 **      
 **      Description : 
 ******************************************************************************/

int Dimemas_Global_OP( FILE *fd,
                       int task, int thread,
                       int opid, int commid,
                       int root_rank, int root_thd,
                       UINT64 sendsize, UINT64 recvsize )
{
#ifdef NEW_DIMEMAS_TRACE
# define GLOBAL_OP_STRING "10:%d:%d:%d:%d:%d:%d:%lld:%lld\n"
#else
# define GLOBAL_OP_STRING "\"global OP\" { %d, %d, %d, %d, %d, %d, %lld, %lld };;\n"
#endif

	Dimemas_User_Event (fd, task, thread, MPI_GLOBAL_OP_SENDSIZE, sendsize);
	Dimemas_User_Event (fd, task, thread, MPI_GLOBAL_OP_RECVSIZE, recvsize);
	Dimemas_User_Event (fd, task, thread, MPI_GLOBAL_OP_COMM, commid);
	if (root_rank == task && root_thd == thread)
		Dimemas_User_Event (fd, task, thread, MPI_GLOBAL_OP_ROOT, 1);

	return fprintf (fd, GLOBAL_OP_STRING, task, thread, opid, commid,
		root_rank, root_thd, sendsize, recvsize );
}

#if defined(DEAD_CODE)
int Dimemas_NX_One_Sided( FILE * fd,
                          int task, int thread,
                          int one_sided_opid, 
                          int handle, int tgt_thid, 
                          int msg_size) 
{
  #define NX_ONE_SIDED_STRING "\"1sided OP\" { %d, %d, %d, %d, %d, %d };;\n"
  return fprintf( fd, NX_ONE_SIDED_STRING, 
                  task, thread, one_sided_opid, 
                  handle, tgt_thid, msg_size);
}
#endif

