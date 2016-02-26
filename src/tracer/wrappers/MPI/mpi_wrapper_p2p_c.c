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
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif
#ifdef HAVE_SYS_STAT_H
# include <sys/stat.h>
#endif
#ifdef HAVE_SYS_FILE_H
# include <sys/file.h>
#endif
#ifdef HAVE_FCNTL_H
# include <fcntl.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif

#include <mpi.h>
#include "utils.h"
#include "utils_mpi.h"
#include "mpi_wrapper.h"
#include "mpi_interface_coll_helper.h"
#include "wrapper.h"
#include "clock.h"
#include "signals.h"
#include "misc_wrapper.h"
#include "mpi_interface.h"
#include "mode.h"
#include "threadinfo.h"

#include "hash_table.h"

#if defined(C_SYMBOLS) && defined(FORTRAN_SYMBOLS)
# define COMBINED_SYMBOLS
#endif

#if defined(HAVE_MRNET)
# include "mrnet_be.h"
#endif

#include "misc_wrapper.h"

#define MPI_CHECK(mpi_error, routine) \
	if (mpi_error != MPI_SUCCESS) \
	{ \
		fprintf (stderr, "Error in MPI call %s (file %s, line %d, routine %s) returned %d\n", \
			#routine, __FILE__, __LINE__, __func__, mpi_error); \
		fflush (stderr); \
		exit (1); \
	}

#if defined(C_SYMBOLS)

#define MAX_WAIT_REQUESTS 16384

/******************************************************************************
 ***  MPI_Bsend_C_Wrapper
 ******************************************************************************/

int MPI_Bsend_C_Wrapper (void *buf, int count, MPI_Datatype datatype, int dest,
                         int tag, MPI_Comm comm)
{
	int size, receiver, ret;

	if (count != 0)
	{
		ret = PMPI_Type_size (datatype, &size);
		MPI_CHECK(ret, PMPI_Type_size);
	}
	else
		size = 0;

	if ((ret = get_rank_obj_C (comm, dest, &receiver, RANK_OBJ_SEND)) != MPI_SUCCESS)
		return ret;

	size *= count;

	/*
	*   event : BSEND_EV                      value : EVT_BEGIN
	*   target : receiver                     size  : send message size
	*   tag : message tag
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_BSEND_EV, EVT_BEGIN, receiver, size, tag, comm,
	  EMPTY);

	ret = PMPI_Bsend (buf, count, datatype, dest, tag, comm);

	/*
	*   event : BSEND_EV                      value : EVT_END
	*   target : receiver                     size  : send message size
	*   tag : message tag
	*/
	TRACE_MPIEVENT (TIME, MPI_BSEND_EV, EVT_END, receiver, size, tag, comm, EMPTY);

	/* MPI Stats */
    updateStats_P2P(global_mpi_stats, receiver, 0, size);

	return ret;
}


/******************************************************************************
 ***  MPI_Ssend_C_Wrapper
 ******************************************************************************/

int MPI_Ssend_C_Wrapper (void *buf, int count, MPI_Datatype datatype, int dest,
                         int tag, MPI_Comm comm)
{
	int size, receiver, ret;

	if (count != 0)
	{
		ret = PMPI_Type_size (datatype, &size);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	if ((ret = get_rank_obj_C (comm, dest, &receiver, RANK_OBJ_SEND)) != MPI_SUCCESS)
		return ret;

	size *= count;

	/*
	*   event : SSEND_EV                      value : EVT_BEGIN
	*   target : receiver                     size  : send message size
	*   tag : message tag
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_SSEND_EV, EVT_BEGIN, receiver, size, tag, comm,
	  EMPTY);

	ret = PMPI_Ssend (buf, count, datatype, dest, tag, comm);

	/*
	*   event : SSEND_EV                      value : EVT_END
	*   target : receiver                     size  : send message size
	*   tag : message tag
	*/
	TRACE_MPIEVENT (TIME, MPI_SSEND_EV, EVT_END, receiver, size, tag, comm,
	  EMPTY);

	/* MPI Stats */
    updateStats_P2P(global_mpi_stats, receiver, 0, size);

	return ret;
}



/******************************************************************************
 ***  MPI_Rsend_C_Wrapper
 ******************************************************************************/

int MPI_Rsend_C_Wrapper (void *buf, int count, MPI_Datatype datatype, int dest,
                         int tag, MPI_Comm comm)
{
	int size, receiver, ret;

	if (count != 0)
	{
		ret = PMPI_Type_size (datatype, &size);
		MPI_CHECK(ret, PMPI_Type_size);
	}
	else
		size = 0;

	if ((ret = get_rank_obj_C (comm, dest, &receiver, RANK_OBJ_SEND)) != MPI_SUCCESS)
		return ret;

	size *= count;

	/*
	*   event : RSEND_EV                      value : EVT_BEGIN
	*   target : receiver                     size  : send message size
	*   tag : message tag
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_RSEND_EV, EVT_BEGIN, receiver, size, tag, comm,
	  EMPTY);

	ret = PMPI_Rsend (buf, count, datatype, dest, tag, comm);

	/*
	*   event : RSEND_EV                      value : EVT_END
	*   target : receiver                     size  : send message size
	*   tag : message tag
	*/
	TRACE_MPIEVENT (TIME, MPI_RSEND_EV, EVT_END, receiver, size, tag, comm, EMPTY);

	/* MPI Stats */
    updateStats_P2P(global_mpi_stats, receiver, 0, size);

	return ret;
}



/******************************************************************************
 ***  MPI_Send_C_Wrapper
 ******************************************************************************/

int MPI_Send_C_Wrapper (void *buf, int count, MPI_Datatype datatype, int dest,
                        int tag, MPI_Comm comm)
{
	int size, receiver, ret;

	if (count != 0)
	{
		ret = PMPI_Type_size (datatype, &size);
		MPI_CHECK(ret, PMPI_Type_size);
	}
	else
		size = 0;

	if ((ret = get_rank_obj_C (comm, dest, &receiver, RANK_OBJ_SEND)) != MPI_SUCCESS)
		return ret;

	size *= count;

	/*
	*   event : SEND_EV                       value : EVT_BEGIN
	*   target : receiver                     size  : send message size
	*   tag : message tag
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_SEND_EV, EVT_BEGIN, receiver, size, tag, comm, EMPTY);

	ret = PMPI_Send (buf, count, datatype, dest, tag, comm);
  
	/*
	*   event : SEND_EV                       value : EVT_END
	*   target : receiver                     size  : send message size
	*   tag : message tag
	*/
	TRACE_MPIEVENT (TIME, MPI_SEND_EV, EVT_END, receiver, size, tag, comm, EMPTY);

	/* MPI Stats */
    updateStats_P2P(global_mpi_stats, receiver, 0, size);

	return ret;
}



/******************************************************************************
 ***  MPI_Ibsend_C_Wrapper
 ******************************************************************************/

int MPI_Ibsend_C_Wrapper (void *buf, int count, MPI_Datatype datatype, int dest,
                          int tag, MPI_Comm comm, MPI_Request *request)
{
	int size, receiver, ret;

	if (count != 0)
	{
		ret = PMPI_Type_size (datatype, &size);
		MPI_CHECK(ret, PMPI_Type_size);
	}
	else
		size = 0;

	if ((ret = get_rank_obj_C (comm, dest, &receiver, RANK_OBJ_SEND)) != MPI_SUCCESS)
		return ret;

	size *= count;

	/*
	*   event : IBSEND_EV                     value : EVT_BEGIN
	*   target : ---                          size  : ---
	*   tag : ---                             commid: ---
	*   aux : ---
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IBSEND_EV, EVT_BEGIN, receiver, size, tag, comm, EMPTY);

	ret = PMPI_Ibsend (buf, count, datatype, dest, tag, comm, request);

	/*
	*   event : IBSEND_EV                     value : EVT_END
	*   target : ---                          size  : ---
	*   tag : ---                             commid: ---
	*   aux : ---
	*/
	TRACE_MPIEVENT (TIME, MPI_IBSEND_EV, EVT_END, receiver, size, tag, comm, EMPTY);

	/* MPI Stats */
    updateStats_P2P(global_mpi_stats, receiver, 0, size);

	return ret;
}



/******************************************************************************
 ***  MPI_Isend_C_Wrapper
 ******************************************************************************/

int MPI_Isend_C_Wrapper (void *buf, int count, MPI_Datatype datatype, int dest,
                         int tag, MPI_Comm comm, MPI_Request *request)
{
	int size, receiver, ret;

	if (count != 0)
	{
		ret = PMPI_Type_size (datatype, &size);
		MPI_CHECK(ret, PMPI_Type_size);
	}
	else
		size = 0;

	if ((ret = get_rank_obj_C (comm, dest, &receiver, RANK_OBJ_SEND)) != MPI_SUCCESS)
		return ret;

	size *= count;

	/*
	*   event : ISEND_EV                      value : EVT_BEGIN
	*   target : ---                          size  : ---
	*   tag : ---                             commid: ---
	*   aux : ---
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_ISEND_EV, EVT_BEGIN, receiver, size, tag, comm, EMPTY);

	ret = PMPI_Isend (buf, count, datatype, dest, tag, comm, request);

	/*
	*   event : ISEND_EV                      value : EVT_END
	*   target : ---                          size  : ---
	*   tag : ---                             commid: ---
	*   aux : ---
	*/
	TRACE_MPIEVENT (TIME, MPI_ISEND_EV, EVT_END, receiver, size, tag, comm, EMPTY);

	/* MPI Stats */
    updateStats_P2P(global_mpi_stats, receiver, 0, size);

	return ret;
}



/******************************************************************************
 ***  MPI_Issend_C_Wrapper
 ******************************************************************************/

int MPI_Issend_C_Wrapper (void *buf, int count, MPI_Datatype datatype, int dest,
                          int tag, MPI_Comm comm, MPI_Request *request)
{
	int size, receiver, ret;

	if (count != 0)
	{
		ret = PMPI_Type_size (datatype, &size);
		MPI_CHECK(ret, PMPI_Type_size);
	}
	else
		size = 0;

	if ((ret = get_rank_obj_C (comm, dest, &receiver, RANK_OBJ_SEND)) != MPI_SUCCESS)
		return ret;

	size *= count;

	/*
	*   event : ISSEND_EV                     value : EVT_BEGIN
	*   target : ---                          size  : ---
	*   tag : ---                             commid: ---
	*   aux : ---
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_ISSEND_EV, EVT_BEGIN, receiver, size, tag, comm, EMPTY);

	ret = PMPI_Issend (buf, count, datatype, dest, tag, comm, request);

	/*
	*   event : ISSEND_EV                     value : EVT_END
	*   target : ---                          size  : ---
	*   tag : ---                             commid: ---
	*   aux : ---
	*/
	TRACE_MPIEVENT (TIME, MPI_ISSEND_EV, EVT_END, receiver, size, tag, comm, EMPTY);

	/* MPI Stats */
    updateStats_P2P(global_mpi_stats, receiver, 0, size);

	return ret;
}



/******************************************************************************
 ***  MPI_Irsend_C_Wrapper
 ******************************************************************************/

int MPI_Irsend_C_Wrapper (void *buf, int count, MPI_Datatype datatype, int dest,
                          int tag, MPI_Comm comm, MPI_Request *request)
{
	int size, receiver, ret;

	if (count != 0)
	{
		ret = PMPI_Type_size (datatype, &size);
		MPI_CHECK(ret, PMPI_Type_size);
	}
	else
		size = 0;

	if ((ret = get_rank_obj_C (comm, dest, &receiver, RANK_OBJ_SEND)) != MPI_SUCCESS)
		return ret;

	size *= count;

	/*
	*   event : IRSEND_EV                     value : EVT_BEGIN
	*   target : ---                          size  : ---
	*   tag : ---                             commid: ---
	*   aux : ---
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IRSEND_EV, EVT_BEGIN, receiver, size, tag, comm, EMPTY);

	ret = PMPI_Irsend (buf, count, datatype, dest, tag, comm, request);

	/*
	*   event : IRSEND_EV                     value : EVT_END
	*   target : ---                          size  : ---
	*   tag : ---                             commid: ---
	*   aux : ---
	*/
	TRACE_MPIEVENT (TIME, MPI_IRSEND_EV, EVT_END, receiver, size, tag, comm, EMPTY);

	/* MPI Stats */
    updateStats_P2P(global_mpi_stats, receiver, 0, size);

	return ret;
}



/******************************************************************************
 ***  MPI_Recv_C_Wrapper
 ******************************************************************************/

int MPI_Recv_C_Wrapper (void *buf, int count, MPI_Datatype datatype, int source,
                        int tag, MPI_Comm comm, MPI_Status *status)
{
	MPI_Status my_status, *ptr_status;
	int size, src_world, sender_src, ret, recved_count, sended_tag, ierror;

	if (count != 0)
	{
		ret = PMPI_Type_size (datatype, &size);
		MPI_CHECK(ret, PMPI_Type_size);
	}
	else
		size = 0;

	if ((ret = get_rank_obj_C (comm, source, &src_world, RANK_OBJ_RECV)) != MPI_SUCCESS)
		return ret;

	/*
	*   event : RECV_EV                      value : EVT_BEGIN    
	*   target : MPI_ANY_SOURCE or sender    size  : receive buffer size    
	*   tag : message tag or MPI_ANY_TAG     commid: Communicator identifier
	*   aux: ---
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_RECV_EV, EVT_BEGIN, src_world, count * size, tag,
	  comm, EMPTY);

	ptr_status = (MPI_STATUS_IGNORE == status)?&my_status:status; 
 
	ierror = PMPI_Recv (buf, count, datatype, source, tag, comm, ptr_status);

	ret = PMPI_Get_count (ptr_status, datatype, &recved_count);
	MPI_CHECK(ret, PMPI_Get_count);

	if (recved_count != MPI_UNDEFINED)
		size *= recved_count;
	else
		size = 0;

	if (source == MPI_ANY_SOURCE)
		sender_src = ptr_status->MPI_SOURCE;
	else
		sender_src = source;

	if (tag == MPI_ANY_TAG)
		sended_tag = ptr_status->MPI_TAG;
	else
		sended_tag = tag;

	if ((ret = get_rank_obj_C (comm, sender_src, &src_world, RANK_OBJ_RECV)) != MPI_SUCCESS)
		return ret;

	/*
	*   event : RECV_EV                      value : EVT_END
	*   target : sender                      size  : received message size    
	*   tag : message tag
	*/
	TRACE_MPIEVENT (TIME, MPI_RECV_EV, EVT_END, src_world, size, sended_tag,
	  comm, EMPTY);

	/* MPI Stats */
    updateStats_P2P(global_mpi_stats, source, size, 0);

	return ierror;
}



/******************************************************************************
 ***  MPI_Irecv_C_Wrapper
 ******************************************************************************/

int MPI_Irecv_C_Wrapper (void *buf, int count, MPI_Datatype datatype,
	int source, int tag, MPI_Comm comm, MPI_Request *request)
{
	hash_data_t hash_req;
	int inter, ret, ierror, size, src_world;

	if (count != 0)
	{
		ret = PMPI_Type_size (datatype, &size);
		MPI_CHECK(ret, PMPI_Type_size);
	}
	else
		size = 0;

	if ((ret = get_rank_obj_C (comm, source, &src_world, RANK_OBJ_RECV)) != MPI_SUCCESS)
		return ret;

	/*
	*   event : IRECV_EV                     value : EVT_BEGIN
	*   target : ---                         size  : ---
	*   tag : ---
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IRECV_EV, EVT_BEGIN, src_world, count * size, tag,
	  comm, EMPTY);

	ierror = PMPI_Irecv (buf, count, datatype, source, tag, comm, request);

	hash_req.key = *request;
	hash_req.commid = comm;
	hash_req.partner = source;
	hash_req.tag = tag;
	hash_req.size = count * size;

	if (comm == MPI_COMM_WORLD)
	{
		hash_req.group = MPI_GROUP_NULL;
	}
	else
	{
		ret = PMPI_Comm_test_inter (comm, &inter);
		MPI_CHECK(ret,PMPI_Comm_test_inter);

		if (inter)
		{
			ret = PMPI_Comm_remote_group (comm, &hash_req.group);
			MPI_CHECK(ret,PMPI_Comm_remote_group);
		}
		else
		{
			ret = PMPI_Comm_group (comm, &hash_req.group);
			MPI_CHECK(ret,PMPI_Comm_group);
		}
	}

	hash_add (&requests, &hash_req);

	/*
	*   event : IRECV_EV                     value : EVT_END
	*   target : partner                     size  : ---
	*   tag : ---                            comm  : communicator
	*   aux: request
	*/
	TRACE_MPIEVENT (TIME, MPI_IRECV_EV, EVT_END, src_world, count * size, tag, comm,
	  hash_req.key);

	return ierror;
}

/******************************************************************************
 ***  MPI_Probe_C_Wrapper
 ******************************************************************************/

int MPI_Probe_C_Wrapper (int source, int tag, MPI_Comm comm, MPI_Status *status)
{
  int ierror;

  /*
   *   event : PROBE_EV                     value : EVT_BEGIN
   *   target : ---                         size  : ---
   *   tag : ---
   */
  TRACE_MPIEVENT (LAST_READ_TIME, MPI_PROBE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, comm,
                  EMPTY);

  ierror = PMPI_Probe (source, tag, comm, status);

  /*
   *   event : PROBE_EV                     value : EVT_END
   *   target : ---                         size  : ---
   *   tag : ---
   */
  TRACE_MPIEVENT (TIME, MPI_PROBE_EV, EVT_END, EMPTY, EMPTY, EMPTY, comm, EMPTY);

    updateStats_OTHER(global_mpi_stats);

  return ierror;
}

/******************************************************************************
 ***  MPI_Iprobe_C_Wrapper
 ******************************************************************************/

int Bursts_MPI_Iprobe_C_Wrapper (int source, int tag, MPI_Comm comm, int * flag, MPI_Status *status)
{
	int ierror;

	/*
	*   event : IPROBE_EV                     value : EVT_BEGIN
	*   target : ---                          size  : ---
	*   tag : ---
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IPROBE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, comm, EMPTY);

	ierror = PMPI_Iprobe (source, tag, comm, flag, status);

	/*
	*   event : IPROBE_EV                    value : EVT_END
	*   target : ---                         size  : ---
	*   tag : ---
	*/
	TRACE_MPIEVENT (TIME, MPI_IPROBE_EV, EVT_END, EMPTY, EMPTY, EMPTY, comm, EMPTY);

	return ierror;
}

int Normal_MPI_Iprobe_C_Wrapper (int source, int tag, MPI_Comm comm, int *flag,
                          MPI_Status *status)
{
	static int IProbe_C_Software_Counter = 0;
	iotimer_t begin_time, end_time;
	static iotimer_t elapsed_time_outside_iprobes_C = 0, last_iprobe_C_exit_time = 0; 
	int ierror;

	begin_time = LAST_READ_TIME;

	if (IProbe_C_Software_Counter == 0)
	{
		/* Primer Iprobe */
		elapsed_time_outside_iprobes_C = 0;
	}
	else
	{
		elapsed_time_outside_iprobes_C += (begin_time - last_iprobe_C_exit_time);
	}

	ierror = PMPI_Iprobe (source, tag, comm, flag, status);
	end_time = TIME;
	last_iprobe_C_exit_time = end_time;

	if (tracejant_mpi)
	{
		if (*flag)
		{
			if (IProbe_C_Software_Counter != 0)
			{
				TRACE_EVENT (begin_time, MPI_TIME_OUTSIDE_IPROBES_EV, elapsed_time_outside_iprobes_C);
				TRACE_EVENT (begin_time, MPI_IPROBE_COUNTER_EV, IProbe_C_Software_Counter);
			}

			TRACE_MPIEVENT (begin_time, MPI_IPROBE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, comm, EMPTY);
    
			TRACE_MPIEVENT (end_time, MPI_IPROBE_EV, EVT_END, EMPTY, EMPTY, EMPTY, comm, EMPTY);
			IProbe_C_Software_Counter = 0;
		} 
		else
		{
			if (IProbe_C_Software_Counter == 0)
			{
				/* El primer iprobe que falla */
				TRACE_EVENTANDCOUNTERS (begin_time, MPI_IPROBE_COUNTER_EV, 0, TRUE);
			}
			IProbe_C_Software_Counter ++;
		}
	}
	return ierror;
}

int MPI_Iprobe_C_Wrapper (int source, int tag, MPI_Comm comm, int * flag, MPI_Status *status)
{
   int ret;

   if (CURRENT_TRACE_MODE(THREADID) == TRACE_MODE_BURSTS)
   { 
      ret = Bursts_MPI_Iprobe_C_Wrapper (source, tag, comm, flag, status);
   } 
   else
   {
      ret = Normal_MPI_Iprobe_C_Wrapper (source, tag, comm, flag, status);
   }
   return ret;
}

/******************************************************************************
 ***  MPI_Test_C_Wrapper
 ******************************************************************************/

int Bursts_MPI_Test_C_Wrapper (MPI_Request *request, int *flag, MPI_Status *status)
{
	MPI_Request req;
	hash_data_t *hash_req;
	int src_world, size, tag, ret, ierror;
	iotimer_t temps_final;

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_TEST_EV, EVT_BEGIN, *request, EMPTY, EMPTY, EMPTY,
                  EMPTY);

	req = *request;

	ierror = PMPI_Test (request, flag, status);

	temps_final = TIME;

	if (ierror == MPI_SUCCESS && *flag && ((hash_req = hash_search (&requests, req)) != NULL))
	{
		if ((ret = get_Irank_obj_C (hash_req, &src_world, &size, &tag, status)) != MPI_SUCCESS)
			return ret;
		if (hash_req->group != MPI_GROUP_NULL)
		{
			ret = PMPI_Group_free (&hash_req->group);
			MPI_CHECK(ret, PMPI_Group_free);
		}

        /* MPI Stats, get_Irank_obj_C above returns size (number of bytes received) */
        updateStats_P2P(global_mpi_stats, src_world, size, 0);

		TRACE_MPIEVENT_NOHWC (temps_final, MPI_IRECVED_EV, EMPTY, src_world, size, hash_req->tag, hash_req->commid, req);
		hash_remove (&requests, req);
	}
	TRACE_MPIEVENT (temps_final, MPI_TEST_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	return ierror;
}

int Normal_MPI_Test_C_Wrapper (MPI_Request *request, int *flag, MPI_Status *status)
{
	MPI_Request req;
	hash_data_t *hash_req;
	int src_world, size, tag, ret, ierror;
	iotimer_t temps_inicial, temps_final;
	static int Test_C_Software_Counter = 0;

	temps_inicial = LAST_READ_TIME;

	req = *request;

	ierror = PMPI_Test (request, flag, status);

	temps_final = TIME;

	if (ierror == MPI_SUCCESS && *flag && ((hash_req = hash_search (&requests, req)) != NULL))
	{
		if (Test_C_Software_Counter != 0)
			TRACE_EVENT(temps_inicial, MPI_TEST_COUNTER_EV, Test_C_Software_Counter);

		TRACE_MPIEVENT (temps_inicial, MPI_TEST_EV, EVT_BEGIN, hash_req->key, EMPTY, EMPTY, EMPTY, EMPTY);
		Test_C_Software_Counter = 0;

		if ((ret = get_Irank_obj_C (hash_req, &src_world, &size, &tag, status)) != MPI_SUCCESS)
			return ret;

		if (hash_req->group != MPI_GROUP_NULL)
		{
			ret = PMPI_Group_free (&hash_req->group);
			MPI_CHECK(ret, PMPI_Group_free);
		}

		TRACE_MPIEVENT_NOHWC (temps_final, MPI_IRECVED_EV, EMPTY, src_world, size, hash_req->tag, hash_req->commid, hash_req->key);
		hash_remove (&requests, req);
		TRACE_MPIEVENT (temps_final, MPI_TEST_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
	}
	else
	{
		if (Test_C_Software_Counter == 0)
			TRACE_EVENTANDCOUNTERS (temps_inicial, MPI_TEST_COUNTER_EV, 0, TRUE);
		Test_C_Software_Counter ++;
	}
	return ierror;
}

int MPI_Test_C_Wrapper (MPI_Request *request, int *flag, MPI_Status *status)
{
	MPI_Status my_status, *ptr_status;
	int ret;

	if (status == MPI_STATUS_IGNORE)
		ptr_status = &my_status;
	else
		ptr_status = status;

	if (CURRENT_TRACE_MODE(THREADID) == TRACE_MODE_BURSTS)
		ret = Bursts_MPI_Test_C_Wrapper (request, flag, ptr_status);
	else
		ret = Normal_MPI_Test_C_Wrapper (request, flag, ptr_status);

	return ret;
}

/******************************************************************************
 ***  MPI_Testall_C_Wrapper
 ******************************************************************************/

int MPI_Testall_C_Wrapper (int count, MPI_Request *array_of_requests, int *flag,
	MPI_Status *array_of_statuses)
{
	MPI_Status my_statuses[MAX_WAIT_REQUESTS], *ptr_array_of_statuses;
	MPI_Request save_reqs[MAX_WAIT_REQUESTS];
	hash_data_t *hash_req;
	int src_world, size, tag, ret, ireq, ierror;
	iotimer_t temps_final, temps_inicial;
#if defined(DEBUG_MPITRACE)
	int index;
#endif
	static int Test_C_Software_Counter = 0;

	temps_inicial = LAST_READ_TIME;

	if (count > MAX_WAIT_REQUESTS)
		fprintf (stderr, PACKAGE_NAME": PANIC! too many requests in mpi_testall\n");
	memcpy (save_reqs, array_of_requests, count * sizeof (MPI_Request));

#if defined(DEBUG_MPITRACE)
	fprintf (stderr,  PACKAGE_NAME" %d: TESTALL summary\n", TASKID);
	for (index = 0; index < count; index++)
# if SIZEOF_LONG == 8
		fprintf (stderr, "%d: position %d -> request %lu\n", TASKID, index, (UINT64) array_of_requests[index]);
# elif SIZEOF_LONG == 4
		fprintf (stderr, "%d: position %d -> request %llu\n", TASKID, index, (UINT64) array_of_requests[index]);
# endif
#endif

	ptr_array_of_statuses = (MPI_STATUSES_IGNORE == array_of_statuses)?my_statuses:array_of_statuses;

	ierror = PMPI_Testall (count, array_of_requests, flag, ptr_array_of_statuses);

	temps_final = TIME;

	if (ierror == MPI_SUCCESS && *flag)
	{
		TRACE_MPIEVENT (temps_inicial, MPI_TESTALL_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY,
			EMPTY);

		if (Test_C_Software_Counter != 0)
			TRACE_EVENT(temps_inicial, MPI_TEST_COUNTER_EV, Test_C_Software_Counter);

		Test_C_Software_Counter = 0;

		for (ireq = 0; ireq < count; ireq++)
		{
			if ((hash_req = hash_search (&requests, save_reqs[ireq])) != NULL)
			{
				if ((ret = get_Irank_obj_C (hash_req, &src_world, &size, &tag, &(ptr_array_of_statuses[ireq]))) != MPI_SUCCESS)
					return ret;

				if (hash_req->group != MPI_GROUP_NULL)
				{
					ret = PMPI_Group_free (&hash_req->group);
					MPI_CHECK(ret, PMPI_Group_free);
				}

                /* MPI Stats get_Irank_obj_C above returns size (number of bytes received) */
                updateStats_P2P(global_mpi_stats, src_world, size, 0);

				TRACE_MPIEVENT_NOHWC (temps_final, MPI_IRECVED_EV, EMPTY, src_world, size, hash_req->tag, hash_req->commid, hash_req->key);
				hash_remove (&requests, save_reqs[ireq]);
			}
		}
		TRACE_MPIEVENT (temps_final, MPI_TESTALL_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
	}
	else
	{
		if (Test_C_Software_Counter == 0)
			TRACE_EVENTANDCOUNTERS (temps_inicial, MPI_TEST_COUNTER_EV, 0, TRUE);
		Test_C_Software_Counter ++;
	}

	return ierror;
}

/******************************************************************************
 ***  MPI_Testany_C_Wrapper
 ******************************************************************************/

int MPI_Testany_C_Wrapper (int count, MPI_Request *array_of_requests,
                           int *index, int *flag, MPI_Status *status)
{
	MPI_Status my_status, *ptr_status;
	MPI_Request save_reqs[MAX_WAIT_REQUESTS];
	hash_data_t *hash_req;
	int src_world, size, tag, ret, ierror;
#if defined(DEBUG_MPITRACE)
	int i;
#endif
	iotimer_t temps_final, temps_inicial;
	static int Test_C_Software_Counter = 0;

	temps_inicial = LAST_READ_TIME;

	if (count > MAX_WAIT_REQUESTS)
		fprintf (stderr, PACKAGE_NAME ": PANIC! too many requests in mpi_testany\n");
	memcpy (save_reqs, array_of_requests, count * sizeof (MPI_Request));

#if defined(DEBUG_MPITRACE)
	fprintf (stderr, PACKAGE_NAME" %d: TESTANY summary\n", TASKID);
	for (i = 0; i < count; i++)
# if SIZEOF_LONG == 8
		fprintf (stderr, "%d: position %d -> request %lu\n", TASKID, i, (UINT64) array_of_requests[i]);
# elif SIZEOF_LONG == 4
		fprintf (stderr, "%d: position %d -> request %llu\n", TASKID, i, (UINT64) array_of_requests[i]);
# endif
#endif

	ptr_status = (MPI_STATUS_IGNORE == status)?&my_status:status;

	ierror = PMPI_Testany (count, array_of_requests, index, flag, ptr_status);

	temps_final = TIME;

	if (*index != MPI_UNDEFINED && ierror == MPI_SUCCESS && *flag)
	{
		TRACE_MPIEVENT (temps_inicial, MPI_TESTANY_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY,
                  EMPTY);
		if (Test_C_Software_Counter != 0)
			TRACE_EVENT(temps_inicial, MPI_TEST_COUNTER_EV, Test_C_Software_Counter);

		Test_C_Software_Counter = 0;

		if ((hash_req = hash_search (&requests, save_reqs[*index])) != NULL)
		{
			if ((ret = get_Irank_obj_C (hash_req, &src_world, &size, &tag, ptr_status)) != MPI_SUCCESS)
				return ret;
			if (hash_req->group != MPI_GROUP_NULL)
			{
				ret = PMPI_Group_free (&hash_req->group);
				MPI_CHECK(ret, PMPI_Group_free);
			}

            /* MPI Stats, get_Irank_obj_C above returns size (number of bytes received) */
            updateStats_P2P(global_mpi_stats, src_world, size, 0);

			TRACE_MPIEVENT_NOHWC (temps_final, MPI_IRECVED_EV, EMPTY, src_world, size, hash_req->tag, hash_req->commid, hash_req->key);
			hash_remove (&requests, save_reqs[*index]);
		}
		TRACE_MPIEVENT (temps_final, MPI_TESTANY_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
	}
	else
	{
		if (Test_C_Software_Counter == 0)
			TRACE_EVENTANDCOUNTERS (temps_inicial, MPI_TEST_COUNTER_EV, 0, TRUE);
		Test_C_Software_Counter ++;
	}

  return ierror;
}

/******************************************************************************
 ***  MPI_Testsome_C_Wrapper
 ******************************************************************************/

int MPI_Testsome_C_Wrapper (int incount, MPI_Request *array_of_requests,
                            int *outcount, int *array_of_indices,
                            MPI_Status *array_of_statuses)
{
	MPI_Status my_statuses[MAX_WAIT_REQUESTS], *ptr_array_of_statuses;
	MPI_Request save_reqs[MAX_WAIT_REQUESTS];
	hash_data_t *hash_req;
	int src_world, size, tag, ret, ierror, ii;
	iotimer_t temps_final, temps_inicial;
#if defined(DEBUG_MPITRACE)
	int index;
#endif
	static int Test_C_Software_Counter = 0;

	temps_inicial = LAST_READ_TIME;

	if (incount > MAX_WAIT_REQUESTS)
		fprintf (stderr, PACKAGE_NAME": PANIC! too many requests in mpi_testsome\n");

	memcpy (save_reqs, array_of_requests, incount * sizeof (MPI_Request));

#if defined(DEBUG_MPITRACE)
	fprintf (stderr, PACKAGE_NAME " %d: WAITSOME summary\n", TASKID);
	for (index = 0; index < incount; index++)
# if SIZEOF_LONG == 8
		fprintf (stderr, "%d: position %d -> request %lu\n", TASKID, index, (UINT64) array_of_requests[index]);
# elif SIZEOF_LONG == 4
		fprintf (stderr, "%d: position %d -> request %llu\n", TASKID, index, (UINT64) array_of_requests[index]);
# endif
#endif

	ptr_array_of_statuses = (MPI_STATUSES_IGNORE == array_of_statuses)?my_statuses:array_of_statuses;

	ierror = PMPI_Waitsome (incount, array_of_requests, outcount, 
		array_of_indices, ptr_array_of_statuses);

	temps_final = TIME;

	if (ierror == MPI_SUCCESS && *outcount > 0)
	{
		TRACE_MPIEVENT (temps_inicial, MPI_TESTSOME_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY,
                  EMPTY);
		if (Test_C_Software_Counter != 0)
			TRACE_EVENT(temps_inicial, MPI_TEST_COUNTER_EV, Test_C_Software_Counter);
		Test_C_Software_Counter = 0;

		for (ii = 0; ii < *outcount; ii++)
		{
			if ((hash_req = hash_search (&requests, save_reqs[array_of_indices[ii]])) != NULL)
			{
				if ((ret = get_Irank_obj_C (hash_req, &src_world, &size, &tag, &(ptr_array_of_statuses[ii]))) != MPI_SUCCESS)
					return ret;
				if (hash_req->group != MPI_GROUP_NULL)
				{
					ret = PMPI_Group_free (&hash_req->group);
					MPI_CHECK(ret, PMPI_Group_free);
				}

                /* MPI Stats, get_Irank_obj_C above returns size (number of bytes received) */
                updateStats_P2P(global_mpi_stats, src_world, size, 0);

				TRACE_MPIEVENT_NOHWC (temps_final, MPI_IRECVED_EV, EMPTY, src_world, size, hash_req->tag, hash_req->commid, save_reqs[array_of_indices[ii]]);
				hash_remove (&requests, save_reqs[array_of_indices[ii]]);
			}
		}
		TRACE_MPIEVENT (temps_final, MPI_TESTSOME_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
	}
	else
	{
		if (Test_C_Software_Counter == 0)
			TRACE_EVENTANDCOUNTERS (temps_inicial, MPI_TEST_COUNTER_EV, 0, TRUE);
		Test_C_Software_Counter ++;
	}

	return ierror;
}



/******************************************************************************
 ***  MPI_Wait_C_Wrapper
 ******************************************************************************/

int MPI_Wait_C_Wrapper (MPI_Request *request, MPI_Status *status)
{
	MPI_Status my_status, *ptr_status;
	MPI_Request req;
	hash_data_t *hash_req;
	int src_world, size, tag, ret, ierror;
	iotimer_t temps_final;

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_WAIT_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	req = *request;

	ptr_status = (MPI_STATUS_IGNORE == status)?&my_status:status;

	ierror = PMPI_Wait (request, ptr_status);

	temps_final = TIME;

	if (ierror == MPI_SUCCESS && ((hash_req = hash_search (&requests, req)) != NULL))
	{
		if ((ret = get_Irank_obj_C (hash_req, &src_world, &size, &tag, ptr_status)) != MPI_SUCCESS)
			return ret;
		if (hash_req->group != MPI_GROUP_NULL)
		{
			ret = PMPI_Group_free (&hash_req->group);
			MPI_CHECK(ret,PMPI_Group_free);
		}

        /* MPI Stats, get_Irank_obj_C above returns size (number of bytes received) */
        updateStats_P2P(global_mpi_stats, src_world, size, 0);

		TRACE_MPIEVENT_NOHWC (temps_final, MPI_IRECVED_EV, EMPTY, src_world, size, hash_req->tag, hash_req->commid, hash_req->key);
		hash_remove (&requests, req);
	}
	TRACE_MPIEVENT (temps_final, MPI_WAIT_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	return ierror;
}



/******************************************************************************
 ***  MPI_Waitall_C_Wrapper
 ******************************************************************************/

int MPI_Waitall_C_Wrapper (int count, MPI_Request *array_of_requests,
                           MPI_Status *array_of_statuses)
{
	MPI_Status my_statuses[MAX_WAIT_REQUESTS], *ptr_array_of_statuses;
	MPI_Request save_reqs[MAX_WAIT_REQUESTS];
	hash_data_t *hash_req;
	int src_world, size, tag, ret, ireq, ierror;
	iotimer_t temps_final;
#if defined(DEBUG_MPITRACE)
	int index;
#endif

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_WAITALL_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY,
	  EMPTY);

	if (count > MAX_WAIT_REQUESTS)
		fprintf (stderr, PACKAGE_NAME": PANIC! too many requests in mpi_waitall\n");
	memcpy (save_reqs, array_of_requests, count * sizeof (MPI_Request));

#if defined(DEBUG_MPITRACE)
	fprintf (stderr,  PACKAGE_NAME" %d: WAITALL summary\n", TASKID);
	for (index = 0; index < count; index++)
# if SIZEOF_LONG == 8
		fprintf (stderr, "%d: position %d -> request %lu\n", TASKID, index, (UINT64) array_of_requests[index]);
# elif SIZEOF_LONG == 4
		fprintf (stderr, "%d: position %d -> request %llu\n", TASKID, index, (UINT64) array_of_requests[index]);
# endif
#endif

	ptr_array_of_statuses = (MPI_STATUSES_IGNORE == array_of_statuses)?my_statuses:array_of_statuses;

	ierror = PMPI_Waitall (count, array_of_requests, ptr_array_of_statuses);

	temps_final = TIME;

	if (ierror == MPI_SUCCESS)
	{
		for (ireq = 0; ireq < count; ireq++)
		{
			if ((hash_req = hash_search (&requests, save_reqs[ireq])) != NULL)
			{
				if ((ret = get_Irank_obj_C (hash_req, &src_world, &size, &tag, &(ptr_array_of_statuses[ireq]))) != MPI_SUCCESS)
					return ret;
				if (hash_req->group != MPI_GROUP_NULL)
				{
					ret = PMPI_Group_free (&hash_req->group);
					MPI_CHECK(ret, PMPI_Group_free);
				}

                /* MPI Stats, get_Irank_obj_C above returns size (number of bytes received) */
                updateStats_P2P(global_mpi_stats, src_world, size, 0);

				TRACE_MPIEVENT_NOHWC (temps_final, MPI_IRECVED_EV, EMPTY, src_world, size, hash_req->tag, hash_req->commid, hash_req->key);
				hash_remove (&requests, save_reqs[ireq]);
			}
		}
	}
	TRACE_MPIEVENT (temps_final, MPI_WAITALL_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
	return ierror;
}



/******************************************************************************
 ***  MPI_Waitany_C_Wrapper
 ******************************************************************************/

int MPI_Waitany_C_Wrapper (int count, MPI_Request *array_of_requests,
                           int *index, MPI_Status *status)
{
	MPI_Status my_status, *ptr_status;
	MPI_Request save_reqs[MAX_WAIT_REQUESTS];
	hash_data_t *hash_req;
	int src_world, size, tag, ret, ierror;
#if defined(DEBUG_MPITRACE)
	int i;
#endif
	iotimer_t temps_final;

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_WAITANY_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY,
	  EMPTY);

	if (count > MAX_WAIT_REQUESTS)
		fprintf (stderr, PACKAGE_NAME ": PANIC! too many requests in mpi_waitany\n");
	memcpy (save_reqs, array_of_requests, count * sizeof (MPI_Request));

#if defined(DEBUG_MPITRACE)
	fprintf (stderr, PACKAGE_NAME" %d: WAITANY summary\n", TASKID);
	for (i = 0; i < count; i++)
# if SIZEOF_LONG == 8
		fprintf (stderr, "%d: position %d -> request %lu\n", TASKID, i, (UINT64) array_of_requests[i]);
# elif SIZEOF_LONG == 4
		fprintf (stderr, "%d: position %d -> request %llu\n", TASKID, i, (UINT64) array_of_requests[i]);
# endif
#endif

	ptr_status = (MPI_STATUS_IGNORE == status)?&my_status:status;

	ierror = PMPI_Waitany (count, array_of_requests, index, ptr_status);

	temps_final = TIME;

	if (*index != MPI_UNDEFINED && ierror == MPI_SUCCESS)
	{
		if ((hash_req = hash_search (&requests, save_reqs[*index])) != NULL)
		{
			if ((ret = get_Irank_obj_C (hash_req, &src_world, &size, &tag, ptr_status)) != MPI_SUCCESS)
				return ret;
			if (hash_req->group != MPI_GROUP_NULL)
			{
				ret = PMPI_Group_free (&hash_req->group);
				MPI_CHECK(ret, PMPI_Group_free);
			}

            /* MPI Stats, get_Irank_obj_C above returns size (number of bytes received) */
            updateStats_P2P(global_mpi_stats, src_world, size, 0);

			TRACE_MPIEVENT_NOHWC (temps_final, MPI_IRECVED_EV, EMPTY, src_world, size, hash_req->tag, hash_req->commid, hash_req->key);
			hash_remove (&requests, save_reqs[*index]);
		}
	}
	TRACE_MPIEVENT (temps_final, MPI_WAITANY_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
	return ierror;
}


/******************************************************************************
 ***  MPI_Waitsome_C_Wrapper
 ******************************************************************************/

int MPI_Waitsome_C_Wrapper (int incount, MPI_Request *array_of_requests,
                            int *outcount, int *array_of_indices,
                            MPI_Status *array_of_statuses)
{
	MPI_Status my_statuses[MAX_WAIT_REQUESTS], *ptr_array_of_statuses;
	MPI_Request save_reqs[MAX_WAIT_REQUESTS];
	hash_data_t *hash_req;
	int src_world, size, tag, ret, ierror, ii;
	iotimer_t temps_final;
#if defined(DEBUG_MPITRACE)
	int index;
#endif

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_WAITSOME_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY,
	  EMPTY);

	if (incount > MAX_WAIT_REQUESTS)
		fprintf (stderr, PACKAGE_NAME": PANIC! too many requests in mpi_waitsome\n");
	memcpy (save_reqs, array_of_requests, incount * sizeof (MPI_Request));

#if defined(DEBUG_MPITRACE)
	fprintf (stderr, PACKAGE_NAME " %d: WAITSOME summary\n", TASKID);
	for (index = 0; index < incount; index++)
# if SIZEOF_LONG == 8
		fprintf (stderr, "%d: position %d -> request %lu\n", TASKID, index, (UINT64) array_of_requests[index]);
# elif SIZEOF_LONG == 4
		fprintf (stderr, "%d: position %d -> request %llu\n", TASKID, index, (UINT64) array_of_requests[index]);
# endif
#endif

	ptr_array_of_statuses = (MPI_STATUSES_IGNORE == array_of_statuses)?my_statuses:array_of_statuses;

	ierror = PMPI_Waitsome (incount, array_of_requests, outcount, 
	  array_of_indices, ptr_array_of_statuses);

	temps_final = TIME;

	if (ierror == MPI_SUCCESS && *outcount > 0)
	{
		for (ii = 0; ii < *outcount; ii++)
		{
			if ((hash_req = hash_search (&requests, save_reqs[array_of_indices[ii]])) != NULL)
			{
				if ((ret = get_Irank_obj_C (hash_req, &src_world, &size, &tag, &(ptr_array_of_statuses[ii]))) != MPI_SUCCESS)
					return ret;
				if (hash_req->group != MPI_GROUP_NULL)
				{
					ret = PMPI_Group_free (&hash_req->group);
					MPI_CHECK(ret, PMPI_Group_free);
				}

                /* MPI Stats, get_Irank_obj_C above returns size (number of bytes received) */
                updateStats_P2P(global_mpi_stats, src_world, size, 0);

				TRACE_MPIEVENT_NOHWC (temps_final, MPI_IRECVED_EV, EMPTY, src_world, size, hash_req->tag, hash_req->commid, save_reqs[array_of_indices[ii]]);
				hash_remove (&requests, save_reqs[array_of_indices[ii]]);
			}
		}
	}
	TRACE_MPIEVENT (temps_final, MPI_WAITSOME_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
	return ierror;
}


/******************************************************************************
 ***  MPI_Recv_init_C_Wrapper
 ******************************************************************************/

int MPI_Recv_init_C_Wrapper (void *buf, int count, MPI_Datatype datatype, int source,
                             int tag, MPI_Comm comm, MPI_Request *request)
{
  int ierror;

  /*
   *   type : RECV_INIT_EV                 value : EVT_BEGIN
   *   target : ---                        size  : ----
   *   tag : ---                           comm : ---
   *   aux : ---
   */
  TRACE_MPIEVENT (LAST_READ_TIME, MPI_RECV_INIT_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY,
                  EMPTY);

  /*
   * Primer cal fer la crida real 
   */
  ierror = PMPI_Recv_init (buf, count, datatype, source, tag, comm,
    request);

  /*
   * Es guarda aquesta request 
   */
	PR_NewRequest (MPI_IRECV_EV, count, datatype, source, tag, comm, *request,
                 &PR_queue);

  /*
   *   type : RECV_INIT_EV                 value : EVT_END
   *   target : ---                        size  : ----
   *   tag : ---                           comm : ---
   *   aux : ---
   */
  TRACE_MPIEVENT (TIME, MPI_RECV_INIT_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY,
                  EMPTY);

	updateStats_OTHER(global_mpi_stats);

  return ierror;
}


/******************************************************************************
 ***  MPI_Send_init_C_Wrapper
 ******************************************************************************/

int MPI_Send_init_C_Wrapper (void *buf, int count, MPI_Datatype datatype, int dest,
                             int tag, MPI_Comm comm, MPI_Request *request)
{
  int ierror;

  /*
   *   type : SEND_INIT_EV                 value : EVT_BEGIN
   *   target : ---                        size  : ----
   *   tag : ---                           comm : ---
   *   aux : ---
   */
  TRACE_MPIEVENT (LAST_READ_TIME, MPI_SEND_INIT_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY,
                  EMPTY);

  /*
   * Primer cal fer la crida real 
   */
  ierror = PMPI_Send_init (buf, count, datatype, dest, tag, comm,
    request);

  /*
   * Es guarda aquesta request 
   */
	PR_NewRequest (MPI_ISEND_EV, count, datatype, dest, tag, comm, *request,
                 &PR_queue);

  /*
   *   type : SEND_INIT_EV                 value : EVT_END
   *   target : ---                        size  : ----
   *   tag : ---                           comm : ---
   *   aux : ---
   */
  TRACE_MPIEVENT (TIME, MPI_SEND_INIT_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY,
                  EMPTY);

	updateStats_OTHER(global_mpi_stats);

  return ierror;
}


/******************************************************************************
 ***  MPI_Bsend_init_C_Wrapper
 ******************************************************************************/

int MPI_Bsend_init_C_Wrapper (void *buf, int count, MPI_Datatype datatype, int dest,
                              int tag, MPI_Comm comm, MPI_Request *request)
{
  int ierror;

  /*
   *   type : BSEND_INIT_EV                value : EVT_BEGIN
   *   target : ---                        size  : ----
   *   tag : ---                           comm : ---
   *   aux : ---
   */
  TRACE_MPIEVENT (LAST_READ_TIME, MPI_BSEND_INIT_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY,
                  EMPTY);

  /*
   * Primer cal fer la crida real 
   */
  ierror = PMPI_Bsend_init (buf, count, datatype, dest, tag, comm,
    request);

  /*
   * Es guarda aquesta request 
   */
	PR_NewRequest (MPI_IBSEND_EV, count, datatype, dest, tag, comm, *request,
                 &PR_queue);

  /*
   *   type : BSEND_INIT_EV                value : EVT_END
   *   target : ---                        size  : ----
   *   tag : ---                           comm : ---
   *   aux : ---
   */
  TRACE_MPIEVENT (TIME, MPI_BSEND_INIT_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY,
                  EMPTY);

	updateStats_OTHER(global_mpi_stats);

  return ierror;
}


/******************************************************************************
 ***  MPI_Rsend_init_C_Wrapper
 ******************************************************************************/

int MPI_Rsend_init_C_Wrapper (void *buf, int count, MPI_Datatype datatype, int dest,
                              int tag, MPI_Comm comm, MPI_Request *request)
{
  int ierror;

  /*
   *   type : RSEND_INIT_EV                value : EVT_BEGIN
   *   target : ---                        size  : ----
   *   tag : ---                           comm : ---
   *   aux : ---
   */
  TRACE_MPIEVENT (LAST_READ_TIME, MPI_RSEND_INIT_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY,
                  EMPTY);

  /*
   * Primer cal fer la crida real 
   */
  ierror = PMPI_Rsend_init (buf, count, datatype, dest, tag, comm,
    request);

  /*
   * Es guarda aquesta request 
   */
	PR_NewRequest (MPI_IRSEND_EV, count, datatype, dest, tag, comm, *request,
                 &PR_queue);

  /*
   *   type : RSEND_INIT_EV                value : EVT_END
   *   target : ---                        size  : ----
   *   tag : ---                           comm : ---
   *   aux : ---
   */
  TRACE_MPIEVENT (TIME, MPI_RSEND_INIT_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY,
                  EMPTY);

	updateStats_OTHER(global_mpi_stats);

  return ierror;
}


/******************************************************************************
 ***  MPI_Ssend_init_C_Wrapper
 ******************************************************************************/

int MPI_Ssend_init_C_Wrapper (void *buf, int count, MPI_Datatype datatype, int dest,
                              int tag, MPI_Comm comm, MPI_Request *request)
{
	int ierror;

	/*
	*   type : SSEND_INIT_EV                value : EVT_BEGIN
	*   target : ---                        size  : ----
	*   tag : ---                           comm : ---
	*   aux : ---
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_SSEND_INIT_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY,
	  EMPTY);

	/*
	* Primer cal fer la crida real 
	*/
	ierror = PMPI_Ssend_init (buf, count, datatype, dest, tag, comm,
	  request);

	/*
	 * Es guarda aquesta request 
	 */
	PR_NewRequest (MPI_ISSEND_EV, count, datatype, dest, tag, comm, *request,
                 &PR_queue);

	/*
	 *   type : SSEND_INIT_EV                value : EVT_END
	 *   target : ---                        size  : ----
	 *   tag : ---                           comm : ---
	 *   aux : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_SSEND_INIT_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY,
	  EMPTY);

	updateStats_OTHER(global_mpi_stats);

	return ierror;
}


int MPI_Sendrecv_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype,
	int dest, int sendtag, void *recvbuf, int recvcount, MPI_Datatype recvtype,
	int source, int recvtag, MPI_Comm comm, MPI_Status * status) 
{
	MPI_Status my_status, *ptr_status;
	int ierror, ret;
	int DataSendSize, DataRecvSize, DataSend, DataSize;
	int SendRank, SourceRank, RecvRank, Count, Tag;

	if ((ret = get_rank_obj_C (comm, dest, &RecvRank, RANK_OBJ_SEND)) != MPI_SUCCESS)
		return ret;

	if (sendcount != 0)
	{
		ret = PMPI_Type_size (sendtype, &DataSendSize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	if (recvcount != 0)
	{
		ret = PMPI_Type_size (recvtype, &DataRecvSize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	DataSend = sendcount * DataSendSize;

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_SENDRECV_EV, EVT_BEGIN, RecvRank, DataSend, sendtag,
		comm, EMPTY);

	ptr_status = (status == MPI_STATUS_IGNORE)?&my_status:status;

	ierror = PMPI_Sendrecv (sendbuf, sendcount, sendtype, dest, sendtag,
		recvbuf, recvcount, recvtype, source, recvtag, comm, ptr_status);

	ret = PMPI_Get_count (ptr_status, recvtype, &Count);
	MPI_CHECK(ret, PMPI_Get_count);

	if (Count != MPI_UNDEFINED)
		DataSize = DataRecvSize * Count;
	else
		DataSize = 0;

	if (source == MPI_ANY_SOURCE)
		SendRank = ptr_status->MPI_SOURCE;
	else
		SendRank = source;

	if (recvtag == MPI_ANY_TAG)
		Tag = ptr_status->MPI_TAG;
	else
		Tag = recvtag;

	if ((ret = get_rank_obj_C (comm, SendRank, &SourceRank, RANK_OBJ_RECV)) != MPI_SUCCESS)
		return ret;

	TRACE_MPIEVENT (TIME, MPI_SENDRECV_EV, EVT_END, SourceRank, DataSize, Tag, comm,
	  EMPTY);

	/* MPI Stats */
    updateStats_P2P(global_mpi_stats, SourceRank, DataSize, DataSend);

	return ierror;
}

int MPI_Sendrecv_replace_C_Wrapper (void *buf, int count, MPI_Datatype type,
  int dest, int sendtag, int source, int recvtag, MPI_Comm comm,
  MPI_Status * status) 
{
	MPI_Status my_status, *ptr_status;
	int ierror, ret;
	int DataSendSize, DataRecvSize, DataSend, DataSize;
	int SendRank, SourceRank, RecvRank, Count, Tag;

	if ((ret = get_rank_obj_C (comm, dest, &RecvRank, RANK_OBJ_SEND)) != MPI_SUCCESS)
		return ret;

	if (count != 0)
	{
		ret = PMPI_Type_size (type, &DataSendSize);
		MPI_CHECK(ret, PMPI_Type_size);
	}

	DataRecvSize = DataSendSize;

	DataSend = count * DataSendSize;

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_SENDRECV_REPLACE_EV, EVT_BEGIN, RecvRank, DataSend,
	  sendtag, comm, EMPTY);

	ptr_status = (status == MPI_STATUS_IGNORE)?&my_status:status;

	ierror = PMPI_Sendrecv_replace (buf, count, type, dest, sendtag, source,
	  recvtag, comm, ptr_status);

	ret = PMPI_Get_count (ptr_status, type, &Count);
	MPI_CHECK(ret, PMPI_Get_count);

	if (Count != MPI_UNDEFINED)
		DataSize = DataRecvSize * Count;
	else
		DataSize = 0;

	if (source == MPI_ANY_SOURCE)
		SendRank = ptr_status->MPI_SOURCE;
	else
		SendRank = source;

	if (recvtag == MPI_ANY_TAG)
		Tag = ptr_status->MPI_TAG;
	else
		Tag = recvtag;

	if ((ret = get_rank_obj_C (comm, SendRank, &SourceRank, RANK_OBJ_RECV)) != MPI_SUCCESS)
		return ret;

	TRACE_MPIEVENT (TIME, MPI_SENDRECV_REPLACE_EV, EVT_END, SourceRank, DataSize,
	  Tag, comm, EMPTY);

	/* MPI Stats */
    updateStats_P2P(global_mpi_stats, SourceRank, DataSize, DataSend);

	return ierror;
}

#endif /* defined(C_SYMBOLS) */

