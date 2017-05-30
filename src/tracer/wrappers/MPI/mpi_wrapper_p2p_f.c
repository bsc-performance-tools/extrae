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
#include "extrae_mpif.h"
#include "utils.h"
#include "utils_mpi.h"
#include "mpi_wrapper.h"
#include "mpi_interface_coll_helper.h"
#include "mpi_wrapper.h"
#include "wrapper.h"
#include "clock.h"
#include "signals.h"
#include "misc_wrapper.h"
#include "mpi_interface.h"
#include "mode.h"
#include "threadinfo.h"

#include "hash_table.h"

#if !defined(MPI_HAS_MPI_F_STATUS_IGNORE)
#warning MPI_F_STATUS_IGNORE and MPI_F_STATUSES_IGNORE definitions not found in mpi.h. Assuming an integer pointer data type. We have detected this situation only in IBM Platform MPI on top of MPICH 1.2, please verify that in your current MPI implementation the datatype is compliant.
MPI_Fint * MPI_F_STATUS_IGNORE = 0;
MPI_Fint * MPI_F_STATUSES_IGNORE = 0;
#endif

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

#if defined(FORTRAN_SYMBOLS)

#define MAX_WAIT_REQUESTS 16384

/******************************************************************************
 ***  PMPI_BSend_Wrapper
 ******************************************************************************/

void PMPI_BSend_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror)
{
	int size, receiver, ret;
	MPI_Comm c = MPI_Comm_f2c (*comm);

	if (*count != 0)
	{
		CtoF77 (pmpi_type_size) (datatype, &size, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}

	size *= (*count);

	if ((ret = get_rank_obj (comm, dest, &receiver, RANK_OBJ_SEND)) != MPI_SUCCESS)
	{
		*ierror = ret;
		return;
	}

	/*
	*   event : BSEND_EV                      value : EVT_BEGIN
	*   target : receiver                     size  : send message size
	*   tag : message tag
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_BSEND_EV, EVT_BEGIN, receiver, size, *tag, c, EMPTY);

	CtoF77 (pmpi_bsend) (buf, count, datatype, dest, tag, comm, ierror);


	/*
	*   event : BSEND_EV                      value : EVT_END
	*   target : receiver                     size  : send message size
	*   tag : message tag
	*/
	TRACE_MPIEVENT (TIME, MPI_BSEND_EV, EVT_END, receiver, size, *tag, c, EMPTY);

	/* MPI Stats */
	updateStats_P2P(global_mpi_stats, receiver, 0, size);
}

/******************************************************************************
 ***  PMPI_SSend_Wrapper
 ******************************************************************************/

void PMPI_SSend_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror)
{
	int size, receiver, ret;
	MPI_Comm c = MPI_Comm_f2c (*comm);

	if (*count != 0)
	{
		CtoF77 (pmpi_type_size) (datatype, &size, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}

	size *= (*count);

	if ((ret = get_rank_obj (comm, dest, &receiver, RANK_OBJ_SEND)) != MPI_SUCCESS)
	{
		*ierror = ret;
		return;
	}

	/*
	*   event : SSEND_EV                      value : EVT_BEGIN
	*   target : receiver                     size  : send message size
	*   tag : message tag
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_SSEND_EV, EVT_BEGIN, receiver, size, *tag, c, EMPTY);

	CtoF77 (pmpi_ssend) (buf, count, datatype, dest, tag, comm, ierror);

	/*
	*   event : SSEND_EV                      value : EVT_END
	*   target : receiver                     size  : send message size
	*   tag : message tag
	*/
	TRACE_MPIEVENT (TIME, MPI_SSEND_EV, EVT_END, receiver, size, *tag, c, EMPTY);

	/* MPI Stats */
	updateStats_P2P(global_mpi_stats, receiver, 0, size);
}

/******************************************************************************
 ***  PMPI_RSend_Wrapper
 ******************************************************************************/

void PMPI_RSend_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror)
{
	int size, receiver, ret;
	MPI_Comm c = MPI_Comm_f2c (*comm);

	if (*count != 0)
	{
		CtoF77 (pmpi_type_size) (datatype, &size, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}

	size *= (*count);

	if ((ret = get_rank_obj (comm, dest, &receiver, RANK_OBJ_SEND)) != MPI_SUCCESS)
	{
		*ierror = ret;
		return;
	}

	/*
	*   event : RSEND_EV                      value : EVT_BEGIN
	*   target : receiver                     size  : send message size
	*   tag : message tag
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_RSEND_EV, EVT_BEGIN, receiver, size, *tag, c, EMPTY);

	CtoF77 (pmpi_rsend) (buf, count, datatype, dest, tag, comm, ierror);

	/*
	*   event : RSEND_EV                      value : EVT_END
	*   target : receiver                     size  : send message size
	*   tag : message tag
	*/
	TRACE_MPIEVENT (TIME, MPI_RSEND_EV, EVT_END, receiver, size, *tag, c, EMPTY);

	/* MPI Stats */
	updateStats_P2P(global_mpi_stats, receiver, 0, size);
}

/******************************************************************************
 ***  PMPI_Send_Wrapper
 ******************************************************************************/

void PMPI_Send_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror)
{
	int size, receiver, ret;
	MPI_Comm c = MPI_Comm_f2c (*comm);

	if (*count != 0)
	{
		CtoF77 (pmpi_type_size) (datatype, &size, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}

	size *= (*count);

	if ((ret = get_rank_obj (comm, dest, &receiver, RANK_OBJ_SEND)) != MPI_SUCCESS)
	{
		*ierror = ret;
		return;
	}

	/*
	*   event : SEND_EV                       value : EVT_BEGIN
	*   target : receiver                     size  : send message size
	*   tag : message tag
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_SEND_EV, EVT_BEGIN, receiver, size, *tag, c, EMPTY);

	CtoF77 (pmpi_send) (buf, count, datatype, dest, tag, comm, ierror);

	/*
	*   event : SEND_EV                       value : EVT_END
	*   target : receiver                     size  : send message size
	*   tag : message tag
	*/
	TRACE_MPIEVENT (TIME, MPI_SEND_EV, EVT_END, receiver, size, *tag, c, EMPTY);

	/* MPI Stats */
	updateStats_P2P(global_mpi_stats, receiver, 0, size);
}


/******************************************************************************
 ***  PMPI_IBSend_Wrapper
 ******************************************************************************/

void PMPI_IBSend_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)
{
	int size, receiver, ret;
	MPI_Comm c = MPI_Comm_f2c (*comm);

	if (*count != 0)
	{
		CtoF77 (pmpi_type_size) (datatype, &size, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}

	size *= (*count);

	if ((ret = get_rank_obj (comm, dest, &receiver, RANK_OBJ_SEND)) != MPI_SUCCESS)
	{
		*ierror = ret;
		return;
	}

	/*
	*   event : IBSEND_EV                     value : EVT_BEGIN
	*   target : ---                          size  : ---
	*   tag : ---                             commid: ---
	*   aux : ---
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IBSEND_EV, EVT_BEGIN, receiver, size, *tag, c, EMPTY);

	CtoF77 (pmpi_ibsend) (buf, count, datatype, dest, tag, comm, request,
	  ierror);

	/*
	*   event : IBSEND_EV                     value : EVT_END
	*   target : ---                          size  : ---
	*   tag : ---                             commid: ---
	*   aux : ---
	*/
	TRACE_MPIEVENT (TIME, MPI_IBSEND_EV, EVT_END, receiver, size, *tag, c, EMPTY);

	/* MPI Stats */
	updateStats_P2P(global_mpi_stats, receiver, 0, size);
}

/******************************************************************************
 ***  PMPI_ISend_Wrapper
 ******************************************************************************/

void PMPI_ISend_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)
{
	int size, receiver, ret;
	MPI_Comm c = MPI_Comm_f2c (*comm);

	if (*count != 0)
	{
		CtoF77 (pmpi_type_size) (datatype, &size, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}

	size *= (*count);

	if ((ret = get_rank_obj (comm, dest, &receiver, RANK_OBJ_SEND)) != MPI_SUCCESS)
	{
		*ierror = ret;
		return;
	} 

	/*
	*   event : ISEND_EV                      value : EVT_BEGIN
	*   target : ---                          size  : ---
	*   tag : ---                             commid: ---
	*   aux : ---
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_ISEND_EV, EVT_BEGIN, receiver, size, *tag, c, EMPTY);

	CtoF77 (pmpi_isend) (buf, count, datatype, dest, tag, comm, request,
	  ierror);
	/*
	*   event : ISEND_EV                      value : EVT_END
	*   target : ---                          size  : ---
	*   tag : ---                             commid: ---
	*   aux : ---
	*/
	TRACE_MPIEVENT (TIME, MPI_ISEND_EV, EVT_END, receiver, size, *tag, c, EMPTY);

	/* MPI Stats */
	updateStats_P2P(global_mpi_stats, receiver, 0, size);
}

/******************************************************************************
 ***  PMPI_ISSend_Wrapper
 ******************************************************************************/

void PMPI_ISSend_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)
{
	int size, receiver, ret;
	MPI_Comm c = MPI_Comm_f2c (*comm);

	if (*count != 0)
	{
		CtoF77 (pmpi_type_size) (datatype, &size, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}

	size *= (*count);

	if ((ret = get_rank_obj (comm, dest, &receiver, RANK_OBJ_SEND)) != MPI_SUCCESS)
	{
		*ierror = ret;
		return;
	}

	/*
	*   event : ISSEND_EV                     value : EVT_BEGIN
	*   target : ---                          size  : ---
	*   tag : ---                             commid: ---
	*   aux : ---
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_ISSEND_EV, EVT_BEGIN, receiver, size, *tag, c, EMPTY);

	CtoF77 (pmpi_issend) (buf, count, datatype, dest, tag, comm, request,
	  ierror);

	/*
	*   event : ISSEND_EV                     value : EVT_END
	*   target : ---                          size  : ---
	*   tag : ---                             commid: ---
	*   aux : ---
	*/
	TRACE_MPIEVENT (TIME, MPI_ISSEND_EV, EVT_END, receiver, size, *tag, c, EMPTY);

	/* MPI Stats */
	updateStats_P2P(global_mpi_stats, receiver, 0, size);
}

/******************************************************************************
 ***  PMPI_IRSend_Wrapper
 ******************************************************************************/

void PMPI_IRSend_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)
{
	int size, receiver, ret;
	MPI_Comm c = MPI_Comm_f2c (*comm);

	if (*count != 0)
	{
		CtoF77 (pmpi_type_size) (datatype, &size, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}

	size *= (*count);

	if ((ret = get_rank_obj (comm, dest, &receiver, RANK_OBJ_SEND)) != MPI_SUCCESS)
	{
		*ierror = ret;
		return;
	}

	/*
	*   event : IRSEND_EV                     value : EVT_BEGIN
	*   target : ---                          size  : ---
	*   tag : ---                             commid: ---
	*   aux : ---
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IRSEND_EV, EVT_BEGIN, receiver, size, *tag, c, EMPTY);

	CtoF77 (pmpi_irsend) (buf, count, datatype, dest, tag, comm, request,
	  ierror);

	/*
	*   event : IRSEND_EV                     value : EVT_END
	*   target : ---                          size  : ---
	*   tag : ---                             commid: ---
	*   aux : ---
	*/
	TRACE_MPIEVENT (TIME, MPI_IRSEND_EV, EVT_END, receiver, size, *tag, c, EMPTY);

	/* MPI Stats */
	updateStats_P2P(global_mpi_stats, receiver, 0, size);
}

/******************************************************************************
 ***  PMPI_Recv_Wrapper
 ******************************************************************************/

void PMPI_Recv_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *status, 
	MPI_Fint *ierror)
{
	MPI_Fint my_status[SIZEOF_MPI_STATUS], *ptr_status;
	MPI_Comm c = MPI_Comm_f2c(*comm);
	int size, src_world, sender_src, ret, recved_count, sended_tag;

	if (*count != 0)
	{
		CtoF77 (pmpi_type_size) (datatype, &size, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		size = 0;

	if ((ret = get_rank_obj (comm, source, &src_world, RANK_OBJ_RECV)) != MPI_SUCCESS)
	{
		*ierror = ret;
		return;
	}

	/*
	*   event : RECV_EV                      value : EVT_BEGIN    
	*   target : MPI_ANY_SOURCE or sender    size  : receive buffer size    
	*   tag : message tag or MPI_ANY_TAG     commid: Communicator identifier
	*   aux: ---
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_RECV_EV, EVT_BEGIN, src_world, (*count) * size, *tag, c, EMPTY);

	ptr_status = (status == MPI_F_STATUS_IGNORE)?my_status:status;

	CtoF77 (pmpi_recv) (buf, count, datatype, source, tag, comm, ptr_status,
	  ierror);

	CtoF77 (pmpi_get_count) (ptr_status, datatype, &recved_count, &ret);
	MPI_CHECK(ret, pmpi_get_count);

	if (recved_count != MPI_UNDEFINED)
		size *= recved_count;
	else
		size = 0;

	if (*source == MPI_ANY_SOURCE)
		sender_src = ptr_status[MPI_SOURCE_OFFSET];
	else
		sender_src = *source;

	if (*tag == MPI_ANY_TAG)
		sended_tag = ptr_status[MPI_TAG_OFFSET];
	else
		sended_tag = *tag;

	if ((ret = get_rank_obj (comm, &sender_src, &src_world, RANK_OBJ_RECV)) != MPI_SUCCESS)
	{
		*ierror = ret;
		return;
	}

	/*
	*   event : RECV_EV                      value : EVT_END
	*   target : sender                      size  : received message size    
	*   tag : message tag
	*/
	TRACE_MPIEVENT (TIME, MPI_RECV_EV, EVT_END, src_world, size, sended_tag, c, EMPTY);

	/* MPI Stats */
	updateStats_P2P(global_mpi_stats, src_world, size, 0);
}

/******************************************************************************
 ***  PMPI_IRecv_Wrapper
 ******************************************************************************/

void PMPI_IRecv_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)
{
	hash_data_t hash_req;
	MPI_Fint inter, ret, size, src_world;
	MPI_Comm c = MPI_Comm_f2c(*comm);

	if (*count != 0)
	{
		CtoF77 (pmpi_type_size) (datatype, &size, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		size = 0;

	if ((ret = get_rank_obj (comm, source, &src_world, RANK_OBJ_RECV)) != MPI_SUCCESS)
	{
		*ierror = ret;
		return;
	}

	/*
	*   event : IRECV_EV                     value : EVT_BEGIN
	*   target : ---                         size  : ---
	*   tag : ---
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IRECV_EV, EVT_BEGIN, src_world, (*count) * size, *tag, c, EMPTY);

	CtoF77 (pmpi_irecv) (buf, count, datatype, source, tag, comm, request,
	  ierror);

	hash_req.key = MPI_Request_f2c(*request);
	hash_req.commid = c;
	hash_req.partner = *source;
	hash_req.tag = *tag;
	hash_req.size = *count * size;
	
	if (c != MPI_COMM_WORLD)
	{
		MPI_Fint group;
		CtoF77 (pmpi_comm_test_inter) (comm, &inter, &ret);
		MPI_CHECK(ret, pmpi_comm_test_inter);

		if (inter)
		{
			CtoF77 (pmpi_comm_remote_group) (comm, &group, &ret);
			MPI_CHECK(ret, pmpi_comm_remote_group);
		}
		else
		{
			CtoF77 (pmpi_comm_group) (comm, &group, &ret);
			MPI_CHECK(ret, pmpi_comm_group);
		}
		hash_req.group = MPI_Group_f2c(group);
	}
	else
		hash_req.group = MPI_GROUP_NULL;

	hash_add (&requests, &hash_req);

	/*
	*   event : IRECV_EV                     value : EVT_END
	*   target : request                     size  : ---
	*   tag : ---
	*/
	TRACE_MPIEVENT (TIME, MPI_IRECV_EV, EVT_END, src_world, (*count) * size, *tag, c, hash_req.key);
}


/******************************************************************************
 ***  PMPI_Probe_Wrapper
 ******************************************************************************/

void PMPI_Probe_Wrapper (MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm,
	MPI_Fint *status, MPI_Fint *ierror)
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

  /*
   *   event : PROBE_EV                     value : EVT_BEGIN
   *   target : ---                         size  : ---
   *   tag : ---
   */
  TRACE_MPIEVENT (LAST_READ_TIME, MPI_PROBE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, c, EMPTY);

  CtoF77 (pmpi_probe) (source, tag, comm, status, ierror);

  /*
   *   event : PROBE_EV                     value : EVT_END
   *   target : ---                         size  : ---
   *   tag : ---
   */
  TRACE_MPIEVENT (TIME, MPI_PROBE_EV, EVT_END, EMPTY, EMPTY, EMPTY, c, EMPTY);

  updateStats_OTHER(global_mpi_stats);
}



/******************************************************************************
 ***  PMPI_IProbe_Wrapper
 ******************************************************************************/

void Bursts_PMPI_IProbe_Wrapper (MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm,
	MPI_Fint *flag, MPI_Fint *status, MPI_Fint *ierror)
{
	MPI_Comm c = MPI_Comm_f2c(*comm);

     /*
      *   event : IPROBE_EV                     value : EVT_BEGIN
      *   target : ---                          size  : ---
      *   tag : ---
      */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IPROBE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, c, EMPTY);

	CtoF77 (pmpi_iprobe) (source, tag, comm, flag, status, ierror);

     /*
      *   event : IPROBE_EV                    value : EVT_END
      *   target : ---                         size  : ---
      *   tag : ---
      */
	TRACE_MPIEVENT (TIME, MPI_IPROBE_EV, EVT_END, EMPTY, EMPTY, EMPTY, c, EMPTY);
}

void Normal_PMPI_IProbe_Wrapper (MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm,
	MPI_Fint *flag, MPI_Fint *status, MPI_Fint *ierror)
{
  static int IProbe_Software_Counter = 0;
  iotimer_t begin_time, end_time;
  static iotimer_t elapsed_time_outside_iprobes = 0, last_iprobe_exit_time = 0;
	MPI_Comm c = MPI_Comm_f2c(*comm);

  begin_time = LAST_READ_TIME;

  if (IProbe_Software_Counter == 0) {
    /* Primer Iprobe */
    elapsed_time_outside_iprobes = 0;
  }
  else {
    elapsed_time_outside_iprobes += (begin_time - last_iprobe_exit_time);
  }

  CtoF77 (pmpi_iprobe) (source, tag, comm, flag, status, ierror);

  end_time = TIME; 
  last_iprobe_exit_time = end_time;

	if (tracejant_mpi)
  {
    if (*flag)
    {
      /*
       *   event : IPROBE_EV                     value : EVT_BEGIN
       *   target : ---                          size  : ---
       *   tag : ---
       */
      if (IProbe_Software_Counter != 0) {
        TRACE_EVENT (begin_time, MPI_TIME_OUTSIDE_IPROBES_EV, elapsed_time_outside_iprobes);
        TRACE_EVENT (begin_time, MPI_IPROBE_COUNTER_EV, IProbe_Software_Counter);
      }
      TRACE_MPIEVENT (begin_time, MPI_IPROBE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, c, EMPTY);

     /*
      *   event : IPROBE_EV                    value : EVT_END
      *   target : ---                         size  : ---
      *   tag : ---
      */
      TRACE_MPIEVENT (end_time, MPI_IPROBE_EV, EVT_END, EMPTY, EMPTY, EMPTY, c, EMPTY);
      IProbe_Software_Counter = 0;
    }
    else
    {
      if (IProbe_Software_Counter == 0)
      {
        /* El primer iprobe que falla */
        TRACE_EVENTANDCOUNTERS (begin_time, MPI_IPROBE_COUNTER_EV, 0, TRUE);
      }
      IProbe_Software_Counter ++;
    }
  }
}

void PMPI_IProbe_Wrapper (MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm,
    MPI_Fint *flag, MPI_Fint *status, MPI_Fint *ierror)
{
	if (CURRENT_TRACE_MODE(THREADID) == TRACE_MODE_BURSTS)
	{
		Bursts_PMPI_IProbe_Wrapper (source, tag, comm, flag, status, ierror);
	}
	else
	{
		Normal_PMPI_IProbe_Wrapper (source, tag, comm, flag, status, ierror);
	}

	updateStats_OTHER(global_mpi_stats);
}

/******************************************************************************
 ***  PMPI_Test_Wrapper
 ******************************************************************************/

void Bursts_PMPI_Test_Wrapper (MPI_Fint *request, MPI_Fint *flag, MPI_Fint *status,
	MPI_Fint *ierror)
{
	MPI_Request req;
	hash_data_t *hash_req = NULL;
	int src_world = -1, size = 0, tag = 0, ret;
	iotimer_t temps_final;

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_TEST_EV, EVT_BEGIN, *request, EMPTY, EMPTY, EMPTY, EMPTY);

	req = MPI_Request_f2c (*request);

	CtoF77 (pmpi_test) (request, flag, status, ierror);

	temps_final = TIME;

	if (*flag && ((hash_req = hash_search (&requests, req)) != NULL))
	{
		int cancelled = 0;

		CtoF77 (pmpi_test_cancelled) (status, &cancelled, ierror);
		if (!cancelled)
		{
			if ((ret = get_Irank_obj (hash_req, &src_world, &size, &tag, status)) != MPI_SUCCESS)
			{
				*ierror = ret;
				return;
			}
			if (hash_req->group != MPI_GROUP_NULL)
			{
				MPI_Fint group = MPI_Group_c2f(hash_req->group);
				CtoF77 (pmpi_group_free) (&group, &ret);
				MPI_CHECK (ret, pmpi_group_free);
			}

			/* MPI Stats */
			/* get_Irank_obj above return size (number of bytes received) */
			updateStats_P2P(global_mpi_stats, src_world, size, 0);
		}
		TRACE_MPIEVENT_NOHWC (temps_final, MPI_IRECVED_EV, cancelled, src_world, size, hash_req->tag, hash_req->commid, req);
		hash_remove (&requests, req);
	}

	TRACE_MPIEVENT (temps_final, MPI_TEST_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
}

void Normal_PMPI_Test_Wrapper (MPI_Fint *request, MPI_Fint *flag, MPI_Fint *status,
	MPI_Fint *ierror)
{
	MPI_Request req;
	hash_data_t *hash_req = NULL;
	int src_world = -1, size = 0, tag = 0, ret;
	iotimer_t begin_time, end_time;
	static int Test_F_Software_Counter = 0;
	static iotimer_t elapsed_time_outside_tests = 0, last_test_exit_time = 0;

	begin_time = LAST_READ_TIME;

	if (Test_F_Software_Counter == 0) {
		/* First MPI_Test */
    		elapsed_time_outside_tests = 0;
  	}
  	else {
    		elapsed_time_outside_tests += (begin_time - last_test_exit_time);
  	}

	req = MPI_Request_f2c(*request);

	CtoF77 (pmpi_test) (request, flag, status, ierror);

	end_time = TIME;

	if (*flag && ((hash_req = hash_search (&requests, req)) != NULL))
	{
                int cancelled = 0;

		if (Test_F_Software_Counter != 0)
		{
			TRACE_EVENT (begin_time, MPI_TIME_OUTSIDE_TESTS_EV, elapsed_time_outside_tests);
			TRACE_EVENT (begin_time, MPI_TEST_COUNTER_EV, Test_F_Software_Counter);
		}
		Test_F_Software_Counter = 0;

		TRACE_MPIEVENT (begin_time, MPI_TEST_EV, EVT_BEGIN, *request, EMPTY, EMPTY, EMPTY, EMPTY);

                CtoF77 (pmpi_test_cancelled) (status, &cancelled, ierror);
                if (!cancelled)
                {

			if ((ret = get_Irank_obj (hash_req, &src_world, &size, &tag, status)) != MPI_SUCCESS)
			{
				*ierror = ret;
				return;
			}
			if (hash_req->group != MPI_GROUP_NULL)
			{
				MPI_Fint group = MPI_Group_c2f (hash_req->group);
				CtoF77 (pmpi_group_free) (&group, &ret);
				MPI_CHECK (ret, pmpi_group_free);
			}
		}
		TRACE_MPIEVENT_NOHWC (begin_time, MPI_IRECVED_EV, cancelled, src_world, size, hash_req->tag, hash_req->commid, req);
		hash_remove (&requests, req);

		TRACE_MPIEVENT (end_time, MPI_TEST_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
	}
	else
	{
		if (Test_F_Software_Counter == 0)
		{
			/* First failed MPI_Test */
			TRACE_EVENTANDCOUNTERS (begin_time, MPI_TEST_COUNTER_EV, 0, TRUE);
		}
		Test_F_Software_Counter ++;
	}
}

void PMPI_Test_Wrapper (MPI_Fint *request, MPI_Fint *flag, MPI_Fint *status,
    MPI_Fint *ierror)
{
	MPI_Fint my_status[SIZEOF_MPI_STATUS], *ptr_status;

	ptr_status = (MPI_F_STATUS_IGNORE == status)?my_status:status;

	if (CURRENT_TRACE_MODE(THREADID) == TRACE_MODE_BURSTS)
		Bursts_PMPI_Test_Wrapper(request, flag, ptr_status, ierror);
	else
		Normal_PMPI_Test_Wrapper(request, flag, ptr_status, ierror);
}


/******************************************************************************
 ***  PMPI_TestAll_Wrapper
 ******************************************************************************/

void PMPI_TestAll_Wrapper (MPI_Fint * count, MPI_Fint array_of_requests[], MPI_Fint *flag,
	MPI_Fint array_of_statuses[][SIZEOF_MPI_STATUS], MPI_Fint * ierror)
{
	MPI_Fint my_statuses[MAX_WAIT_REQUESTS][SIZEOF_MPI_STATUS], *ptr_statuses;
	MPI_Request save_reqs[MAX_WAIT_REQUESTS];
	hash_data_t *hash_req = NULL;
	int src_world = -1, size = 0, tag = 0, ret, ireq;
	iotimer_t begin_time, end_time;
	int i;
	static int Test_F_Software_Counter = 0;
	static iotimer_t elapsed_time_outside_tests = 0, last_test_exit_time = 0;

	begin_time = LAST_READ_TIME;

        if (Test_F_Software_Counter == 0) {
                /* First MPI_Testall */
                elapsed_time_outside_tests = 0;
        }
        else {
                elapsed_time_outside_tests += (begin_time - last_test_exit_time);
        }

	if (*count > MAX_WAIT_REQUESTS)
		fprintf (stderr, "PANIC: too many requests in mpi_testtall\n");
	else
		for (i = 0; i < *count; i++)
			save_reqs[i] = MPI_Request_f2c(array_of_requests[i]);

	ptr_statuses = (MPI_F_STATUSES_IGNORE == (MPI_Fint*)array_of_statuses)?my_statuses:array_of_statuses;

	CtoF77 (pmpi_testall) (count, array_of_requests, flag, ptr_statuses, ierror);

	end_time = TIME;
	if (*ierror == MPI_SUCCESS && *flag)
	{
		TRACE_MPIEVENT (LAST_READ_TIME, MPI_TESTALL_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY,
                  EMPTY);

		if (Test_F_Software_Counter != 0)
		{
			TRACE_EVENT (begin_time, MPI_TIME_OUTSIDE_TESTS_EV, elapsed_time_outside_tests);
			TRACE_EVENT(begin_time, MPI_TEST_COUNTER_EV, Test_F_Software_Counter);
		}
		Test_F_Software_Counter = 0;

		for (ireq = 0; ireq < *count; ireq++)
		{
			if ((hash_req = hash_search (&requests, save_reqs[ireq])) != NULL)
			{
				int cancelled = 0;

				CtoF77 (pmpi_test_cancelled) (&ptr_statuses[ireq*SIZEOF_MPI_STATUS], &cancelled, ierror);
				if (!cancelled)
				{
					if ((ret = get_Irank_obj (hash_req, &src_world, &size, &tag, &ptr_statuses[ireq*SIZEOF_MPI_STATUS])) != MPI_SUCCESS)
					{
						*ierror = ret;
						return;
					}
					if (hash_req->group != MPI_GROUP_NULL)
					{
						MPI_Fint group = MPI_Group_c2f(hash_req->group);
						CtoF77 (pmpi_group_free) (&group, &ret);
						MPI_CHECK(ret, pmpi_group_free);
					}

					/* MPI Stats, get_Irank_obj above returns size (the number of bytes received) */
					updateStats_P2P(global_mpi_stats, src_world, size, 0);
				}
				TRACE_MPIEVENT_NOHWC (end_time, MPI_IRECVED_EV, cancelled, src_world, size, hash_req->tag, hash_req->commid, save_reqs[ireq]);
				hash_remove (&requests, save_reqs[ireq]);
			}
		}
		TRACE_MPIEVENT (end_time, MPI_TESTALL_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
	}
	else
	{
		if (Test_F_Software_Counter == 0)
		{
			/* First failed MPI_TestAll */
			TRACE_EVENTANDCOUNTERS (begin_time, MPI_TEST_COUNTER_EV, 0, TRUE);
		}
		Test_F_Software_Counter ++;
	}
}


/******************************************************************************
 ***  PMPI_TestAny_Wrapper
 ******************************************************************************/

void PMPI_TestAny_Wrapper (MPI_Fint *count, MPI_Fint array_of_requests[],
	MPI_Fint *index, MPI_Fint *flag, MPI_Fint *status, MPI_Fint *ierror)
{
	MPI_Fint my_status[SIZEOF_MPI_STATUS], *ptr_status;
	MPI_Request save_reqs[MAX_WAIT_REQUESTS];
	hash_data_t *hash_req = NULL;
	int src_world = -1, size = 0, tag = 0, ret, i;
	iotimer_t begin_time, end_time;
	static int Test_F_Software_Counter = 0;
	static iotimer_t elapsed_time_outside_tests = 0, last_test_exit_time = 0;

	begin_time = LAST_READ_TIME;

	if (Test_F_Software_Counter == 0) {
                /* First MPI_Testany */
                elapsed_time_outside_tests = 0;
        }
        else {
                elapsed_time_outside_tests += (begin_time - last_test_exit_time);
        }


	if (*count > MAX_WAIT_REQUESTS)
		fprintf (stderr, "PANIC: too many requests in mpi_testany\n");
	else
		for (i = 0; i < *count; i++)
			save_reqs[i] = MPI_Request_f2c(array_of_requests[i]);

	ptr_status = (MPI_F_STATUS_IGNORE == status)?my_status:status;

	CtoF77 (pmpi_testany) (count, array_of_requests, index, flag, ptr_status, ierror);

	end_time = TIME;

	if (*index != MPI_UNDEFINED && *ierror == MPI_SUCCESS && *flag)
	{
		TRACE_MPIEVENT (begin_time, MPI_TESTANY_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY,
                  EMPTY);

		if (Test_F_Software_Counter != 0)
		{
			TRACE_EVENT (begin_time, MPI_TIME_OUTSIDE_TESTS_EV, elapsed_time_outside_tests);
			TRACE_EVENT(begin_time, MPI_TEST_COUNTER_EV, Test_F_Software_Counter);
		}
		Test_F_Software_Counter = 0;

		MPI_Request req = save_reqs[*index-1];

		if ((hash_req = hash_search (&requests, req)) != NULL)
		{
			int cancelled = 0;

			CtoF77 (pmpi_test_cancelled) (ptr_status, &cancelled, ierror);
			if (!cancelled)
			{
				if ((ret = get_Irank_obj (hash_req, &src_world, &size, &tag, ptr_status)) != MPI_SUCCESS)
				{
					*ierror = ret;
					return;
				}
				if (hash_req->group != MPI_GROUP_NULL)
				{
					MPI_Fint group = MPI_Group_c2f(hash_req->group);
					CtoF77 (pmpi_group_free) (&group, &ret);
					MPI_CHECK(ret, pmpi_group_free);
				}

				/* MPI Stats, get_Irank_obj above returns size (the number of bytes received) */
				updateStats_P2P(global_mpi_stats, src_world, size, 0);
			}
			TRACE_MPIEVENT_NOHWC (end_time, MPI_IRECVED_EV, cancelled, src_world, size, hash_req->tag, hash_req->commid, req);
			hash_remove (&requests, req);
		}
		TRACE_MPIEVENT (end_time, MPI_TESTANY_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
	}
	else
	{
		if (Test_F_Software_Counter == 0)
		{
			/* First failed MPI_Testany */
			TRACE_EVENTANDCOUNTERS (begin_time, MPI_TEST_COUNTER_EV, 0, TRUE);
		}
		Test_F_Software_Counter ++;
	}
}

/*****************************************************************************
 ***  PMPI_TestSome_Wrapper
 ******************************************************************************/

void PMPI_TestSome_Wrapper (MPI_Fint *incount, MPI_Fint array_of_requests[],
	MPI_Fint *outcount, MPI_Fint array_of_indices[],
	MPI_Fint array_of_statuses[][SIZEOF_MPI_STATUS], MPI_Fint *ierror)
{
	MPI_Fint my_statuses[MAX_WAIT_REQUESTS][SIZEOF_MPI_STATUS], *ptr_statuses;
	MPI_Request save_reqs[MAX_WAIT_REQUESTS];
	hash_data_t *hash_req = NULL;
	int src_world = -1, size = 0, tag = 0, ret, i;
	iotimer_t begin_time, end_time;
	static int Test_F_Software_Counter = 0;
	static iotimer_t elapsed_time_outside_tests = 0, last_test_exit_time = 0;

        begin_time = LAST_READ_TIME;

        if (Test_F_Software_Counter == 0) {
                /* First MPI_Testsome */
                elapsed_time_outside_tests = 0;
        }
        else {
                elapsed_time_outside_tests += (begin_time - last_test_exit_time);
        }


	if (*incount > MAX_WAIT_REQUESTS)
		fprintf (stderr, "PANIC: too many requests in mpi_testsome\n");
	else
		for (i = 0; i < *incount; i++)
			save_reqs[i] = MPI_Request_f2c(array_of_requests[i]);

	ptr_statuses = (MPI_F_STATUSES_IGNORE == (MPI_Fint*) array_of_statuses)?my_statuses:array_of_statuses;

	CtoF77(pmpi_testsome) (incount, array_of_requests, outcount, array_of_indices,
	  ptr_statuses, ierror);

	end_time = TIME;

	if (*ierror == MPI_SUCCESS && *outcount > 0)
	{
		TRACE_MPIEVENT (begin_time, MPI_TESTSOME_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY,
                  EMPTY);

		if (Test_F_Software_Counter != 0)
		{
			TRACE_EVENT (begin_time, MPI_TIME_OUTSIDE_TESTS_EV, elapsed_time_outside_tests);
			TRACE_EVENT(begin_time, MPI_TEST_COUNTER_EV, Test_F_Software_Counter);
		}
		Test_F_Software_Counter = 0;

		for (i = 0; i < *outcount; i++)
		{
			MPI_Request req = save_reqs[array_of_indices[i]];
			if ((hash_req = hash_search (&requests, req)) != NULL)
			{
				int cancelled = 0;

				CtoF77(pmpi_test_cancelled) (&ptr_statuses[i*SIZEOF_MPI_STATUS], &cancelled, ierror);
				if (!cancelled)
				{
					if ((ret = get_Irank_obj (hash_req, &src_world, &size, &tag, &ptr_statuses[i*SIZEOF_MPI_STATUS])) != MPI_SUCCESS)
					{
						*ierror = ret;
						return;
					}
					if (hash_req->group != MPI_GROUP_NULL)
					{
						MPI_Fint group = MPI_Group_c2f(hash_req->group);
						CtoF77 (pmpi_group_free) (&group, &ret);
						MPI_CHECK(ret, pmpi_group_free);
					}

					/* MPI Stats. get_Irank_obj above returns size (the number of bytes received) */
					updateStats_P2P(global_mpi_stats, src_world, size, 0);
				}
				TRACE_MPIEVENT_NOHWC (end_time, MPI_IRECVED_EV, cancelled, src_world, size, hash_req->tag, hash_req->commid, req);
				hash_remove (&requests, req);
			}
		}
		TRACE_MPIEVENT (end_time, MPI_TESTSOME_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
	}
	else
	{
		if (Test_F_Software_Counter == 0)
		{
			/* First failed MPI_Testsome */
			TRACE_EVENTANDCOUNTERS (begin_time, MPI_TEST_COUNTER_EV, 0, TRUE);
		}
		Test_F_Software_Counter ++;
	}
}


/******************************************************************************
 ***  PMPI_Wait_Wrapper
 ******************************************************************************/

void PMPI_Wait_Wrapper (MPI_Fint *request, MPI_Fint *status, MPI_Fint *ierror)
{
	MPI_Fint my_status[SIZEOF_MPI_STATUS], *ptr_status;
	hash_data_t *hash_req = NULL;
	iotimer_t temps_final;
	int src_world = -1, size = 0, tag = 0, ret;
	MPI_Request req = MPI_Request_f2c(*request);

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_WAIT_EV, EVT_BEGIN, req, EMPTY, EMPTY, EMPTY, EMPTY);

	ptr_status = (MPI_F_STATUS_IGNORE == status)?my_status:status;

	CtoF77 (pmpi_wait) (request, ptr_status, ierror);

	temps_final = TIME;

	if (*ierror == MPI_SUCCESS && ((hash_req = hash_search (&requests, req)) != NULL))
	{
		int cancelled = 0;

		CtoF77(pmpi_test_cancelled) (ptr_status, &cancelled, ierror);
		if (!cancelled)
		{
			if ((ret = get_Irank_obj (hash_req, &src_world, &size, &tag, ptr_status)) != MPI_SUCCESS)
			{
				*ierror = ret;
				return;
			}
			if (hash_req->group != MPI_GROUP_NULL)
			{
				MPI_Fint group = MPI_Group_c2f (hash_req->group);
				CtoF77 (pmpi_group_free) (&group, &ret);
				MPI_CHECK (ret, pmpi_group_free);
			}

			/* MPI Stats get_Irank_obj above returns size (the number of bytes received) */
			updateStats_P2P(global_mpi_stats, src_world, size, 0);
		}
		TRACE_MPIEVENT_NOHWC (temps_final, MPI_IRECVED_EV, cancelled, src_world, size, hash_req->tag, hash_req->commid, req); /* NOHWC */
		hash_remove (&requests, req);
	}

	TRACE_MPIEVENT (temps_final, MPI_WAIT_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
}

/******************************************************************************
 ***  PMPI_WaitAll_Wrapper
 ******************************************************************************/

void PMPI_WaitAll_Wrapper (MPI_Fint * count, MPI_Fint array_of_requests[],
	MPI_Fint array_of_statuses[][SIZEOF_MPI_STATUS], MPI_Fint * ierror)
{
	MPI_Fint my_statuses[MAX_WAIT_REQUESTS][SIZEOF_MPI_STATUS], *ptr_statuses;
	MPI_Request save_reqs[MAX_WAIT_REQUESTS];
	hash_data_t *hash_req = NULL;
	int src_world = -1, size = 0, tag = 0, ret, ireq;
	iotimer_t temps_final;
	int i;

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_WAITALL_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY,
	  EMPTY);

	if (*count > MAX_WAIT_REQUESTS)
		fprintf (stderr, "PANIC: too many requests in mpi_waitall\n");
	else
		for (i = 0; i < *count; i++)
			save_reqs[i] = MPI_Request_f2c(array_of_requests[i]);

	ptr_statuses = (MPI_F_STATUSES_IGNORE == (MPI_Fint*)array_of_statuses)?my_statuses:array_of_statuses;

	CtoF77 (pmpi_waitall) (count, array_of_requests, ptr_statuses, ierror);

	temps_final = TIME;
	if (*ierror == MPI_SUCCESS)
	{
		for (ireq = 0; ireq < *count; ireq++)
		{
			if ((hash_req = hash_search (&requests, save_reqs[ireq])) != NULL)
			{
				int cancelled = 0;

				CtoF77(pmpi_test_cancelled) (&ptr_statuses[ireq*SIZEOF_MPI_STATUS], &cancelled, ierror);
				if (!cancelled)
				{
					if ((ret = get_Irank_obj (hash_req, &src_world, &size, &tag, &ptr_statuses[ireq*SIZEOF_MPI_STATUS])) != MPI_SUCCESS)
					{
						*ierror = ret;
						return;
					}
					if (hash_req->group != MPI_GROUP_NULL)
					{
						MPI_Fint group = MPI_Group_c2f(hash_req->group);
						CtoF77 (pmpi_group_free) (&group, &ret);
						MPI_CHECK(ret, pmpi_group_free);
					}

					/* MPI Stats, get_Irank_obj above returns size (the number of bytes received) */
					updateStats_P2P(global_mpi_stats, src_world, size, 0);
				}
				TRACE_MPIEVENT_NOHWC (temps_final, MPI_IRECVED_EV, cancelled, src_world, size, hash_req->tag, hash_req->commid, save_reqs[ireq]);
				hash_remove (&requests, save_reqs[ireq]);
			}
		}
	}
	TRACE_MPIEVENT (temps_final, MPI_WAITALL_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
}


/******************************************************************************
 ***  PMPI_WaitAny_Wrapper
 ******************************************************************************/

void PMPI_WaitAny_Wrapper (MPI_Fint *count, MPI_Fint array_of_requests[],
	MPI_Fint *index, MPI_Fint *status, MPI_Fint *ierror)
{
	MPI_Fint my_status[SIZEOF_MPI_STATUS], *ptr_status;
	MPI_Request save_reqs[MAX_WAIT_REQUESTS];
	hash_data_t *hash_req = NULL;
	int src_world = -1, size = 0, tag = 0, ret, i;
	iotimer_t temps_final;

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_WAITANY_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY,
	  EMPTY);

	if (*count > MAX_WAIT_REQUESTS)
		fprintf (stderr, "PANIC: too many requests in mpi_waitany\n");
	else
		for (i = 0; i < *count; i++)
			save_reqs[i] = MPI_Request_f2c(array_of_requests[i]);

	ptr_status = (MPI_F_STATUS_IGNORE == status)?my_status:status;

	CtoF77 (pmpi_waitany) (count, array_of_requests, index, ptr_status, ierror);

	temps_final = TIME;

	if (*index != MPI_UNDEFINED && *ierror == MPI_SUCCESS)
	{
		MPI_Request req = save_reqs[*index-1];

		if ((hash_req = hash_search (&requests, req)) != NULL)
		{
			int cancelled = 0;

			CtoF77(pmpi_test_cancelled) (ptr_status, &cancelled, ierror);
			if (!cancelled)
			{
				if ((ret = get_Irank_obj (hash_req, &src_world, &size, &tag, ptr_status)) != MPI_SUCCESS)
				{
					*ierror = ret;
					return;
				}
				if (hash_req->group != MPI_GROUP_NULL)
				{
					MPI_Fint group = MPI_Group_c2f(hash_req->group);
					CtoF77 (pmpi_group_free) (&group, &ret);
					MPI_CHECK(ret, pmpi_group_free);
				}

				/* MPI Stats, get_Irank_obj above returns size (the number of bytes received) */
				updateStats_P2P(global_mpi_stats, src_world, size, 0);
			}
			TRACE_MPIEVENT_NOHWC (temps_final, MPI_IRECVED_EV, cancelled, src_world, size, hash_req->tag, hash_req->commid, req);
			hash_remove (&requests, req);
		}
	}
	TRACE_MPIEVENT (temps_final, MPI_WAITANY_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
}

/*****************************************************************************
 ***  PMPI_WaitSome_Wrapper
 ******************************************************************************/

void PMPI_WaitSome_Wrapper (MPI_Fint *incount, MPI_Fint array_of_requests[],
	MPI_Fint *outcount, MPI_Fint array_of_indices[],
	MPI_Fint array_of_statuses[][SIZEOF_MPI_STATUS], MPI_Fint *ierror)
{
	MPI_Fint my_statuses[MAX_WAIT_REQUESTS][SIZEOF_MPI_STATUS], *ptr_statuses;
	MPI_Request save_reqs[MAX_WAIT_REQUESTS];
	hash_data_t *hash_req = NULL;
	int src_world = -1, size = 0, tag = 0, ret, i;
	iotimer_t temps_final;

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_WAITSOME_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY,
	  EMPTY);

	if (*incount > MAX_WAIT_REQUESTS)
		fprintf (stderr, "PANIC: too many requests in mpi_waitsome\n");
	else
		for (i = 0; i < *incount; i++)
			save_reqs[i] = MPI_Request_f2c(array_of_requests[i]);

	ptr_statuses = (MPI_F_STATUSES_IGNORE == (MPI_Fint*) array_of_statuses)?my_statuses:array_of_statuses;

	CtoF77(pmpi_waitsome) (incount, array_of_requests, outcount, array_of_indices,
	  ptr_statuses, ierror);

	temps_final = TIME;

	if (*ierror == MPI_SUCCESS)
	{
		for (i = 0; i < *outcount; i++)
		{
			MPI_Request req = save_reqs[array_of_indices[i]];
			if ((hash_req = hash_search (&requests, req)) != NULL)
			{
				int cancelled = 0;
				
				CtoF77(pmpi_test_cancelled) (&ptr_statuses[i*SIZEOF_MPI_STATUS], &cancelled, ierror);
				if (!cancelled)
				{
					if ((ret = get_Irank_obj (hash_req, &src_world, &size, &tag, &ptr_statuses[i*SIZEOF_MPI_STATUS])) != MPI_SUCCESS)
					{
						*ierror = ret;
						return;
					}
					if (hash_req->group != MPI_GROUP_NULL)
					{
						MPI_Fint group = MPI_Group_c2f(hash_req->group);
						CtoF77 (pmpi_group_free) (&group, &ret);
						MPI_CHECK(ret, pmpi_group_free);
					}

					/* MPI Stats, get_Irank_obj above returns size (the number of bytes received) */
					updateStats_P2P(global_mpi_stats, src_world, size, 0);
				}
				TRACE_MPIEVENT_NOHWC (temps_final, MPI_IRECVED_EV, cancelled, src_world, size, hash_req->tag, hash_req->commid, req);
				hash_remove (&requests, req);
			}
		}
	}
	TRACE_MPIEVENT (temps_final, MPI_WAITSOME_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
}

/******************************************************************************
 ***  PMPI_Recv_init_Wrapper
 ******************************************************************************/

void PMPI_Recv_init_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)
{
	MPI_Request req;
	MPI_Comm c = MPI_Comm_f2c (*comm);
	MPI_Datatype type = MPI_Type_f2c (*datatype);

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
  CtoF77 (pmpi_recv_init) (buf, count, datatype, source, tag,
                           comm, request, ierror);

  /*
   * Es guarda aquesta request 
   */
	req = MPI_Request_f2c (*request);
	PR_NewRequest (MPI_IRECV_EV, *count, type, *source, *tag, c, req, &PR_queue);

  /*
   *   type : RECV_INIT_EV                 value : EVT_END
   *   target : ---                        size  : ----
   *   tag : ---                           comm : ---
   *   aux : ---
   */
  TRACE_MPIEVENT (TIME, MPI_RECV_INIT_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY,
                  EMPTY);

	updateStats_OTHER(global_mpi_stats);
}



/******************************************************************************
 ***  PMPI_Send_init_Wrapper
 ******************************************************************************/

void PMPI_Send_init_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)
{
	MPI_Request req;
	MPI_Datatype type = MPI_Type_f2c(*datatype);
	MPI_Comm c = MPI_Comm_f2c(*comm);

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
  CtoF77 (pmpi_send_init) (buf, count, datatype, dest, tag,
                           comm, request, ierror);

  /*
   * Es guarda aquesta request 
   */
	req = MPI_Request_f2c (*request);
	PR_NewRequest (MPI_ISEND_EV, *count, type, *dest, *tag, c, req, &PR_queue);

  /*
   *   type : SEND_INIT_EV                 value : EVT_END
   *   target : ---                        size  : ----
   *   tag : ---                           comm : ---
   *   aux : ---
   */

  TRACE_MPIEVENT (TIME, MPI_SEND_INIT_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY,
                  EMPTY);

	updateStats_OTHER(global_mpi_stats);
}



/******************************************************************************
 ***  PMPI_Bsend_init_Wrapper
 ******************************************************************************/

void PMPI_Bsend_init_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)
{
	MPI_Request req;
	MPI_Datatype type = MPI_Type_f2c(*datatype);
	MPI_Comm c = MPI_Comm_f2c(*comm);

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
  CtoF77 (pmpi_bsend_init) (buf, count, datatype, dest, tag,
                            comm, request, ierror);

  /*
   * Es guarda aquesta request 
   */
	req = MPI_Request_f2c (*request);
	PR_NewRequest (MPI_IBSEND_EV, *count, type, *dest, *tag, c, req, &PR_queue);

  /*
   *   type : BSEND_INIT_EV                value : EVT_END
   *   target : ---                        size  : ----
   *   tag : ---                           comm : ---
   *   aux : ---
   */
  TRACE_MPIEVENT (TIME, MPI_BSEND_INIT_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY,
                  EMPTY);

	updateStats_OTHER(global_mpi_stats);
}


/******************************************************************************
 ***  PMPI_Rsend_init_Wrapper
 ******************************************************************************/

void PMPI_Rsend_init_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)
{
	MPI_Request req;
	MPI_Datatype type = MPI_Type_f2c(*datatype);
	MPI_Comm c = MPI_Comm_f2c(*comm);

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
  CtoF77 (pmpi_rsend_init) (buf, count, datatype, dest, tag,
                            comm, request, ierror);

  /*
   * Es guarda aquesta request 
   */
	req = MPI_Request_f2c (*request);
	PR_NewRequest (MPI_IRSEND_EV, *count, type, *dest, *tag, c, req, &PR_queue);

  /*
   *   type : RSEND_INIT_EV                value : EVT_END
   *   target : ---                        size  : ----
   *   tag : ---                           comm : ---
   *   aux : ---
   */
  TRACE_MPIEVENT (TIME, MPI_RSEND_INIT_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY,
                  EMPTY);

	updateStats_OTHER(global_mpi_stats);
}


/******************************************************************************
 ***  PMPI_Ssend_init_Wrapper
 ******************************************************************************/

void PMPI_Ssend_init_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)
{
	MPI_Request req;
	MPI_Datatype type = MPI_Type_f2c(*datatype);
	MPI_Comm c = MPI_Comm_f2c(*comm);

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
  CtoF77 (pmpi_ssend_init) (buf, count, datatype, dest, tag,
                            comm, request, ierror);

  /*
   * Es guarda aquesta request 
   */
	req = MPI_Request_f2c (*request);
	PR_NewRequest (MPI_ISSEND_EV, *count, type, *dest, *tag, c, req, &PR_queue);

  /*
   *   type : SSEND_INIT_EV                value : EVT_END
   *   target : ---                        size  : ----
   *   tag : ---                           comm : ---
   *   aux : ---
   */
  TRACE_MPIEVENT (TIME, MPI_SSEND_INIT_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY,
                  EMPTY);

	updateStats_OTHER(global_mpi_stats);
}

void MPI_Sendrecv_Fortran_Wrapper (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, MPI_Fint *dest, MPI_Fint *sendtag, void *recvbuf,
	MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *source, MPI_Fint *recvtag,
	MPI_Fint *comm, MPI_Fint *status, MPI_Fint *ierr) 
{
	MPI_Fint my_status[SIZEOF_MPI_STATUS], *ptr_status;
	MPI_Comm c = MPI_Comm_f2c (*comm);
	int DataSendSize, DataRecvSize, DataSend, DataSize, ret;
	int sender_src, SourceRank, RecvRank, Count, sender_tag;

	if ((ret = get_rank_obj (comm, dest, &RecvRank, RANK_OBJ_SEND)) != MPI_SUCCESS)
		return; 

	if (*sendcount != 0)
	{
		CtoF77(pmpi_type_size) (sendtype, &DataSendSize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		DataSendSize = 0;

	if (*recvcount != 0)
	{
		CtoF77(pmpi_type_size) (recvtype, &DataRecvSize, &ret);
		MPI_CHECK(ret, pmpi_type_size);
	}
	else
		DataRecvSize = 0;

	DataSend = *sendcount * DataSendSize;

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_SENDRECV_EV, EVT_BEGIN, RecvRank, DataSend, *sendtag, c, EMPTY);

	ptr_status = (MPI_F_STATUS_IGNORE == status)?my_status:status;

	CtoF77(pmpi_sendrecv) (sendbuf, sendcount, sendtype, dest, sendtag,
	  recvbuf, recvcount, recvtype, source, recvtag, comm, ptr_status, ierr);

	CtoF77(pmpi_get_count) (ptr_status, recvtype, &Count, &ret);
	MPI_CHECK(ret, pmpi_get_count);

	if (Count != MPI_UNDEFINED)
		DataSize = DataRecvSize * Count;
	else
		DataSize = 0;

	if (*source == MPI_ANY_SOURCE)
		sender_src = ptr_status[MPI_SOURCE_OFFSET];
	else
		sender_src = *source;

	if (*recvtag == MPI_ANY_TAG)
		sender_tag = ptr_status[MPI_TAG_OFFSET];
	else
		sender_tag = *recvtag;

	if ((ret = get_rank_obj (comm, &sender_src, &SourceRank, RANK_OBJ_RECV)) != MPI_SUCCESS)
		return; 

	TRACE_MPIEVENT (TIME, MPI_SENDRECV_EV, EVT_END, SourceRank, DataSize, sender_tag, c, EMPTY);

	/* MPI Stats */
    updateStats_P2P(global_mpi_stats, RecvRank, 0, DataSend);
    updateStats_P2P(global_mpi_stats, SourceRank, DataSize, 0);
}

void MPI_Sendrecv_replace_Fortran_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *type,
	MPI_Fint *dest, MPI_Fint *sendtag, MPI_Fint *source, MPI_Fint *recvtag,
	MPI_Fint *comm, MPI_Fint *status, MPI_Fint *ierr) 
{
	MPI_Fint my_status[SIZEOF_MPI_STATUS], *ptr_status;
	MPI_Comm c = MPI_Comm_f2c (*comm);
	int DataSendSize, DataRecvSize, DataSend, DataSize, ret;
	int sender_src, SourceRank, RecvRank, Count, sender_tag;

	if ((ret = get_rank_obj (comm, dest, &RecvRank, RANK_OBJ_SEND)) != MPI_SUCCESS)
		return;

	if (*count != 0)
	{
		CtoF77(pmpi_type_size) (type, &DataSendSize, &ret);
		DataRecvSize = DataSendSize;
	}
	else
		DataRecvSize = DataSendSize = 0;

	DataSend = *count * DataSendSize;

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_SENDRECV_REPLACE_EV, EVT_BEGIN, RecvRank, DataSend, *sendtag, c, EMPTY);

	ptr_status = (MPI_F_STATUS_IGNORE == status)?my_status:status;

	CtoF77(pmpi_sendrecv_replace) (buf, count, type, dest, sendtag, source, recvtag, comm, ptr_status, ierr);

	CtoF77(pmpi_get_count) (ptr_status, type, &Count, &ret);
	MPI_CHECK(ret, pmpi_get_count);

	if (Count != MPI_UNDEFINED)
		DataSize = DataRecvSize * Count;
	else
		DataSize = 0;

	if (*source == MPI_ANY_SOURCE)
		sender_src = ptr_status[MPI_SOURCE_OFFSET];
	else
		sender_src = *source;

	if (*recvtag == MPI_ANY_TAG)
		sender_tag = ptr_status[MPI_TAG_OFFSET];
	else
		sender_tag = *recvtag;

	if ((ret = get_rank_obj (comm, &sender_src, &SourceRank, RANK_OBJ_RECV)) != MPI_SUCCESS)
		return;

	TRACE_MPIEVENT (TIME, MPI_SENDRECV_REPLACE_EV, EVT_END, SourceRank, DataSize, sender_tag, c, EMPTY);

	/* MPI Stats */
    updateStats_P2P(global_mpi_stats, RecvRank, 0, DataSend);
    updateStats_P2P(global_mpi_stats, SourceRank, DataSize, 0);
}

#endif /* defined(FORTRAN_SYMBOLS) */

