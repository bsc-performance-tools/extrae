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


/******************************************************************************
 ***  MPI_Bsend_C_Wrapper
 ******************************************************************************/

int MPI_Bsend_C_Wrapper (void *buf, int count, MPI_Datatype datatype, int dest,
                         int tag, MPI_Comm comm)
{
	int size, receiver, ret;

	size = getMsgSizeFromCountAndDatatype (count, datatype);

	translateLocalToGlobalRank (comm, MPI_GROUP_NULL, dest, &receiver, OP_TYPE_SEND);

	/*
	 *   event  : BSEND_EV                     value  : EVT_BEGIN
	 *   target : receiver rank                size   : bytes sent
	 *   tag    : message tag                  commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_BSEND_EV, EVT_BEGIN, receiver, size, tag, comm, EMPTY);

	ret = PMPI_Bsend (buf, count, datatype, dest, tag, comm);

	/*
	 *   event  : BSEND_EV                     value  : EVT_END
	 *   target : receiver rank                size   : bytes sent
	 *   tag    : message tag                  commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_BSEND_EV, EVT_END, receiver, size, tag, comm, EMPTY);

	// MPI stats 
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

	size = getMsgSizeFromCountAndDatatype (count, datatype);        

	translateLocalToGlobalRank (comm, MPI_GROUP_NULL, dest, &receiver, OP_TYPE_SEND);

	/*
	 *   event  : SSEND_EV                     value  : EVT_BEGIN
	 *   target : receiver rank                size   : bytes sent
	 *   tag    : message tag                  commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_SSEND_EV, EVT_BEGIN, receiver, size, tag, comm, EMPTY);

	ret = PMPI_Ssend (buf, count, datatype, dest, tag, comm);

	/*
	 *   event  : SSEND_EV                     value  : EVT_END
	 *   target : receiver rank                size   : bytes sent
	 *   tag    : message tag                  commid : communicator id
	 *   aux    : ---  
	 */
	TRACE_MPIEVENT (TIME, MPI_SSEND_EV, EVT_END, receiver, size, tag, comm, EMPTY);

	// MPI stats
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

	size = getMsgSizeFromCountAndDatatype (count, datatype);

	translateLocalToGlobalRank (comm, MPI_GROUP_NULL, dest, &receiver, OP_TYPE_SEND);

	/*
	 *   event  : RSEND_EV                     value  : EVT_BEGIN
	 *   target : receiver rank                size   : bytes sent
	 *   tag    : message tag                  commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_RSEND_EV, EVT_BEGIN, receiver, size, tag, comm, EMPTY);

	ret = PMPI_Rsend (buf, count, datatype, dest, tag, comm);

	/*
	 *   event  : RSEND_EV                     value  : EVT_END
	 *   target : receiver rank                size   : bytes sent
	 *   tag    : message tag                  commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_RSEND_EV, EVT_END, receiver, size, tag, comm, EMPTY);

	// MPI stats
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

	size = getMsgSizeFromCountAndDatatype (count, datatype);

	translateLocalToGlobalRank (comm, MPI_GROUP_NULL, dest, &receiver, OP_TYPE_SEND);

	/*
	 *   event  : SEND_EV                      value  : EVT_BEGIN
	 *   target : receiver rank                size   : bytes sent
	 *   tag    : message tag                  commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_SEND_EV, EVT_BEGIN, receiver, size, tag, comm, EMPTY);

	ret = PMPI_Send (buf, count, datatype, dest, tag, comm);
  
	/*
	 *   event  : SEND_EV                      value  : EVT_END
	 *   target : receiver rank                size   : bytes sent
	 *   tag    : message tag                  commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_SEND_EV, EVT_END, receiver, size, tag, comm, EMPTY);

	// MPI stats
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

	size = getMsgSizeFromCountAndDatatype (count, datatype);

	translateLocalToGlobalRank (comm, MPI_GROUP_NULL, dest, &receiver, OP_TYPE_SEND);

	/*
	 *   event  : IBSEND_EV                    value  : EVT_BEGIN
	 *   target : receiver rank                size   : bytes sent 
	 *   tag    : message tag                  commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IBSEND_EV, EVT_BEGIN, receiver, size, tag, comm, EMPTY);

	ret = PMPI_Ibsend (buf, count, datatype, dest, tag, comm, request);

	/*
	 *   event  : IBSEND_EV                    value  : EVT_END
	 *   target : receiver rank                size   : bytes sent
	 *   tag    : message tag                  commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_IBSEND_EV, EVT_END, receiver, size, tag, comm, EMPTY);

	// MPI stats
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

	size = getMsgSizeFromCountAndDatatype (count, datatype);

	translateLocalToGlobalRank (comm, MPI_GROUP_NULL, dest, &receiver, OP_TYPE_SEND);

	/*
	 *   event  : ISEND_EV                     value  : EVT_BEGIN
	 *   target : receiver rank                size   : bytes sent
	 *   tag    : message tag                  commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_ISEND_EV, EVT_BEGIN, receiver, size, tag, comm, EMPTY);

	ret = PMPI_Isend (buf, count, datatype, dest, tag, comm, request);

	/*
	 *   event  : ISEND_EV                     value  : EVT_END
	 *   target : receiver rank                size   : bytes sent
	 *   tag    : message tag                  commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_ISEND_EV, EVT_END, receiver, size, tag, comm, EMPTY);

	/* MPI stats */
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

	size = getMsgSizeFromCountAndDatatype (count, datatype);

	translateLocalToGlobalRank (comm, MPI_GROUP_NULL, dest, &receiver, OP_TYPE_SEND);

	/*
	 *   event  : ISSEND_EV                    value  : EVT_BEGIN
	 *   target : receiver rank                size   : bytes sent
	 *   tag    : message tag                  commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_ISSEND_EV, EVT_BEGIN, receiver, size, tag, comm, EMPTY);

	ret = PMPI_Issend (buf, count, datatype, dest, tag, comm, request);

	/*
	 *   event  : ISSEND_EV                    value  : EVT_END
	 *   target : receiver rank                size   : bytes sent
	 *   tag    : message tag                  commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_ISSEND_EV, EVT_END, receiver, size, tag, comm, EMPTY);

	// MPI stats
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

	size = getMsgSizeFromCountAndDatatype (count, datatype);

	translateLocalToGlobalRank (comm, MPI_GROUP_NULL, dest, &receiver, OP_TYPE_SEND);

	/*
	 *   event  : IRSEND_EV                    value  : EVT_BEGIN
	 *   target : receiver rank                size   : bytes sent
	 *   tag    : message tag                  commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IRSEND_EV, EVT_BEGIN, receiver, size, tag, comm, EMPTY);

	ret = PMPI_Irsend (buf, count, datatype, dest, tag, comm, request);

	/*
	 *   event  : IRSEND_EV                    value  : EVT_END
	 *   target : receiver rank                size   : bytes sent
	 *   tag    : message tag                  commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_IRSEND_EV, EVT_END, receiver, size, tag, comm, EMPTY);

	// MPI stats
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
	int size, src_world, sender_tag, ierror;

	size = getMsgSizeFromCountAndDatatype (count, datatype);

	translateLocalToGlobalRank (comm, MPI_GROUP_NULL, source, &src_world, OP_TYPE_RECV);

	/*
	 *   event  : RECV_EV                      value  : EVT_BEGIN    
	 *   target : MPI_ANY_SOURCE or sender     size   : receive buffer size    
	 *   tag    : message tag or MPI_ANY_TAG   commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_RECV_EV, EVT_BEGIN, src_world, size, tag, comm, EMPTY);

	ptr_status = (MPI_STATUS_IGNORE == status) ? &my_status : status; 
 
	ierror = PMPI_Recv (buf, count, datatype, source, tag, comm, ptr_status);

	getCommDataFromStatus (ptr_status, datatype, comm, MPI_GROUP_NULL, &size, &sender_tag, &src_world);

	/*
	 *   event  : RECV_EV                      value  : EVT_END
	 *   target : sender rank                  size   : received message size    
	 *   tag    : received message tag         commid : communicator id 
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_RECV_EV, EVT_END, src_world, size, sender_tag, comm, EMPTY);

	// MPI stats 
	updateStats_P2P(global_mpi_stats, src_world, size, 0);

	return ierror;
}



/******************************************************************************
 ***  MPI_Irecv_C_Wrapper
 ******************************************************************************/

int MPI_Irecv_C_Wrapper (void *buf, int count, MPI_Datatype datatype,
	int source, int tag, MPI_Comm comm, MPI_Request *request)
{
	int ierror, size, src_world;

	size = getMsgSizeFromCountAndDatatype (count, datatype);

	translateLocalToGlobalRank (comm, MPI_GROUP_NULL, source, &src_world, OP_TYPE_RECV);

	/*
	 *   event  : IRECV_EV                     value  : EVT_BEGIN
	 *   target : MPI_ANY_SOURCE or sender     size   : receive buffer size
	 *   tag    : message tag or MPI_ANY_TAG   commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IRECV_EV, EVT_BEGIN, src_world, size, tag, comm, EMPTY);

	ierror = PMPI_Irecv (buf, count, datatype, source, tag, comm, request);

	SaveRequest(*request, comm);

	/*
	 *   event  : IRECV_EV                     value  : EVT_END
	 *   target : sender rank                  size   : receive buffer size
	 *   tag    : received message tag         commid : communicator id
	 *   aux    : request id
	 */
	TRACE_MPIEVENT (TIME, MPI_IRECV_EV, EVT_END, src_world, size, tag, comm, *request);

	return ierror;
}


#if defined(MPI3)

/******************************************************************************
 ***  MPI_Mrecv_C_Wrapper
 ******************************************************************************/

int MPI_Mrecv_C_Wrapper (void *buf, int count, MPI_Datatype datatype, 
                         MPI_Message *message, MPI_Status *status)
{
	MPI_Status  my_status, *ptr_status;
	MPI_Comm    comm;
	int         size, src_world, sender_tag, ierror;
	MPI_Message save_message = *message; // Save input value as it changes inside the PMPI

	size = getMsgSizeFromCountAndDatatype (count, datatype);

	/*
	 *   event  : MRECV_EV                          value  : EVT_BEGIN    
	 *   target : --- (source not avail)            size   : receive buffer size    
	 *   tag    : --- (tag not avail)               commid : --- (comm not avail)
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_MRECV_EV, EVT_BEGIN, EMPTY, size, EMPTY, EMPTY, EMPTY);

	ptr_status = (MPI_STATUS_IGNORE == status) ? &my_status : status; 
 
	ierror = PMPI_Mrecv (buf, count, datatype, message, ptr_status);

	comm = ProcessMessage (save_message, NULL);

	getCommDataFromStatus (ptr_status, datatype, comm, MPI_GROUP_NULL, &size, &sender_tag, &src_world);

	/*
	 *   event  : MRECV_EV                          value  : EVT_END
	 *   target : sender rank                       size   : received message size
	 *   tag    : message tag                       commid : communicator id (hashed)
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_MRECV_EV, EVT_END, src_world, size, sender_tag, comm, EMPTY);

	// MPI stats 
	updateStats_P2P(global_mpi_stats, src_world, size, 0);

	return ierror;
}


/******************************************************************************
 ***  MPI_Imrecv_C_Wrapper
 ******************************************************************************/

int MPI_Imrecv_C_Wrapper (void *buf, int count, MPI_Datatype datatype,
                          MPI_Message *message, MPI_Request *request)
{
	MPI_Comm    comm;
	int         ierror, size;
	MPI_Message save_message = *message; // Save input value as it changes inside the PMPI

	size = getMsgSizeFromCountAndDatatype (count, datatype);

	/*
	 *   event  : IMRECV_EV                  value  : EVT_BEGIN
	 *   target : --- (source not avail)     size   : receive buffer size
	 *   tag    : --- (tag not avail)        commid : --- (comm not avail)
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IRECV_EV, EVT_BEGIN, EMPTY, size, EMPTY, EMPTY, EMPTY);

	ierror = PMPI_Imrecv (buf, count, datatype, message, request);

	comm = ProcessMessage (save_message, request);

	/*
	 *   event  : IMRECV_EV                  value  : EVT_END
	 *   target : --- (source not avail)     size   : receive buffer size
	 *   tag    : --- (tag not avail)        commid : communicator id (hashed)
	 *   aux    : request id
	 */
	TRACE_MPIEVENT (TIME, MPI_IMRECV_EV, EVT_END, EMPTY, size, EMPTY, comm, *request);

	return ierror;
}

#endif /* MPI3 */


/******************************************************************************
 ***  MPI_Probe_C_Wrapper
 ******************************************************************************/

int MPI_Probe_C_Wrapper (int source, int tag, MPI_Comm comm, MPI_Status *status)
{
	int ierror;

	/*
	 *   event  : PROBE_EV                     value  : EVT_BEGIN
	 *   target : ---                          size   : ---
	 *   tag    : ---                          commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_PROBE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, comm, EMPTY);

	ierror = PMPI_Probe (source, tag, comm, status);

	/*
	 *   event  : PROBE_EV                     value  : EVT_END
	 *   target : ---                          size   : ---
	 *   tag    : ---                          commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_PROBE_EV, EVT_END, EMPTY, EMPTY, EMPTY, comm, EMPTY);

	// MPI stats 
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
	 *   event  : IPROBE_EV                    value  : EVT_BEGIN
	 *   target : ---                          size   : ---
	 *   tag    : ---                          commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IPROBE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, comm, EMPTY);

	ierror = PMPI_Iprobe (source, tag, comm, flag, status);

	/*
	 *   event  : IPROBE_EV                    value  : EVT_END
	 *   target : ---                          size   : ---
	 *   tag    : ---                          commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_IPROBE_EV, EVT_END, EMPTY, EMPTY, EMPTY, comm, EMPTY);

	return ierror;
}

int Normal_MPI_Iprobe_C_Wrapper (int source, int tag, MPI_Comm comm, int *flag,
                          MPI_Status *status)
{
	static int       MPI_Iprobe_software_counter = 0;
	static iotimer_t MPI_Iprobe_elapsed_time = 0;
	iotimer_t        MPI_Iprobe_begin_time = 0;
	int              ierror;
	
	MPI_Iprobe_begin_time = LAST_READ_TIME;
	
	ierror = PMPI_Iprobe (source, tag, comm, flag, status);

	if (*flag)
	{
		// MPI_Iprobe was successful
		
		if (MPI_Iprobe_software_counter > 0)
		{
			// Only emit software counters if there were previous calls to MPI_Iprobe omitted
			TRACE_EVENT (MPI_Iprobe_begin_time, MPI_TIME_IN_IPROBE_EV, MPI_Iprobe_elapsed_time);
			TRACE_EVENT (MPI_Iprobe_begin_time, MPI_IPROBE_COUNTER_EV, MPI_Iprobe_software_counter);
		}
		// The successful MPI_Iprobe is marked in the trace
		TRACE_MPIEVENT (MPI_Iprobe_begin_time, MPI_IPROBE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, comm, EMPTY);
		TRACE_MPIEVENT (TIME, MPI_IPROBE_EV, EVT_END, EMPTY, EMPTY, EMPTY, comm, EMPTY);

		MPI_Iprobe_software_counter = 0;
		MPI_Iprobe_elapsed_time = 0;
	}
	else
	{
		// MPI_Iprobe was unsuccessful -- accumulate software counters

		if (MPI_Iprobe_software_counter == 0)
		{
			// Mark the first unsuccessful MPI_Iprobe
			TRACE_EVENTANDCOUNTERS (MPI_Iprobe_begin_time, MPI_IPROBE_COUNTER_EV, 0, TRUE);
		}

		MPI_Iprobe_software_counter ++;
		MPI_Iprobe_elapsed_time += (TIME - MPI_Iprobe_begin_time);
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

	// MPI stats
	updateStats_OTHER(global_mpi_stats);

	return ret;
}


#if defined(MPI3)

/******************************************************************************
 ***  MPI_Mprobe_C_Wrapper
 ******************************************************************************/

int MPI_Mprobe_C_Wrapper (int source, int tag, MPI_Comm comm, MPI_Message *message, MPI_Status *status)
{
	int ierror;

	/*
	 *   event  : MPROBE_EV                  value  : EVT_BEGIN
	 *   target : ---                        size   : ---
	 *   tag    : ---                        commid : communicator id 
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_MPROBE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, comm, EMPTY);

	ierror = PMPI_Mprobe (source, tag, comm, message, status);

	SaveMessage (*message, comm);

	/*
	 *   event  : MPROBE_EV                  value  : EVT_END
	 *   target : ---                        size   : ---
	 *   tag    : ---                        commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_MPROBE_EV, EVT_END, EMPTY, EMPTY, EMPTY, comm, EMPTY);

	// MPI stats
	updateStats_OTHER(global_mpi_stats);

	return ierror;
}


/******************************************************************************
 ***  MPI_Improbe_C_Wrapper
 ******************************************************************************/

int Bursts_MPI_Improbe_C_Wrapper (int source, int tag, MPI_Comm comm, int *flag, MPI_Message *message, MPI_Status *status)
{
	int ierror;

	/*
	 *   event  : IMPROBE_EV                 value  : EVT_BEGIN
	 *   target : ---                        size   : ---
	 *   tag    : ---                        commid : communicator id 
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IMPROBE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, comm, EMPTY);

	ierror = PMPI_Improbe (source, tag, comm, flag, message, status);

	SaveMessage (*message, comm);

	/*
	 *   event  : IMPROBE_EV                 value  : EVT_END
	 *   target : ---                        size   : ---
	 *   tag    : ---                        commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_IMPROBE_EV, EVT_END, EMPTY, EMPTY, EMPTY, comm, EMPTY);

	return ierror;
}

int Normal_MPI_Improbe_C_Wrapper (int source, int tag, MPI_Comm comm, int *flag, MPI_Message *message, MPI_Status *status)
{
	static int       MPI_Improbe_software_counter = 0;
	static iotimer_t MPI_Improbe_elapsed_time = 0;
	iotimer_t        MPI_Improbe_begin_time = 0;
	int              ierror;

	MPI_Improbe_begin_time = LAST_READ_TIME;

	ierror = PMPI_Improbe (source, tag, comm, flag, message, status);

	if (*flag)
	{
		// MPI_Improbe was successful

		SaveMessage(*message, comm);

		if (MPI_Improbe_software_counter > 0)
		{
			// Only emit software counters if there were previous calls to MPI_Iprobe omitted
			TRACE_EVENT (MPI_Improbe_begin_time, MPI_TIME_IN_IMPROBE_EV, MPI_Improbe_elapsed_time);
			TRACE_EVENT (MPI_Improbe_begin_time, MPI_IMPROBE_COUNTER_EV, MPI_Improbe_software_counter);
		}
		// The successful MPI_Improbe is marked in the trace
		TRACE_MPIEVENT (MPI_Improbe_begin_time, MPI_IMPROBE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, comm, EMPTY);
		TRACE_MPIEVENT (TIME, MPI_IMPROBE_EV, EVT_END, EMPTY, EMPTY, EMPTY, comm, EMPTY);
			
		MPI_Improbe_software_counter = 0;
		MPI_Improbe_elapsed_time = 0;
	}
	else
	{
		// MPI_Improbre was unsuccessful -- accumulate software counters

		if (MPI_Improbe_software_counter == 0)
		{
			// Mark the first unsuccessful MPI_Iprobe
			TRACE_EVENTANDCOUNTERS (MPI_Improbe_begin_time, MPI_IMPROBE_COUNTER_EV, 0, TRUE);
		}

		MPI_Improbe_software_counter ++;
		MPI_Improbe_elapsed_time += (TIME - MPI_Improbe_begin_time);
	}

	return ierror;
}

int MPI_Improbe_C_Wrapper (int source, int tag, MPI_Comm comm, int *flag, MPI_Message *message, MPI_Status *status)
{
	int ret;

	if (CURRENT_TRACE_MODE(THREADID) == TRACE_MODE_BURSTS)
	{
		ret = Bursts_MPI_Improbe_C_Wrapper (source, tag, comm, flag, message, status);
	}
	else
	{
		ret = Normal_MPI_Improbe_C_Wrapper (source, tag, comm, flag, message, status);
	}
	return ret;
}

#endif /* MPI3 */


void copyRequests_C (int count, MPI_Request *array_of_requests, MPI_Request *copy, char *where)
{
	if (count > MAX_WAIT_REQUESTS)
	{
		fprintf (stderr, "PANIC! Number of requests in %s (%d) exceeds tha maximum supported (%d). Please increase the value of MAX_WAIT_REQUESTS and recompile Extrae.\n", where, count, MAX_WAIT_REQUESTS);
	}
	memcpy (copy, array_of_requests, count * sizeof(MPI_Request));
}


/******************************************************************************
 ***  MPI_Test_C_Wrapper
 ******************************************************************************/

int Bursts_MPI_Test_C_Wrapper (MPI_Request *request, int *flag, MPI_Status *status)
{
	int         ierror;
	iotimer_t   MPI_Test_end_time;
	MPI_Request save_req;

	/*
	 *   event  : TEST_EV                    value  : EVT_BEGIN
	 *   target : request id                 size   : ---
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_TEST_EV, EVT_BEGIN, *request, EMPTY, EMPTY, EMPTY, EMPTY);

	copyRequests_C (1, request, &save_req, "MPI_Test");

	ierror = PMPI_Test (request, flag, status);

	MPI_Test_end_time = TIME;

        if (ierror == MPI_SUCCESS && *flag)
	{
		ProcessRequest (MPI_Test_end_time, save_req, status);
	}

	/*
	 *   event  : TEST_EV                    value  : EVT_END
	 *   target : ---                        size   : ---
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (MPI_Test_end_time, MPI_TEST_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	return ierror;
}

int Normal_MPI_Test_C_Wrapper (MPI_Request *request, int *flag, MPI_Status *status)
{
	static int       MPI_Test_software_counter = 0;
	static iotimer_t MPI_Test_elapsed_time = 0;
	iotimer_t        MPI_Test_begin_time = 0;
	iotimer_t        MPI_Test_end_time = 0;
	int              ierror;
	MPI_Request      save_req;

	MPI_Test_begin_time = LAST_READ_TIME;

	copyRequests_C (1, request, &save_req, "MPI_Test");

	ierror = PMPI_Test (request, flag, status);

        if (ierror == MPI_SUCCESS && *flag)
	{
		// MPI_Test was successful

		if (MPI_Test_software_counter > 0)
		{
			// Only emit software counters if there were previous calls to MPI_Test omitted
			TRACE_EVENT(MPI_Test_begin_time, MPI_TIME_IN_TEST_EV, MPI_Test_elapsed_time);
			TRACE_EVENT(MPI_Test_begin_time, MPI_TEST_COUNTER_EV, MPI_Test_software_counter);
		}
		// The successful MPI_Test is marked in the trace
		TRACE_MPIEVENT (MPI_Test_begin_time, MPI_TEST_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

		MPI_Test_end_time = TIME;

		ProcessRequest (MPI_Test_end_time, save_req, status);

		TRACE_MPIEVENT (MPI_Test_end_time, MPI_TEST_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

		MPI_Test_software_counter = 0;
		MPI_Test_elapsed_time = 0;
	}
	else
	{
		// MPI_Test was unsuccessful -- accumulate software counters
		if (MPI_Test_software_counter == 0)
		{
			// Mark the first unsuccessful MPI_Test 
			TRACE_EVENTANDCOUNTERS (MPI_Test_begin_time, MPI_TEST_COUNTER_EV, 0, TRUE);
		}
		MPI_Test_software_counter ++;
		MPI_Test_elapsed_time += (TIME - MPI_Test_begin_time);
	}
	return ierror;
}

int MPI_Test_C_Wrapper (MPI_Request *request, int *flag, MPI_Status *status)
{
	MPI_Status my_status, *ptr_status = NULL;
	int ret;

	ptr_status = (MPI_STATUS_IGNORE == status) ? &my_status : status;

	if (CURRENT_TRACE_MODE(THREADID) == TRACE_MODE_BURSTS)
	{
		ret = Bursts_MPI_Test_C_Wrapper (request, flag, ptr_status);
	}
	else
	{
		ret = Normal_MPI_Test_C_Wrapper (request, flag, ptr_status);
	}

	return ret;
}

/******************************************************************************
 ***  MPI_Testall_C_Wrapper
 ******************************************************************************/

int MPI_Testall_C_Wrapper (int count, MPI_Request *array_of_requests, int *flag,
	MPI_Status *array_of_statuses)
{
	MPI_Request      save_reqs[MAX_WAIT_REQUESTS];
	MPI_Status       my_statuses[MAX_WAIT_REQUESTS];
	MPI_Status      *ptr_array_of_statuses = (array_of_statuses == MPI_STATUSES_IGNORE) ? my_statuses : array_of_statuses;
	static int       MPI_Testall_software_counter = 0;
	static iotimer_t MPI_Testall_elapsed_time = 0;
	iotimer_t        MPI_Testall_begin_time = 0;
	iotimer_t        MPI_Testall_end_time = 0;
	int              ierror;
#if defined(DEBUG_MPITRACE)
	int              index;
#endif

	MPI_Testall_begin_time = LAST_READ_TIME;
	
	copyRequests_C (count, array_of_requests, save_reqs, "MPI_Testall");

#if defined(DEBUG_MPITRACE)
	fprintf (stderr,  PACKAGE_NAME" %d: TESTALL summary\n", TASKID);
	for (index = 0; index < count; index++)
# if SIZEOF_LONG == 8
		fprintf (stderr, "%d: position %d -> request %lu\n", TASKID, index, (UINT64) array_of_requests[index]);
# elif SIZEOF_LONG == 4
		fprintf (stderr, "%d: position %d -> request %llu\n", TASKID, index, (UINT64) array_of_requests[index]);
# endif
#endif

	ierror = PMPI_Testall (count, array_of_requests, flag, ptr_array_of_statuses);

	if (ierror == MPI_SUCCESS && *flag)
	{
		int ireq;

		// MPI_Testall was successful

		if (MPI_Testall_software_counter > 0)
		{
			// Only emit software counters if there were previous calls to MPI_Testall omitted
			TRACE_EVENT(MPI_Testall_begin_time, MPI_TIME_IN_TEST_EV, MPI_Testall_elapsed_time);
			TRACE_EVENT(MPI_Testall_begin_time, MPI_TEST_COUNTER_EV, MPI_Testall_software_counter);
		}

		// The successful MPI_Testall is marked in the trace
		TRACE_MPIEVENT (MPI_Testall_begin_time, MPI_TESTALL_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

		MPI_Testall_end_time = TIME;

		for (ireq = 0; ireq < count; ireq++)
		{
			ProcessRequest (MPI_Testall_end_time, save_reqs[ireq], &(ptr_array_of_statuses[ireq]));
		}
		TRACE_MPIEVENT (MPI_Testall_end_time, MPI_TESTALL_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

		MPI_Testall_software_counter = 0;
		MPI_Testall_elapsed_time = 0;
	}
	else
	{
		// MPI_Testall was unsuccessful -- accumulate software counters 
		if (MPI_Testall_software_counter == 0)
		{
			// Mark the first unsuccessful MPI_Testall 
			TRACE_EVENTANDCOUNTERS (MPI_Testall_begin_time, MPI_TEST_COUNTER_EV, 0, TRUE);
		}
		MPI_Testall_software_counter ++;
		MPI_Testall_elapsed_time += (TIME - MPI_Testall_begin_time);
	}

	return ierror;
}

/******************************************************************************
 ***  MPI_Testany_C_Wrapper
 ******************************************************************************/

int MPI_Testany_C_Wrapper (int count, MPI_Request *array_of_requests,
                           int *index, int *flag, MPI_Status *status)
{
	MPI_Status       my_status, *ptr_status;
	MPI_Request      save_reqs[MAX_WAIT_REQUESTS];
	static int       MPI_Testany_software_counter = 0;
	static iotimer_t MPI_Testany_elapsed_time = 0;
	iotimer_t        MPI_Testany_begin_time = 0;
	iotimer_t        MPI_Testany_end_time = 0;
	int              ierror;
#if defined(DEBUG_MPITRACE)
	int              i;
#endif

	MPI_Testany_begin_time = LAST_READ_TIME;
	
        copyRequests_C (count, array_of_requests, save_reqs, "MPI_Testany");

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

	if (*index != MPI_UNDEFINED && ierror == MPI_SUCCESS && *flag)
	{
		// MPI_Testany was successful 

		if (MPI_Testany_software_counter > 0)
		{
			// Only emit software counters if there were previous calls to MPI_Testany omitted
			TRACE_EVENT(MPI_Testany_begin_time, MPI_TIME_IN_TEST_EV, MPI_Testany_elapsed_time);
			TRACE_EVENT(MPI_Testany_begin_time, MPI_TEST_COUNTER_EV, MPI_Testany_software_counter);
		}
		// The successful MPI_Testany is marked in the trace
		TRACE_MPIEVENT (MPI_Testany_begin_time, MPI_TESTANY_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

		MPI_Testany_end_time = TIME;

		ProcessRequest (MPI_Testany_end_time, save_reqs[*index], ptr_status);

		TRACE_MPIEVENT (MPI_Testany_end_time, MPI_TESTANY_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

		MPI_Testany_software_counter = 0;
		MPI_Testany_elapsed_time = 0;
	}
	else
	{
		// MPI_Testany was unsuccessful -- accumulate software counters
		if (MPI_Testany_software_counter == 0)
		{
			// Mark the first unsuccessful MPI_Testany
			TRACE_EVENTANDCOUNTERS (MPI_Testany_begin_time, MPI_TEST_COUNTER_EV, 0, TRUE);
		}
		MPI_Testany_software_counter ++;
		MPI_Testany_elapsed_time += (TIME - MPI_Testany_begin_time);
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
	MPI_Status       my_statuses[MAX_WAIT_REQUESTS];
	MPI_Status      *ptr_array_of_statuses = (array_of_statuses == MPI_STATUSES_IGNORE) ? my_statuses : array_of_statuses;
	MPI_Request      save_reqs[MAX_WAIT_REQUESTS];
	static int       MPI_Testsome_software_counter = 0;
	static iotimer_t MPI_Testsome_elapsed_time = 0;
	iotimer_t        MPI_Testsome_begin_time = 0;
	iotimer_t        MPI_Testsome_end_time = 0;
	int              ierror, ii;
#if defined(DEBUG_MPITRACE)
	int              index;
#endif
	
	MPI_Testsome_begin_time = LAST_READ_TIME;

        copyRequests_C (incount, array_of_requests, save_reqs, "MPI_Testsome");

#if defined(DEBUG_MPITRACE)
	fprintf (stderr, PACKAGE_NAME " %d: TESTSOME summary\n", TASKID);
	for (index = 0; index < incount; index++)
# if SIZEOF_LONG == 8
		fprintf (stderr, "%d: position %d -> request %lu\n", TASKID, index, (UINT64) array_of_requests[index]);
# elif SIZEOF_LONG == 4
		fprintf (stderr, "%d: position %d -> request %llu\n", TASKID, index, (UINT64) array_of_requests[index]);
# endif
#endif

	ierror = PMPI_Testsome (incount, array_of_requests, outcount, array_of_indices, ptr_array_of_statuses);

	if (ierror == MPI_SUCCESS && *outcount > 0)
	{
		// MPI_Testsome was successful
		
		if (MPI_Testsome_software_counter > 0)
		{
			// Only emit software counters if there where previous calls to MPI_Testsome omitted
			TRACE_EVENT(MPI_Testsome_begin_time, MPI_TIME_IN_TEST_EV, MPI_Testsome_elapsed_time);
			TRACE_EVENT(MPI_Testsome_begin_time, MPI_TEST_COUNTER_EV, MPI_Testsome_software_counter);
		}

		// The successful MPI_Testsome is marked in the trace
		TRACE_MPIEVENT (MPI_Testsome_begin_time, MPI_TESTSOME_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

		MPI_Testsome_end_time = TIME;

		for (ii = 0; ii < *outcount; ii++)
		{
			ProcessRequest (MPI_Testsome_end_time, save_reqs[array_of_indices[ii]], &(ptr_array_of_statuses[ii]));
		}

		TRACE_MPIEVENT (MPI_Testsome_end_time, MPI_TESTSOME_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

		MPI_Testsome_software_counter = 0;
		MPI_Testsome_elapsed_time = 0;
	}
	else
	{
		// MPI_Testsome was unsuccessful -- accumulate software counters
		if (MPI_Testsome_software_counter == 0)
		{
			// Mark the first unsuccessful MPI_Testsome 
			TRACE_EVENTANDCOUNTERS (MPI_Testsome_begin_time, MPI_TEST_COUNTER_EV, 0, TRUE);
		}
		MPI_Testsome_software_counter ++;
		MPI_Testsome_elapsed_time += (TIME - MPI_Testsome_begin_time);
	}

	return ierror;
}



/******************************************************************************
 ***  MPI_Wait_C_Wrapper
 ******************************************************************************/

int MPI_Wait_C_Wrapper (MPI_Request *request, MPI_Status *status)
{
	MPI_Status  my_status, *ptr_status = NULL;
	iotimer_t   MPI_Wait_end_time;
	int         ierror;
	MPI_Request save_req;

	/*
	 *   event  : WAIT_EV                    value  : EVT_BEGIN
	 *   target : request id                 size   : ---
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_WAIT_EV, EVT_BEGIN, *request, EMPTY, EMPTY, EMPTY, EMPTY);

	copyRequests_C (1, request, &save_req, "MPI_Wait");

	ptr_status = (MPI_STATUS_IGNORE == status) ? &my_status : status;

	ierror = PMPI_Wait (request, ptr_status);

	MPI_Wait_end_time = TIME;

	if (ierror == MPI_SUCCESS)
	{
		ProcessRequest (MPI_Wait_end_time, save_req, ptr_status);
	}

	/*
	 *   event  : WAIT_EV                    value  : EVT_END
	 *   target : ---                        size   : ---
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (MPI_Wait_end_time, MPI_WAIT_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	return ierror;
}


/******************************************************************************
 ***  MPI_Waitall_C_Wrapper
 ******************************************************************************/

int MPI_Waitall_C_Wrapper (int count, MPI_Request *array_of_requests,
                           MPI_Status *array_of_statuses)
{
	MPI_Status  my_statuses[MAX_WAIT_REQUESTS];
	MPI_Status *ptr_array_of_statuses = (array_of_statuses == MPI_STATUSES_IGNORE) ? my_statuses : array_of_statuses;
	MPI_Request save_reqs[MAX_WAIT_REQUESTS];
	iotimer_t   MPI_Waitall_end_time;
	int         ierror;
#if defined(DEBUG_MPITRACE)
	int index;
#endif

	/*
	 *   event  : WAITALL_EV                 value  : EVT_BEGIN
	 *   target : ---                        size   : ---
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_WAITALL_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

        copyRequests_C (count, array_of_requests, save_reqs, "MPI_Waitall");

#if defined(DEBUG_MPITRACE)
	fprintf (stderr,  PACKAGE_NAME" %d: WAITALL summary\n", TASKID);
	for (index = 0; index < count; index++)
# if SIZEOF_LONG == 8
		fprintf (stderr, "%d: position %d -> request %lu\n", TASKID, index, (UINT64) array_of_requests[index]);
# elif SIZEOF_LONG == 4
		fprintf (stderr, "%d: position %d -> request %llu\n", TASKID, index, (UINT64) array_of_requests[index]);
# endif
#endif

	ierror = PMPI_Waitall (count, array_of_requests, ptr_array_of_statuses);

	MPI_Waitall_end_time = TIME;

	if (ierror == MPI_SUCCESS)
	{
		int ireq;

		for (ireq = 0; ireq < count; ireq++)
		{
			ProcessRequest (MPI_Waitall_end_time, save_reqs[ireq], &(ptr_array_of_statuses[ireq]));
		}
	}

	/*
	 *   event  : WAITATLL_EV                value  : EVT_END
	 *   target : ---                        size   : ---
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (MPI_Waitall_end_time, MPI_WAITALL_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	return ierror;
}


/******************************************************************************
 ***  MPI_Waitany_C_Wrapper
 ******************************************************************************/

int MPI_Waitany_C_Wrapper (int count, MPI_Request *array_of_requests,
                           int *index, MPI_Status *status)
{
	MPI_Status  my_status, *ptr_status = NULL;
	MPI_Request save_reqs[MAX_WAIT_REQUESTS];
	iotimer_t   MPI_Waitany_end_time;
	int         ierror;
#if defined(DEBUG_MPITRACE)
	int         i;
#endif

	/*
	 *   event  : WAITANY_EV                 value  : EVT_BEGIN
	 *   target : ---                        size   : ---
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_WAITANY_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

        copyRequests_C (count, array_of_requests, save_reqs, "MPI_Waitany");

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

	MPI_Waitany_end_time = TIME;

	if (*index != MPI_UNDEFINED && ierror == MPI_SUCCESS)
	{
		ProcessRequest (MPI_Waitany_end_time, save_reqs[*index], ptr_status);
	}

	/*
	 *   event  : WAITANY_EV                 value  : EVT_END
	 *   target : ---                        size   : ---
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (MPI_Waitany_end_time, MPI_WAITANY_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	return ierror;
}


/******************************************************************************
 ***  MPI_Waitsome_C_Wrapper
 ******************************************************************************/

int MPI_Waitsome_C_Wrapper (int incount, MPI_Request *array_of_requests,
                            int *outcount, int *array_of_indices,
                            MPI_Status *array_of_statuses)
{
	MPI_Status  my_statuses[MAX_WAIT_REQUESTS];
	MPI_Status *ptr_array_of_statuses = (array_of_statuses == MPI_STATUSES_IGNORE) ? my_statuses : array_of_statuses;
	MPI_Request save_reqs[MAX_WAIT_REQUESTS];
	iotimer_t   MPI_Waitsome_end_time;
	int         ierror;
#if defined(DEBUG_MPITRACE)
	int         index;
#endif

	/*
	 *   event  : WAITSOME_EV                value  : EVT_BEGIN
	 *   target : ---                        size   : ---
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_WAITSOME_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

        copyRequests_C (incount, array_of_requests, save_reqs, "MPI_Waitsome");

#if defined(DEBUG_MPITRACE)
	fprintf (stderr, PACKAGE_NAME " %d: WAITSOME summary\n", TASKID);
	for (index = 0; index < incount; index++)
# if SIZEOF_LONG == 8
		fprintf (stderr, "%d: position %d -> request %lu\n", TASKID, index, (UINT64) array_of_requests[index]);
# elif SIZEOF_LONG == 4
		fprintf (stderr, "%d: position %d -> request %llu\n", TASKID, index, (UINT64) array_of_requests[index]);
# endif
#endif

	ierror = PMPI_Waitsome (incount, array_of_requests, outcount, array_of_indices, ptr_array_of_statuses);

	MPI_Waitsome_end_time = TIME;

	if (ierror == MPI_SUCCESS && *outcount > 0)
	{
		int ii;

		for (ii = 0; ii < *outcount; ii++)
		{
			ProcessRequest (MPI_Waitsome_end_time, save_reqs[array_of_indices[ii]], &(ptr_array_of_statuses[ii]));
		}
	}

	/*
	 *   event  : WAITSOME_EV                value  : EVT_END
	 *   target : ---                        size   : ---
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (MPI_Waitsome_end_time, MPI_WAITSOME_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

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
	 *   event  : RECV_INIT_EV               value  : EVT_BEGIN
	 *   target : ---                        size   : ----
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_RECV_INIT_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	// First call PMPI_Recv_init to initialize the persistent request
	ierror = PMPI_Recv_init (buf, count, datatype, source, tag, comm, request);

	// Save this persistent request
	PR_NewRequest (MPI_IRECV_EV, count, datatype, source, tag, comm, *request, &PR_queue);

	/*
	 *   event  : RECV_INIT_EV               value  : EVT_END
	 *   target : ---                        size   : ----
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_RECV_INIT_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

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
	 *   event  : SEND_INIT_EV               value  : EVT_BEGIN
	 *   target : ---                        size   : ----
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_SEND_INIT_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	// First call PMPI_Send_init to initialize the persistent request
	ierror = PMPI_Send_init (buf, count, datatype, dest, tag, comm, request);

	// Save this persistent request
	PR_NewRequest (MPI_ISEND_EV, count, datatype, dest, tag, comm, *request, &PR_queue);

	/*
	 *   event  : SEND_INIT_EV               value  : EVT_END
	 *   target : ---                        size   : ----
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_SEND_INIT_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

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
	 *   event  : BSEND_INIT_EV              value  : EVT_BEGIN
	 *   target : ---                        size   : ----
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_BSEND_INIT_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	// First call PMPI_Bsend_init to initialize the persistent request
	ierror = PMPI_Bsend_init (buf, count, datatype, dest, tag, comm, request);

	// Save this persistent request
	PR_NewRequest (MPI_IBSEND_EV, count, datatype, dest, tag, comm, *request, &PR_queue);

	/*
	 *   event  : BSEND_INIT_EV              value  : EVT_END
	 *   target : ---                        size   : ----
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_BSEND_INIT_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

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
	 *   event  : RSEND_INIT_EV              value  : EVT_BEGIN
	 *   target : ---                        size   : ----
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_RSEND_INIT_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	// First call PMPI_Rsend_init to initialize the persistent request
	ierror = PMPI_Rsend_init (buf, count, datatype, dest, tag, comm, request);

	// Save this persistent request
	PR_NewRequest (MPI_IRSEND_EV, count, datatype, dest, tag, comm, *request, &PR_queue);

	/*
	 *   event  : RSEND_INIT_EV              value  : EVT_END
	 *   target : ---                        size   : ----
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_RSEND_INIT_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

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
	 *   event  : SSEND_INIT_EV              value  : EVT_BEGIN
	 *   target : ---                        size   : ----
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_SSEND_INIT_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY,
	  EMPTY);

	// First call PMPI_Ssend_init to initialize the persistent request
	ierror = PMPI_Ssend_init (buf, count, datatype, dest, tag, comm, request);

	// Save this persistent request
	PR_NewRequest (MPI_ISSEND_EV, count, datatype, dest, tag, comm, *request, &PR_queue);

	/*
	 *   event  : SSEND_INIT_EV              value  : EVT_END
	 *   target : ---                        size   : ----
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_SSEND_INIT_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);

	return ierror;
}


/******************************************************************************
 ***  MPI_Sendrecv_C_Wrapper
 ******************************************************************************/

int MPI_Sendrecv_C_Wrapper (void *sendbuf, int sendcount, MPI_Datatype sendtype,
                            int dest,      int sendtag,   void *recvbuf, 
                            int recvcount, MPI_Datatype recvtype,
                            int source,    int recvtag,   MPI_Comm comm, MPI_Status * status) 
{
	MPI_Status my_status, *ptr_status = NULL;
	int        SentSize, ReceivedSize, SenderRank, ReceiverRank, Tag;
	int        ierror;

	SentSize = getMsgSizeFromCountAndDatatype (sendcount, sendtype);

	translateLocalToGlobalRank (comm, MPI_GROUP_NULL, dest, &ReceiverRank, OP_TYPE_SEND);

	/*
	 *   event  : SENDRECV_EV                value  : EVT_BEGIN
	 *   target : receiver rank              size   : bytes sent
	 *   tag    : message tag                commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_SENDRECV_EV, EVT_BEGIN, ReceiverRank, SentSize, sendtag, comm, EMPTY);

	ptr_status = (status == MPI_STATUS_IGNORE) ? &my_status : status;

	ierror = PMPI_Sendrecv (sendbuf, sendcount, sendtype, dest,   sendtag,
	                        recvbuf, recvcount, recvtype, source, recvtag, 
	                        comm,    ptr_status);

	getCommDataFromStatus (ptr_status, recvtype, comm, MPI_GROUP_NULL, &ReceivedSize, &Tag, &SenderRank);

	/*
	 *   event  : SENDRECV_EV                value  : EVT_END
	 *   target : sender rank                size   : bytes received
	 *   tag    : message tag                commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_SENDRECV_EV, EVT_END, SenderRank, ReceivedSize, Tag, comm, EMPTY);

	// MPI stats 
	updateStats_P2P(global_mpi_stats, SenderRank, ReceivedSize, SentSize);

	return ierror;
}


/******************************************************************************
 ***  MPI_Sendrecv_replace
 ******************************************************************************/

int MPI_Sendrecv_replace_C_Wrapper (void *buf, int count, MPI_Datatype type,
  int dest, int sendtag, int source, int recvtag, MPI_Comm comm,
  MPI_Status * status) 
{
	MPI_Status my_status, *ptr_status = NULL;
	int        SentSize, ReceivedSize, SenderRank, ReceiverRank, Tag;
	int        ierror;

	SentSize = getMsgSizeFromCountAndDatatype (count, type);

	translateLocalToGlobalRank (comm, MPI_GROUP_NULL, dest, &ReceiverRank, OP_TYPE_SEND);
	
	/*
	 *   event  : SENDRECV_REPLACE_EV        value  : EVT_BEGIN
	 *   target : receiver rank              size   : bytes sent
	 *   tag    : message tag                commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_SENDRECV_REPLACE_EV, EVT_BEGIN, ReceiverRank, SentSize, sendtag, comm, EMPTY);

	ptr_status = (status == MPI_STATUS_IGNORE) ? &my_status : status;

	ierror = PMPI_Sendrecv_replace (buf, count, type, dest, sendtag, source, recvtag, comm, ptr_status);

	getCommDataFromStatus (ptr_status, type, comm, MPI_GROUP_NULL, &ReceivedSize, &Tag, &SenderRank);

	/*
	 *   event  : SENDRECV_REPLACE_EV        value  : EVT_END
	 *   target : sender rank                size   : bytes received
	 *   tag    : message tag                commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_SENDRECV_REPLACE_EV, EVT_END, SenderRank, ReceivedSize, Tag, comm, EMPTY);

	// MPI stats 
	updateStats_P2P(global_mpi_stats, SenderRank, ReceivedSize, SentSize);

	return ierror;
}

#endif /* defined(C_SYMBOLS) */

