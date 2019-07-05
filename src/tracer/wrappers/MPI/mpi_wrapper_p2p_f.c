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


/******************************************************************************
 ***  PMPI_BSend_Wrapper
 ******************************************************************************/

void PMPI_BSend_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
                         MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror)
{
	MPI_Datatype c_type         = PMPI_Type_f2c(*datatype);
	int          size           = 0;
	int          receiver_world = MPI_PROC_NULL;
	int          c_tag          = *tag;
	MPI_Comm     c_comm         = PMPI_Comm_f2c (*comm);

	size = getMsgSizeFromCountAndDatatype (*count, c_type);

	translateLocalToGlobalRank (c_comm, MPI_GROUP_NULL, *dest, &receiver_world, OP_TYPE_SEND);

	/*
	 *   event  : BSEND_EV                          value  : EVT_BEGIN
	 *   target : receiver rank                     size   : bytes sent
	 *   tag    : message tag                       commid : communicator id 
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_BSEND_EV, EVT_BEGIN, receiver_world, size, c_tag, c_comm, EMPTY);

	CtoF77 (pmpi_bsend) (buf, count, datatype, dest, tag, comm, ierror);

	/*
	 *   event  : BSEND_EV                          value  : EVT_END
	 *   target : receiver rank                     size   : bytes sent
	 *   tag    : message tag                       commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_BSEND_EV, EVT_END, receiver_world, size, c_tag, c_comm, EMPTY);

	// MPI stats
	updateStats_P2P(global_mpi_stats, receiver_world, 0, size);
}


/******************************************************************************
 ***  PMPI_SSend_Wrapper
 ******************************************************************************/

void PMPI_SSend_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
                         MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror)
{
	MPI_Datatype c_type         = PMPI_Type_f2c(*datatype);
	int          size           = 0;
	int          receiver_world = MPI_PROC_NULL;
	int          c_tag          = *tag;
	MPI_Comm     c_comm         = PMPI_Comm_f2c (*comm);

	size = getMsgSizeFromCountAndDatatype (*count, c_type);

	translateLocalToGlobalRank (c_comm, MPI_GROUP_NULL, *dest, &receiver_world, OP_TYPE_SEND);

	/*
	 *   event  : SSEND_EV                          value  : EVT_BEGIN
	 *   target : receiver rank                     size   : bytes sent
	 *   tag    : message tag                       commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_SSEND_EV, EVT_BEGIN, receiver_world, size, c_tag, c_comm, EMPTY);

	CtoF77 (pmpi_ssend) (buf, count, datatype, dest, tag, comm, ierror);

	/*
	 *   event  : SSEND_EV                          value  : EVT_END
	 *   target : receiver rank                     size   : bytes sent
	 *   tag    : message tag                       commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_SSEND_EV, EVT_END, receiver_world, size, c_tag, c_comm, EMPTY);

	// MPI stats
	updateStats_P2P(global_mpi_stats, receiver_world, 0, size);
}

/******************************************************************************
 ***  PMPI_RSend_Wrapper
 ******************************************************************************/

void PMPI_RSend_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
                         MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror)
{
	MPI_Datatype c_type         = PMPI_Type_f2c(*datatype);
	int          size           = 0;
	int          receiver_world = MPI_PROC_NULL;
	int          c_tag          = *tag;
	MPI_Comm     c_comm         = PMPI_Comm_f2c (*comm);

	size = getMsgSizeFromCountAndDatatype (*count, c_type);

	translateLocalToGlobalRank (c_comm, MPI_GROUP_NULL, *dest, &receiver_world, OP_TYPE_SEND);

	/*
	 *   event  : RSEND_EV                   value  : EVT_BEGIN
	 *   target : receiver rank              size   : bytes sent
	 *   tag    : message tag                commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_RSEND_EV, EVT_BEGIN, receiver_world, size, c_tag, c_comm, EMPTY);

	CtoF77 (pmpi_rsend) (buf, count, datatype, dest, tag, comm, ierror);

	/*
	 *   event  : RSEND_EV                   value  : EVT_END
	 *   target : receiver rank              size   : bytes sent
	 *   tag    : message tag                commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_RSEND_EV, EVT_END, receiver_world, size, c_tag, c_comm, EMPTY);

	// MPI stats
	updateStats_P2P(global_mpi_stats, receiver_world, 0, size);
}

/******************************************************************************
 ***  PMPI_Send_Wrapper
 ******************************************************************************/

void PMPI_Send_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
                        MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror)
{
	MPI_Datatype c_type         = PMPI_Type_f2c(*datatype);
	int          size           = 0;
	int          receiver_world = MPI_PROC_NULL;
	int          c_tag          = *tag;
	MPI_Comm     c_comm         = PMPI_Comm_f2c (*comm);

	size = getMsgSizeFromCountAndDatatype (*count, c_type);

	translateLocalToGlobalRank (c_comm, MPI_GROUP_NULL, *dest, &receiver_world, OP_TYPE_SEND);

	/*
	 *   event  : SEND_EV                           value  : EVT_BEGIN
	 *   target : receiver rank                     size   : bytes sent
	 *   tag    : message tag                       commid : communicator id 
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_SEND_EV, EVT_BEGIN, receiver_world, size, c_tag, c_comm, EMPTY);

	CtoF77 (pmpi_send) (buf, count, datatype, dest, tag, comm, ierror);

	/*
	 *   event  : SEND_EV                           value  : EVT_END
	 *   target : receiver rank                     size   : bytes sent
	 *   tag    : message tag                       commid : communicator id
	 *   aux    : --- 
	 */
	TRACE_MPIEVENT (TIME, MPI_SEND_EV, EVT_END, receiver_world, size, c_tag, c_comm, EMPTY);

	// MPI stats
	updateStats_P2P(global_mpi_stats, receiver_world, 0, size);
}


/******************************************************************************
 ***  PMPI_IBSend_Wrapper
 ******************************************************************************/

void PMPI_IBSend_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
                          MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
                          MPI_Fint *ierror)
{
	MPI_Datatype c_type         = PMPI_Type_f2c(*datatype);
	int          size           = 0;
	int          receiver_world = MPI_PROC_NULL;
	int          c_tag          = *tag;
	MPI_Comm     c_comm         = PMPI_Comm_f2c (*comm);

	size = getMsgSizeFromCountAndDatatype (*count, c_type);

	translateLocalToGlobalRank (c_comm, MPI_GROUP_NULL, *dest, &receiver_world, OP_TYPE_SEND);

	/*
	 *   event  : IBSEND_EV                         value  : EVT_BEGIN
	 *   target : receiver rank                     size   : bytes sent
	 *   tag    : message tag                       commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IBSEND_EV, EVT_BEGIN, receiver_world, size, c_tag, c_comm, EMPTY);

	CtoF77 (pmpi_ibsend) (buf, count, datatype, dest, tag, comm, request, ierror);

	/*
	 *   event  : IBSEND_EV                         value  : EVT_END
	 *   target : receiver rank                     size   : bytes sent
	 *   tag    : message tag                       commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_IBSEND_EV, EVT_END, receiver_world, size, c_tag, c_comm, EMPTY);

	// MPI stats
	updateStats_P2P(global_mpi_stats, receiver_world, 0, size);
}


/******************************************************************************
 ***  PMPI_ISend_Wrapper
 ******************************************************************************/

void PMPI_ISend_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
                         MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
                         MPI_Fint *ierror)
{
	MPI_Datatype c_type         = PMPI_Type_f2c(*datatype);
	int          size           = 0;
	int          receiver_world = MPI_PROC_NULL;
	int          c_tag          = *tag;
	MPI_Comm     c_comm         = PMPI_Comm_f2c (*comm);

	size = getMsgSizeFromCountAndDatatype (*count, c_type);

	translateLocalToGlobalRank (c_comm, MPI_GROUP_NULL, *dest, &receiver_world, OP_TYPE_SEND);

	/*
	 *   event  : ISEND_EV                          value  : EVT_BEGIN
	 *   target : receiver rank                     size   : bytes sent
	 *   tag    : message tag                       commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_ISEND_EV, EVT_BEGIN, receiver_world, size, c_tag, c_comm, EMPTY);

	CtoF77 (pmpi_isend) (buf, count, datatype, dest, tag, comm, request, ierror);

	/*
	 *   event  : ISEND_EV                          value  : EVT_END
	 *   target : receiver rank                     size   : bytes sent
	 *   tag    : message tag                       commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_ISEND_EV, EVT_END, receiver_world, size, c_tag, c_comm, EMPTY);

	// MPI stats
	updateStats_P2P(global_mpi_stats, receiver_world, 0, size);
}


/******************************************************************************
 ***  PMPI_ISSend_Wrapper
 ******************************************************************************/

void PMPI_ISSend_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
                          MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
                          MPI_Fint *ierror)
{
	MPI_Datatype c_type         = PMPI_Type_f2c(*datatype);
	int          size           = 0;
	int          receiver_world = MPI_PROC_NULL;
	int          c_tag          = *tag;
	MPI_Comm     c_comm         = PMPI_Comm_f2c (*comm);

	size = getMsgSizeFromCountAndDatatype (*count, c_type);

	translateLocalToGlobalRank (c_comm, MPI_GROUP_NULL, *dest, &receiver_world, OP_TYPE_SEND);

	/*
	 *   event : ISSEND_EV                     value : EVT_BEGIN
	 *   target : receiver rank                size  : bytes sent
	 *   tag : message tag                     commid: communicator id
	 *   aux : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_ISSEND_EV, EVT_BEGIN, receiver_world, size, c_tag, c_comm, EMPTY);

	CtoF77 (pmpi_issend) (buf, count, datatype, dest, tag, comm, request, ierror);

	/*
	 *   event : ISSEND_EV                     value : EVT_END
	 *   target : receiver rank                size  : bytes sent
	 *   tag : message tag                     commid: communicator id
	 *   aux : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_ISSEND_EV, EVT_END, receiver_world, size, c_tag, c_comm, EMPTY);

	// MPI stats 
	updateStats_P2P(global_mpi_stats, receiver_world, 0, size);
}


/******************************************************************************
 ***  PMPI_IRSend_Wrapper
 ******************************************************************************/

void PMPI_IRSend_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
                          MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
                          MPI_Fint *ierror)
{
	MPI_Datatype c_type         = PMPI_Type_f2c(*datatype);
	int          size           = 0;
	int          receiver_world = MPI_PROC_NULL;
	int          c_tag          = *tag;
	MPI_Comm     c_comm         = PMPI_Comm_f2c (*comm);

	size = getMsgSizeFromCountAndDatatype (*count, c_type);

	translateLocalToGlobalRank (c_comm, MPI_GROUP_NULL, *dest, &receiver_world, OP_TYPE_SEND);

	/*
	 *   event  : IRSEND_EV                         value  : EVT_BEGIN
	 *   target : receiver rank                     size   : bytes sent
	 *   tag    : message tag                       commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IRSEND_EV, EVT_BEGIN, receiver_world, size, c_tag, c_comm, EMPTY);

	CtoF77 (pmpi_irsend) (buf, count, datatype, dest, tag, comm, request, ierror);

	/*
	 *   event  : IRSEND_EV                         value  : EVT_END
	 *   target : receiver rank                     size   : bytes sent
	 *   tag    : message tag                       commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_IRSEND_EV, EVT_END, receiver_world, size, c_tag, c_comm, EMPTY);

	// MPI stats
	updateStats_P2P(global_mpi_stats, receiver_world, 0, size);
}


/******************************************************************************
 ***  PMPI_Recv_Wrapper
 ******************************************************************************/

void PMPI_Recv_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
                        MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *status, 
                        MPI_Fint *ierror)
{
	MPI_Datatype c_type       = PMPI_Type_f2c(*datatype);
	int          size         = 0; 
	int          source_world = MPI_ANY_SOURCE;
	int          c_tag        = *tag;
	int          source_tag   = MPI_ANY_TAG;
	MPI_Comm     c_comm       = PMPI_Comm_f2c(*comm);
	MPI_Fint     f_status[SIZEOF_MPI_STATUS];
	MPI_Fint    *f_status_ptr = NULL;
	MPI_Status   c_status;

	size = getMsgSizeFromCountAndDatatype (*count, c_type);

	translateLocalToGlobalRank (c_comm, MPI_GROUP_NULL, *source, &source_world, OP_TYPE_RECV);

	/*
	 *   event  : RECV_EV                           value  : EVT_BEGIN    
	 *   target : MPI_ANY_SOURCE or sender          size   : receive buffer size    
	 *   tag    : MPI_ANY_TAG or message tag        commid : communicator identifier
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_RECV_EV, EVT_BEGIN, source_world, size, c_tag, c_comm, EMPTY);

	f_status_ptr = (status == MPI_F_STATUS_IGNORE) ? f_status : status;

	CtoF77 (pmpi_recv) (buf, count, datatype, source, tag, comm, f_status_ptr, ierror);

	PMPI_Status_f2c (f_status_ptr, &c_status);
	getCommDataFromStatus (&c_status, c_type, c_comm, MPI_GROUP_NULL, &size, &source_tag, &source_world);

	/*
	 *   event  : RECV_EV                           value  : EVT_END
	 *   target : sender rank                       size   : received message size    
	 *   tag    : message tag                       commid : communicator identifier
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_RECV_EV, EVT_END, source_world, size, source_tag, c_comm, EMPTY);

	// MPI stats
	updateStats_P2P(global_mpi_stats, source_world, size, 0);
}


/******************************************************************************
 ***  PMPI_IRecv_Wrapper
 ******************************************************************************/

void PMPI_IRecv_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
                         MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
                         MPI_Fint *ierror)
{
	MPI_Datatype c_type       = PMPI_Type_f2c(*datatype);
	int          size         = 0;
	int          source_world = MPI_ANY_SOURCE;
	int          c_tag        = *tag;
	MPI_Comm     c_comm       = PMPI_Comm_f2c(*comm);
	MPI_Request  c_request    = MPI_REQUEST_NULL;

	size = getMsgSizeFromCountAndDatatype (*count, c_type);

	translateLocalToGlobalRank (c_comm, MPI_GROUP_NULL, *source, &source_world, OP_TYPE_RECV);

	/*
	 *   event  : IRECV_EV                          value  : EVT_BEGIN
	 *   target : MPI_ANY_SOURCE or sender          size   : receive buffer size
	 *   tag    : MPI_ANY_TAG or message tag        commid : communicator identifier
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IRECV_EV, EVT_BEGIN, source_world, size, c_tag, c_comm, EMPTY);

	CtoF77 (pmpi_irecv) (buf, count, datatype, source, tag, comm, request, ierror);

	c_request = PMPI_Request_f2c(*request);
	SaveRequest(c_request, c_comm);

	/*
	 *   event  : IRECV_EV                          value  : EVT_END
	 *   target : MPI_ANY_SOURCE or sender          size   : receive buffer size
	 *   tag    : MPI_ANY_TAG or message tag        commid : communicator identifier
	 *   aux    : request id
	 */
	TRACE_MPIEVENT (TIME, MPI_IRECV_EV, EVT_END, source_world, size, c_tag, c_comm, c_request);
}


#if defined(MPI3)

/******************************************************************************
 ***  PMPI_Mrecv_Wrapper
 ******************************************************************************/

void PMPI_Mrecv_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
                         MPI_Fint *message, MPI_Fint *status, MPI_Fint *ierror)
{
	MPI_Datatype c_type        = PMPI_Type_f2c(*datatype);
	int          size          = 0;
	int          source_world  = MPI_ANY_SOURCE;
	int          source_tag    = MPI_ANY_TAG;
	MPI_Comm     c_comm        = MPI_COMM_NULL;
	MPI_Message  c_save_message     = PMPI_Message_f2c(*message); // Save input value as it changes inside the PMPI
	MPI_Fint     f_status[SIZEOF_MPI_STATUS];
	MPI_Fint    *f_status_ptr  = NULL;
	MPI_Status   c_status;

	size = getMsgSizeFromCountAndDatatype (*count, c_type);

	/*
	 *   event  : MRECV_EV                          value  : EVT_BEGIN    
	 *   target : --- (source not avail)            size   : receive buffer size    
	 *   tag    : --- (tag not avail)               commid : --- (comm not avail)
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_MRECV_EV, EVT_BEGIN, EMPTY, size, EMPTY, EMPTY, EMPTY);

	f_status_ptr = (status == MPI_F_STATUS_IGNORE) ? f_status : status;

	CtoF77 (pmpi_mrecv) (buf, count, datatype, message, f_status_ptr, ierror);

	c_comm = ProcessMessage (c_save_message, NULL);

	PMPI_Status_f2c (f_status_ptr, &c_status);
	getCommDataFromStatus (&c_status, c_type, c_comm, MPI_GROUP_NULL, &size, &source_tag, &source_world);

	/*
	 *   event  : MRECV_EV                          value  : EVT_END
	 *   target : sender rank                       size   : received message size    
	 *   tag    : message tag                       commid : communicator identifier
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_MRECV_EV, EVT_END, source_world, size, source_tag, c_comm, EMPTY);

	// MPI stats 
	updateStats_P2P(global_mpi_stats, source_world, size, 0);
}


/******************************************************************************
 ***  PMPI_Imrecv_Wrapper
 ******************************************************************************/

void PMPI_Imrecv_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
                          MPI_Fint *message, MPI_Fint *request, MPI_Fint *ierror)
{
	MPI_Datatype c_type         = PMPI_Type_f2c(*datatype);
	int          size           = 0;
	MPI_Comm     c_comm         = MPI_COMM_NULL;
	MPI_Message  c_save_message = PMPI_Message_f2c(*message); // Save input value as it changes inside the PMPI
        MPI_Request  c_request      = MPI_REQUEST_NULL;

	size = getMsgSizeFromCountAndDatatype (*count, c_type);

	/*
	 *   event  : IMRECV_EV                         value  : EVT_BEGIN
	 *   target : --- (source not avail)            size   : receive buffer size
	 *   tag    : --- (tag not avail)               commid : --- (comm not avail)   
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IMRECV_EV, EVT_BEGIN, EMPTY, size, EMPTY, EMPTY, EMPTY);

	CtoF77 (pmpi_imrecv) (buf, count, datatype, message, request, ierror);

	c_request = PMPI_Request_f2c(*request);
	c_comm = ProcessMessage(c_save_message, &c_request);

	/*
	 *   event  : IMRECV_EV                         value  : EVT_END
	 *   target : --- (source not avail)            size   : received message size
	 *   tag    : --- (tag not avail)               commid : communicator identifier
	 *   aux    : request id
	 */
	TRACE_MPIEVENT (TIME, MPI_IMRECV_EV, EVT_END, EMPTY, size, EMPTY, c_comm, c_request);
}

#endif /* MPI3 */


/******************************************************************************
 ***  PMPI_Probe_Wrapper
 ******************************************************************************/

void PMPI_Probe_Wrapper (MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm,
                         MPI_Fint *status, MPI_Fint *ierror)
{
	MPI_Comm c_comm = PMPI_Comm_f2c(*comm);

	/*
	 *   event  : PROBE_EV                          value  : EVT_BEGIN
	 *   target : ---                               size   : ---
	 *   tag    : ---                               commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_PROBE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, c_comm, EMPTY);

	CtoF77 (pmpi_probe) (source, tag, comm, status, ierror);

	/*
	 *   event  : PROBE_EV                          value  : EVT_END
	 *   target : ---                               size   : ---
	 *   tag    : ---                               commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_PROBE_EV, EVT_END, EMPTY, EMPTY, EMPTY, c_comm, EMPTY);

	// MPI stats
	updateStats_OTHER(global_mpi_stats);
}


/******************************************************************************
 ***  PMPI_IProbe_Wrapper
 ******************************************************************************/

void Bursts_PMPI_IProbe_Wrapper (MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm,
                                 MPI_Fint *flag, MPI_Fint *status, MPI_Fint *ierror)
{
	MPI_Comm c_comm = PMPI_Comm_f2c(*comm);

	/*
	 *   event  : IPROBE_EV                         value  : EVT_BEGIN
	 *   target : ---                               size   : ---
	 *   tag    : ---                               commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IPROBE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, c_comm, EMPTY);

	CtoF77 (pmpi_iprobe) (source, tag, comm, flag, status, ierror);

	/*
	 *   event  : IPROBE_EV                         value  : EVT_END
	 *   target : ---                               size   : ---
	 *   tag    : ---                               commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_IPROBE_EV, EVT_END, EMPTY, EMPTY, EMPTY, c_comm, EMPTY);
}

void Normal_PMPI_IProbe_Wrapper (MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, 
                                 MPI_Fint *flag, MPI_Fint *status, MPI_Fint *ierror)
{
	static int       mpi_iprobe_software_counter = 0;
	static iotimer_t mpi_iprobe_elapsed_time     = 0;
	iotimer_t        mpi_iprobe_begin_time       = 0;
	MPI_Comm         c_comm                      = PMPI_Comm_f2c(*comm);

	mpi_iprobe_begin_time = LAST_READ_TIME;

	CtoF77 (pmpi_iprobe) (source, tag, comm, flag, status, ierror);

	if (*flag)
	{
		// mpi_iprobe was successful

		if (mpi_iprobe_software_counter > 0)
		{
			// Only emit software counters if there were previous calls to MPI_Iprobe omitted
			TRACE_EVENT (mpi_iprobe_begin_time, MPI_TIME_IN_IPROBE_EV, mpi_iprobe_elapsed_time);
			TRACE_EVENT (mpi_iprobe_begin_time, MPI_IPROBE_COUNTER_EV, mpi_iprobe_software_counter);
		}

		// The successful mpi_iprobe is marked on the trace
		TRACE_MPIEVENT (mpi_iprobe_begin_time, MPI_IPROBE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, c_comm, EMPTY);
		TRACE_MPIEVENT (TIME, MPI_IPROBE_EV, EVT_END, EMPTY, EMPTY, EMPTY, c_comm, EMPTY);

		mpi_iprobe_software_counter = 0;
		mpi_iprobe_elapsed_time = 0;
	}
	else
	{
		// mpi_iprobe was unsuccessful

		if (mpi_iprobe_software_counter == 0)
		{
			// Mark the first unsuccessful MPI_Iprobe
			TRACE_EVENTANDCOUNTERS (mpi_iprobe_begin_time, MPI_IPROBE_COUNTER_EV, 0, TRUE);
		}

		mpi_iprobe_software_counter ++;
		mpi_iprobe_elapsed_time += (TIME - mpi_iprobe_begin_time);
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

	// MPI stats
	updateStats_OTHER(global_mpi_stats);
}


#if defined(MPI3)

/******************************************************************************
 ***  PMPI_Mprobe_Wrapper
 ******************************************************************************/

void PMPI_Mprobe_Wrapper (MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm,
                          MPI_Fint *message, MPI_Fint *status, MPI_Fint *ierror)
{
	MPI_Comm    c_comm    = PMPI_Comm_f2c(*comm);
	MPI_Message c_message = PMPI_Message_f2c(*message);

	/*
	 *   event  : MPROBE_EV                         value  : EVT_BEGIN
	 *   target : ---                               size   : ---
	 *   tag    : ---                               commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_MPROBE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, c_comm, EMPTY);

	CtoF77 (pmpi_mprobe) (source, tag, comm, message, status, ierror);

	SaveMessage (c_message, c_comm);

	/*
	 *   event  : MPROBE_EV                         value  : EVT_END
	 *   target : ---                               size   : ---
	 *   tag    : ---                               commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_MPROBE_EV, EVT_END, EMPTY, EMPTY, EMPTY, c_comm, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}


/******************************************************************************
 ***  PMPI_Improbe_Wrapper
 ******************************************************************************/

void Bursts_PMPI_Improbe_Wrapper (MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm,
                                  MPI_Fint *flag, MPI_Fint *message, MPI_Fint *status, 
                                  MPI_Fint *ierror)
{
	MPI_Comm    c_comm    = PMPI_Comm_f2c(*comm);
        MPI_Message c_message = PMPI_Message_f2c(*message);

	/*
	 *   event  : IMPROBE_EV                        value :  EVT_BEGIN
	 *   target : ---                               size   : ---
	 *   tag    : ---                               commid : communicator id
	 *   aux    : --- 
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_IMPROBE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, c_comm, EMPTY);

	CtoF77 (pmpi_improbe) (source, tag, comm, flag, message, status, ierror);

	SaveMessage (c_message, c_comm);

	/*
	 *   event  : IMPROBE_EV                        value  : EVT_END
	 *   target : ---                               size   : ---
	 *   tag    : ---                               commid : communicator id
	 *   aux    : --- 
	 */
        TRACE_MPIEVENT (TIME, MPI_IMPROBE_EV, EVT_END, EMPTY, EMPTY, EMPTY, c_comm, EMPTY);
}

void Normal_PMPI_Improbe_Wrapper (MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm,
                                  MPI_Fint *flag, MPI_Fint *message, MPI_Fint *status, 
                                  MPI_Fint *ierror)
{
	static int       mpi_improbe_software_counter = 0;
	static iotimer_t mpi_improbe_elapsed_time     = 0;
	iotimer_t        mpi_improbe_begin_time       = 0;
	MPI_Comm         c_comm                       = PMPI_Comm_f2c(*comm);
        MPI_Message      c_message                    = PMPI_Message_f2c(*message);

	mpi_improbe_begin_time = LAST_READ_TIME;

	CtoF77 (pmpi_improbe) (source, tag, comm, flag, message, status, ierror);
	
	if (*flag)
	{
		// mpi_improbe was successful

		SaveMessage (c_message, c_comm);

		if (mpi_improbe_software_counter > 0)
		{
			// Only emit software counters if there were previous calls to MPI_Iprobe omitted
			TRACE_EVENT (mpi_improbe_begin_time, MPI_TIME_IN_IMPROBE_EV, mpi_improbe_elapsed_time);
			TRACE_EVENT (mpi_improbe_begin_time, MPI_IMPROBE_COUNTER_EV, mpi_improbe_software_counter);
		}

		// The successful mpi_improbe is marked on the trace
		TRACE_MPIEVENT (mpi_improbe_begin_time, MPI_IMPROBE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, c_comm, EMPTY);
		TRACE_MPIEVENT (TIME, MPI_IMPROBE_EV, EVT_END, EMPTY, EMPTY, EMPTY, c_comm, EMPTY);

		mpi_improbe_software_counter = 0;
		mpi_improbe_elapsed_time = 0;
	}
	else
	{
		// mpi_improbe was unsuccessful

		if (mpi_improbe_software_counter == 0)
		{
			// Mark the first unsuccessful mpi_improbe
			TRACE_EVENTANDCOUNTERS (mpi_improbe_begin_time, MPI_IMPROBE_COUNTER_EV, 0, TRUE);
		}

		mpi_improbe_software_counter ++;
		mpi_improbe_elapsed_time += (TIME - mpi_improbe_begin_time);
	}
}

void PMPI_Improbe_Wrapper (MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm,
                           MPI_Fint *flag, MPI_Fint *message, MPI_Fint *status, 
                           MPI_Fint *ierror)
{
	if (CURRENT_TRACE_MODE(THREADID) == TRACE_MODE_BURSTS)
	{
		Bursts_PMPI_Improbe_Wrapper (source, tag, comm, flag, message, status, ierror);
	}
        else
	{
		Normal_PMPI_Improbe_Wrapper (source, tag, comm, flag, message, status, ierror);
	}
}

#endif /* MPI3 */


/******************************************************************************
 ***  PMPI_Test_Wrapper
 ******************************************************************************/

void Bursts_PMPI_Test_Wrapper (MPI_Fint *request, MPI_Fint *flag, MPI_Fint *status,
                               MPI_Fint *ierror)
{
	iotimer_t   mpi_test_end_time = 0;
	MPI_Request c_request         = PMPI_Request_f2c (*request);

	/*
	 *   event  : TEST_EV                    value  : EVT_BEGIN
	 *   target : request id                 size   : ---
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_TEST_EV, EVT_BEGIN, c_request, EMPTY, EMPTY, EMPTY, EMPTY);

	CtoF77 (pmpi_test) (request, flag, status, ierror);

	mpi_test_end_time = TIME;

	if (*flag)
	{
		MPI_Status c_status;
		PMPI_Status_f2c (status, &c_status);
		ProcessRequest (mpi_test_end_time, c_request, &c_status);
	}

	/*
	 *   event  : TEST_EV                    value  : EVT_END
	 *   target : ---                        size   : ---
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (mpi_test_end_time, MPI_TEST_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
}

void Normal_PMPI_Test_Wrapper (MPI_Fint *request, MPI_Fint *flag, MPI_Fint *status,
                               MPI_Fint *ierror)
{
	static int       mpi_test_software_counter = 0;
	static iotimer_t mpi_test_elapsed_time     = 0;
	iotimer_t        mpi_test_begin_time       = 0;
	iotimer_t        mpi_test_end_time         = 0;
	MPI_Request      c_request                 = PMPI_Request_f2c(*request);

	mpi_test_begin_time = LAST_READ_TIME;

	CtoF77 (pmpi_test) (request, flag, status, ierror);

	if (*flag)
	{
		// mpi_test was successful 
		MPI_Status c_status;
		PMPI_Status_f2c (status, &c_status);

		if (mpi_test_software_counter > 0)
		{
			// Only emit software counters if there were previous calls to mpi_test omitted
			TRACE_EVENT (mpi_test_begin_time, MPI_TIME_IN_TEST_EV, mpi_test_elapsed_time);
			TRACE_EVENT (mpi_test_begin_time, MPI_TEST_COUNTER_EV, mpi_test_software_counter);
		}
		// The successful mpi_test is marked in the trace
		TRACE_MPIEVENT (mpi_test_begin_time, MPI_TEST_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

		mpi_test_end_time = TIME;

		ProcessRequest (mpi_test_end_time, c_request, &c_status);		

		TRACE_MPIEVENT (mpi_test_end_time, MPI_TEST_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

		mpi_test_software_counter = 0;
		mpi_test_elapsed_time = 0;
	}
	else
	{
		// mpi_test was unsuccessful -- accumulate software counters
		if (mpi_test_software_counter == 0)
		{
			// Mark the first unsuccessful mpi_test
			TRACE_EVENTANDCOUNTERS (mpi_test_begin_time, MPI_TEST_COUNTER_EV, 0, TRUE);
		}
		mpi_test_software_counter ++;
		mpi_test_elapsed_time += (TIME - mpi_test_begin_time);
	}
}

void PMPI_Test_Wrapper (MPI_Fint *request, MPI_Fint *flag, MPI_Fint *status,
                        MPI_Fint *ierror)
{
	MPI_Fint  f_status[SIZEOF_MPI_STATUS];
	MPI_Fint *f_status_ptr = NULL;

	f_status_ptr = (MPI_F_STATUS_IGNORE == status) ? f_status : status;

	if (CURRENT_TRACE_MODE(THREADID) == TRACE_MODE_BURSTS)
	{
		Bursts_PMPI_Test_Wrapper(request, flag, f_status_ptr, ierror);
	}
	else
	{
		Normal_PMPI_Test_Wrapper(request, flag, f_status_ptr, ierror);
	}
}

void copyRequests_F (int count, MPI_Fint *array_of_requests, MPI_Request *copy, char *where)
{
	int i = 0;

	if (count > MAX_WAIT_REQUESTS)
	{
		fprintf (stderr, "PANIC! Number of requests in %s (%d) exceeds tha maximum supported (%d). Please increase the value of MAX_WAIT_REQUESTS and recompile Extrae.\n", where, count, MAX_WAIT_REQUESTS);
	}
	else
	{
		for (i = 0; i < count; i ++)
		{
			copy[i] = PMPI_Request_f2c(array_of_requests[i]);
		}
	}
}

/******************************************************************************
 ***  PMPI_TestAll_Wrapper
 ******************************************************************************/

void PMPI_TestAll_Wrapper (MPI_Fint *count, MPI_Fint array_of_requests[], MPI_Fint *flag,
                           MPI_Fint array_of_statuses[][SIZEOF_MPI_STATUS], MPI_Fint * ierror)
{
	static int       mpi_testall_software_counter = 0;
	static iotimer_t mpi_testall_elapsed_time     = 0;
	iotimer_t        mpi_testall_begin_time       = 0;
	iotimer_t        mpi_testall_end_time         = 0;
	MPI_Fint         f_statuses[MAX_WAIT_REQUESTS][SIZEOF_MPI_STATUS];
	MPI_Fint        *f_statuses_ptr               = (MPI_Fint *)(MPI_F_STATUSES_IGNORE == (MPI_Fint *)array_of_statuses) ? f_statuses : array_of_statuses;
	MPI_Request      c_save_requests[MAX_WAIT_REQUESTS];

	mpi_testall_begin_time = LAST_READ_TIME;

	copyRequests_F(*count, array_of_requests, c_save_requests, "mpi_testall");

	CtoF77 (pmpi_testall) (count, array_of_requests, flag, f_statuses_ptr, ierror);

	if (*ierror == MPI_SUCCESS && *flag)
	{
		int i = 0;

		// mpi_testall was successful

		if (mpi_testall_software_counter > 0)
		{
			// Only emit software counters if there were previous calls to mpi_testall omitted
			TRACE_EVENT (mpi_testall_begin_time, MPI_TIME_IN_TEST_EV, mpi_testall_elapsed_time);
			TRACE_EVENT (mpi_testall_begin_time, MPI_TEST_COUNTER_EV, mpi_testall_software_counter);
		}
		
		// The successful mpi_testall is marked in the trace
		TRACE_MPIEVENT (mpi_testall_begin_time, MPI_TESTALL_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
		
		mpi_testall_end_time = TIME;

		for (i = 0; i < *count; i ++)
		{
			MPI_Status c_status;
			PMPI_Status_f2c (&f_statuses_ptr[i * SIZEOF_MPI_STATUS], &c_status);
			ProcessRequest (mpi_testall_end_time, c_save_requests[i], &c_status);
		}

		TRACE_MPIEVENT (mpi_testall_end_time, MPI_TESTALL_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

		mpi_testall_software_counter = 0;
		mpi_testall_elapsed_time = 0;
	}
	else
	{
		// mpi_testall was unsuccessful -- accumulate software counters
		if (mpi_testall_software_counter == 0)
		{
			// Mark the first unsuccessful mpi_testall
			TRACE_EVENTANDCOUNTERS (mpi_testall_begin_time, MPI_TEST_COUNTER_EV, 0, TRUE);
		}
		mpi_testall_software_counter ++;
		mpi_testall_elapsed_time += (TIME - mpi_testall_begin_time);
	}
}


/******************************************************************************
 ***  PMPI_TestAny_Wrapper
 ******************************************************************************/

void PMPI_TestAny_Wrapper (MPI_Fint *count, MPI_Fint array_of_requests[],
                           MPI_Fint *index, MPI_Fint *flag, MPI_Fint *status, 
                           MPI_Fint *ierror)
{
	static int       mpi_testany_software_counter = 0;
	static iotimer_t mpi_testany_elapsed_time     = 0;
	iotimer_t        mpi_testany_begin_time       = 0;
	iotimer_t        mpi_testany_end_time         = 0;
	MPI_Fint         f_status[SIZEOF_MPI_STATUS];
	MPI_Fint        *f_status_ptr                 = NULL;
	MPI_Request      c_save_requests[MAX_WAIT_REQUESTS];

	mpi_testany_begin_time = LAST_READ_TIME;

        copyRequests_F(*count, array_of_requests, c_save_requests, "mpi_testany");

	f_status_ptr = (MPI_F_STATUS_IGNORE == status)?f_status:status;

	CtoF77 (pmpi_testany) (count, array_of_requests, index, flag, f_status_ptr, ierror);

	if (*index != MPI_UNDEFINED && *ierror == MPI_SUCCESS && *flag)
	{
		// mpi_testany was successful
		MPI_Status c_status;
		PMPI_Status_f2c (f_status_ptr, &c_status);

		if (mpi_testany_software_counter > 0)
		{
			TRACE_EVENT (mpi_testany_begin_time, MPI_TIME_IN_TEST_EV, mpi_testany_elapsed_time);
			TRACE_EVENT (mpi_testany_begin_time, MPI_TEST_COUNTER_EV, mpi_testany_software_counter);
		}

		// The successful mpi_testany is marked in the trace
		TRACE_MPIEVENT (mpi_testany_begin_time, MPI_TESTANY_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

		mpi_testany_end_time = TIME;

		MPI_Request c_request = c_save_requests[*index-1];

		ProcessRequest (mpi_testany_end_time, c_request, &c_status);

		TRACE_MPIEVENT (mpi_testany_end_time, MPI_TESTANY_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

		mpi_testany_software_counter = 0;
		mpi_testany_elapsed_time = 0;
	}
	else
	{
		// mpi_testany was unsuccessful -- accumulate software counters
		if (mpi_testany_software_counter == 0)
		{
			// Mark the first unsuccessful mpi_testany
			TRACE_EVENTANDCOUNTERS (mpi_testany_begin_time, MPI_TEST_COUNTER_EV, 0, TRUE);
		}
		mpi_testany_software_counter ++;
		mpi_testany_elapsed_time += (TIME - mpi_testany_begin_time);
	}
}

/*****************************************************************************
 ***  PMPI_TestSome_Wrapper
 ******************************************************************************/

void PMPI_TestSome_Wrapper (MPI_Fint *incount, MPI_Fint array_of_requests[],
                            MPI_Fint *outcount, MPI_Fint array_of_indices[],
                            MPI_Fint array_of_statuses[][SIZEOF_MPI_STATUS], MPI_Fint *ierror)
{
	static int       mpi_testsome_software_counter = 0;
	static iotimer_t mpi_testsome_elapsed_time     = 0;
	iotimer_t        mpi_testsome_begin_time       = 0;
	iotimer_t        mpi_testsome_end_time         = 0;
	MPI_Fint         f_statuses[MAX_WAIT_REQUESTS][SIZEOF_MPI_STATUS];
	MPI_Fint        *f_statuses_ptr                = (MPI_F_STATUSES_IGNORE == (MPI_Fint *) array_of_statuses) ? f_statuses : array_of_statuses;
	MPI_Request      c_save_requests[MAX_WAIT_REQUESTS];

        mpi_testsome_begin_time = LAST_READ_TIME;

        copyRequests_F(*incount, array_of_requests, c_save_requests, "mpi_testsome");

	CtoF77(pmpi_testsome) (incount, array_of_requests, outcount, array_of_indices,
	  f_statuses_ptr, ierror);

	if (*ierror == MPI_SUCCESS && *outcount > 0)
	{
		// mpi_testsome was successful 
		int i = 0;

		if (mpi_testsome_software_counter > 0)
		{
			// Only emit software counters if there were previous calls to mpi_testsome omitted
			TRACE_EVENT (mpi_testsome_begin_time, MPI_TIME_IN_TEST_EV, mpi_testsome_elapsed_time);
			TRACE_EVENT (mpi_testsome_begin_time, MPI_TEST_COUNTER_EV, mpi_testsome_software_counter);
		}
	
		// The successful mpi_testsome is marked in the trace
		TRACE_MPIEVENT (mpi_testsome_begin_time, MPI_TESTSOME_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

		mpi_testsome_end_time = TIME;

		for (i = 0; i < *outcount; i++)
		{
			MPI_Request c_request = c_save_requests[array_of_indices[i]];
			MPI_Status  c_status;
			PMPI_Status_f2c (&f_statuses_ptr[i*SIZEOF_MPI_STATUS], &c_status);

			ProcessRequest (mpi_testsome_end_time, c_request, &c_status);
		}

		TRACE_MPIEVENT (mpi_testsome_end_time, MPI_TESTSOME_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
	
		mpi_testsome_software_counter = 0;
		mpi_testsome_elapsed_time = 0;
	}
	else
	{
		// mpi_testsome was unsuccessful -- accumulate software counters
		if (mpi_testsome_software_counter == 0)
		{
			// Mark the first unsuccessful mpi_testsome
			TRACE_EVENTANDCOUNTERS (mpi_testsome_begin_time, MPI_TEST_COUNTER_EV, 0, TRUE);
		}
		mpi_testsome_software_counter ++;
		mpi_testsome_elapsed_time += (TIME - mpi_testsome_begin_time);
	}
}


/******************************************************************************
 ***  PMPI_Wait_Wrapper
 ******************************************************************************/

void PMPI_Wait_Wrapper (MPI_Fint *request, MPI_Fint *status, MPI_Fint *ierror)
{
	MPI_Request c_request;
	MPI_Fint    f_status[SIZEOF_MPI_STATUS];
	MPI_Fint   *f_status_ptr      = NULL;
	iotimer_t   mpi_wait_end_time = 0;

	/*
	 *   event  : WAIT_EV                    value  : EVT_BEGIN
	 *   target : request id                 size   : ---
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_WAIT_EV, EVT_BEGIN, *request, EMPTY, EMPTY, EMPTY, EMPTY);

	copyRequests_F (1, request, &c_request, "mpì_wait");

	f_status_ptr = (MPI_F_STATUS_IGNORE == status) ? f_status : status;

	CtoF77 (pmpi_wait) (request, f_status_ptr, ierror);

	mpi_wait_end_time = TIME;

	if (*ierror == MPI_SUCCESS)
	{
		MPI_Status c_status;
		PMPI_Status_f2c (f_status_ptr, &c_status);
		ProcessRequest (mpi_wait_end_time, c_request, &c_status);
	}

	/*
	 *   event  : WAIT_EV                    value  : EVT_END
	 *   target : ---                        size   : ---
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (mpi_wait_end_time, MPI_WAIT_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
}

/******************************************************************************
 ***  PMPI_WaitAll_Wrapper
 ******************************************************************************/

void PMPI_WaitAll_Wrapper (MPI_Fint *count, MPI_Fint array_of_requests[],
	MPI_Fint array_of_statuses[][SIZEOF_MPI_STATUS], MPI_Fint * ierror)
{
	MPI_Request c_save_requests[MAX_WAIT_REQUESTS];
	MPI_Fint    f_statuses[MAX_WAIT_REQUESTS][SIZEOF_MPI_STATUS];
	MPI_Fint   *f_statuses_ptr       = (MPI_F_STATUSES_IGNORE == (MPI_Fint *)array_of_statuses) ? f_statuses : array_of_statuses;
	iotimer_t   mpi_waitall_end_time = 0;

	/*
	 *   event  : WAITALL_EV                 value  : EVT_BEGIN
	 *   target : ---                        size   : ---
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_WAITALL_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

        copyRequests_F(*count, array_of_requests, c_save_requests, "mpi_waitall");

	CtoF77 (pmpi_waitall) (count, array_of_requests, f_statuses_ptr, ierror);

	mpi_waitall_end_time = TIME;

	if (*ierror == MPI_SUCCESS)
	{
		int i = 0;

		for (i = 0; i < *count; i ++)
		{
			MPI_Status c_status;
			PMPI_Status_f2c (&f_statuses_ptr[i * SIZEOF_MPI_STATUS], &c_status);
			ProcessRequest (mpi_waitall_end_time, c_save_requests[i], &c_status);
		}
	}

	/*
	 *   event  : WAITATLL_EV                value  : EVT_END
	 *   target : ---                        size   : ---
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (mpi_waitall_end_time, MPI_WAITALL_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
}


/******************************************************************************
 ***  PMPI_WaitAny_Wrapper
 ******************************************************************************/

void PMPI_WaitAny_Wrapper (MPI_Fint *count, MPI_Fint array_of_requests[],
	MPI_Fint *index, MPI_Fint *status, MPI_Fint *ierror)
{
	MPI_Request c_save_requests[MAX_WAIT_REQUESTS];
	MPI_Fint    f_status[SIZEOF_MPI_STATUS];
	MPI_Fint   *f_status_ptr         = NULL;
	iotimer_t   mpi_waitany_end_time = 0;

	/*
	 *   event  : WAITANY_EV                 value  : EVT_BEGIN
	 *   target : ---                        size   : ---
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_WAITANY_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

        copyRequests_F(*count, array_of_requests, c_save_requests, "mpi_waitany");

	f_status_ptr = (MPI_F_STATUS_IGNORE == status) ? f_status : status;

	CtoF77 (pmpi_waitany) (count, array_of_requests, index, f_status_ptr, ierror);

	mpi_waitany_end_time = TIME;

	if (*index != MPI_UNDEFINED && *ierror == MPI_SUCCESS)
	{
		MPI_Request c_request = c_save_requests[*index-1];
		MPI_Status  c_status;
		PMPI_Status_f2c (f_status_ptr, &c_status);

		ProcessRequest (mpi_waitany_end_time, c_request, &c_status);
	}

        /*
         *   event  : WAITANY_EV                 value  : EVT_END
         *   target : ---                        size   : ---
         *   tag    : ---                        commid : ---
         *   aux    : ---
         */
	TRACE_MPIEVENT (mpi_waitany_end_time, MPI_WAITANY_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
}

/*****************************************************************************
 ***  PMPI_WaitSome_Wrapper
 ******************************************************************************/

void PMPI_WaitSome_Wrapper (MPI_Fint *incount, MPI_Fint array_of_requests[],
	MPI_Fint *outcount, MPI_Fint array_of_indices[],
	MPI_Fint array_of_statuses[][SIZEOF_MPI_STATUS], MPI_Fint *ierror)
{
	MPI_Request c_save_requests[MAX_WAIT_REQUESTS];
	MPI_Fint    f_statuses[MAX_WAIT_REQUESTS][SIZEOF_MPI_STATUS];
	MPI_Fint   *f_statuses_ptr        = (MPI_F_STATUSES_IGNORE == (MPI_Fint *)array_of_statuses) ? f_statuses : array_of_statuses;
	iotimer_t   mpi_waitsome_end_time = 0;

	/*
	 *   event  : WAITSOME_EV                value  : EVT_BEGIN
	 *   target : ---                        size   : ---
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_WAITSOME_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

        copyRequests_F(*incount, array_of_requests, c_save_requests, "mpi_waitsome");

	CtoF77(pmpi_waitsome) (incount, array_of_requests, outcount, array_of_indices, f_statuses_ptr, ierror);

	mpi_waitsome_end_time = TIME;

	if (*ierror == MPI_SUCCESS)
	{
		int i = 0;

		for (i = 0; i < *outcount; i++)
		{
			MPI_Request c_request = c_save_requests[array_of_indices[i]];
			MPI_Status  c_status;
			PMPI_Status_f2c (&f_statuses_ptr[i * SIZEOF_MPI_STATUS], &c_status);
			ProcessRequest (mpi_waitsome_end_time, c_request, &c_status);
		}
	}

	/*
	 *   event  : WAITSOME_EV                value  : EVT_END
	 *   target : ---                        size   : ---
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (mpi_waitsome_end_time, MPI_WAITSOME_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
}

/******************************************************************************
 ***  PMPI_Recv_init_Wrapper
 ******************************************************************************/

void PMPI_Recv_init_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)
{
	MPI_Datatype c_type = PMPI_Type_f2c (*datatype);
	MPI_Comm     c_comm = PMPI_Comm_f2c (*comm);
	MPI_Request  c_request;

	/*
	 *   event  : RECV_INIT_EV               value  : EVT_BEGIN
	 *   target : ---                        size   : ----
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_RECV_INIT_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	// First call pmpi_recv_init to initialize the persistent request
	CtoF77 (pmpi_recv_init) (buf, count, datatype, source, tag, comm, request, ierror); 

	// Save this persistent request
	c_request = PMPI_Request_f2c (*request);
	PR_NewRequest (MPI_IRECV_EV, *count, c_type, *source, *tag, c_comm, c_request, &PR_queue);

	/*
	 *   event  : RECV_INIT_EV               value  : EVT_END
	 *   target : ---                        size   : ----
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_RECV_INIT_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}



/******************************************************************************
 ***  PMPI_Send_init_Wrapper
 ******************************************************************************/

void PMPI_Send_init_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)
{
	MPI_Datatype c_type = PMPI_Type_f2c(*datatype);
	MPI_Comm     c_comm = PMPI_Comm_f2c(*comm);
	MPI_Request  c_request;

	/*
	 *   event  : SEND_INIT_EV               value  : EVT_BEGIN
	 *   target : ---                        size   : ----
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_SEND_INIT_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	// First call pmpi_send_init to initialize the persistent request
	CtoF77 (pmpi_send_init) (buf, count, datatype, dest, tag, comm, request, ierror);

	// Save this persistent request
	c_request = PMPI_Request_f2c (*request);
	PR_NewRequest (MPI_ISEND_EV, *count, c_type, *dest, *tag, c_comm, c_request, &PR_queue);

	/*
	 *   event  : SEND_INIT_EV               value  : EVT_END
	 *   target : ---                        size   : ----
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_SEND_INIT_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}



/******************************************************************************
 ***  PMPI_Bsend_init_Wrapper
 ******************************************************************************/

void PMPI_Bsend_init_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)
{
	MPI_Datatype c_type = PMPI_Type_f2c(*datatype);
	MPI_Comm     c_comm = PMPI_Comm_f2c(*comm);
	MPI_Request  c_request;

	/*
	 *   event  : BSEND_INIT_EV              value  : EVT_BEGIN
	 *   target : ---                        size   : ----
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_BSEND_INIT_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	// First call pmpi_bsend init to initialize the persistent request
	CtoF77 (pmpi_bsend_init) (buf, count, datatype, dest, tag, comm, request, ierror);

	// Save this persistent request 
	c_request = PMPI_Request_f2c (*request);
	PR_NewRequest (MPI_IBSEND_EV, *count, c_type, *dest, *tag, c_comm, c_request, &PR_queue);

	/*
	 *   event  : BSEND_INIT_EV              value  : EVT_END
	 *   target : ---                        size   : ----
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_BSEND_INIT_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}


/******************************************************************************
 ***  PMPI_Rsend_init_Wrapper
 ******************************************************************************/

void PMPI_Rsend_init_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)
{
	MPI_Datatype c_type = PMPI_Type_f2c(*datatype);
	MPI_Comm     c_comm = PMPI_Comm_f2c(*comm);
	MPI_Request  c_request;

	/*
	 *   event  : RSEND_INIT_EV              value  : EVT_BEGIN
	 *   target : ---                        size   : ----
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_RSEND_INIT_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	// First call pmpi_rsend_init to initialize the persistent request
	CtoF77 (pmpi_rsend_init) (buf, count, datatype, dest, tag, comm, request, ierror);

	// Save this persistent request
	c_request = PMPI_Request_f2c (*request);
	PR_NewRequest (MPI_IRSEND_EV, *count, c_type, *dest, *tag, c_comm, c_request, &PR_queue);

	/*
	 *   event  : RSEND_INIT_EV              value  : EVT_END
	 *   target : ---                        size   : ----
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_RSEND_INIT_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}


/******************************************************************************
 ***  PMPI_Ssend_init_Wrapper
 ******************************************************************************/

void PMPI_Ssend_init_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror)
{
	MPI_Datatype c_type = PMPI_Type_f2c(*datatype);
	MPI_Comm     c_comm = PMPI_Comm_f2c(*comm);
	MPI_Request  c_request;

	/*
	 *   event  : SSEND_INIT_EV              value  : EVT_BEGIN
	 *   target : ---                        size   : ----
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_SSEND_INIT_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	// First call pmpi_ssend_init to initialize the persistent request
	CtoF77 (pmpi_ssend_init) (buf, count, datatype, dest, tag, comm, request, ierror);

	// Save this persistent request
	c_request = PMPI_Request_f2c (*request);
	PR_NewRequest (MPI_ISSEND_EV, *count, c_type, *dest, *tag, c_comm, c_request, &PR_queue);

	/*
	 *   event  : SSEND_INIT_EV              value  : EVT_END
	 *   target : ---                        size   : ----
	 *   tag    : ---                        commid : ---
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_SSEND_INIT_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}

void MPI_Sendrecv_Fortran_Wrapper (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, MPI_Fint *dest, MPI_Fint *sendtag, void *recvbuf,
	MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *source, MPI_Fint *recvtag,
	MPI_Fint *comm, MPI_Fint *status, MPI_Fint *ierr) 
{
	MPI_Datatype c_sendtype   = PMPI_Type_f2c (*sendtype);
	MPI_Datatype c_recvtype   = PMPI_Type_f2c (*recvtype);
	MPI_Comm     c_comm       = PMPI_Comm_f2c (*comm);
	MPI_Fint     f_status[SIZEOF_MPI_STATUS];
	MPI_Fint    *f_status_ptr = NULL;
	MPI_Status   c_status;
	int SentSize = 0, ReceivedSize = 0;
	int SenderRank = MPI_ANY_SOURCE, ReceiverRank = MPI_ANY_SOURCE, Tag = MPI_ANY_TAG;

	SentSize = getMsgSizeFromCountAndDatatype (*sendcount, c_sendtype);

	translateLocalToGlobalRank (c_comm, MPI_GROUP_NULL, *dest, &ReceiverRank, OP_TYPE_SEND);

	/*
	 *   event  : SENDRECV_REPLACE_EV        value  : EVT_BEGIN
	 *   target : receiver rank              size   : bytes sent
	 *   tag    : message tag                commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_SENDRECV_EV, EVT_BEGIN, ReceiverRank, SentSize, *sendtag, c_comm, EMPTY);

	f_status_ptr = (MPI_F_STATUS_IGNORE == status) ? f_status : status;

	CtoF77(pmpi_sendrecv) (sendbuf, sendcount, sendtype, dest, sendtag,
	                       recvbuf, recvcount, recvtype, source, recvtag, 
	                       comm, f_status_ptr, ierr);

	PMPI_Status_f2c (f_status_ptr, &c_status);
	getCommDataFromStatus (&c_status, c_recvtype, c_comm, MPI_GROUP_NULL, &ReceivedSize, &Tag, &SenderRank);

	/*
	 *   event  : SENDRECV_REPLACE_EV        value  : EVT_END
	 *   target : sender rank                size   : bytes received
	 *   tag    : message tag                commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_SENDRECV_EV, EVT_END, SenderRank, ReceivedSize, Tag, c_comm, EMPTY);

	// MPI stats
	updateStats_P2P(global_mpi_stats, SenderRank, ReceivedSize, SentSize);
}

void MPI_Sendrecv_replace_Fortran_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *type,
	MPI_Fint *dest, MPI_Fint *sendtag, MPI_Fint *source, MPI_Fint *recvtag,
	MPI_Fint *comm, MPI_Fint *status, MPI_Fint *ierr) 
{
	MPI_Datatype c_type       = PMPI_Type_f2c (*type);
	MPI_Comm     c_comm       = PMPI_Comm_f2c (*comm);
	MPI_Fint     f_status[SIZEOF_MPI_STATUS];
	MPI_Fint    *f_status_ptr = NULL;
	MPI_Status   c_status;
	int SentSize = 0, ReceivedSize = 0;
	int SenderRank = MPI_ANY_SOURCE, ReceiverRank = MPI_ANY_SOURCE, Tag = MPI_ANY_TAG;

	SentSize = getMsgSizeFromCountAndDatatype (*count, c_type);

	translateLocalToGlobalRank (c_comm, MPI_GROUP_NULL, *dest, &ReceiverRank, OP_TYPE_SEND);

	/*
	 *   event  : SENDRECV_REPLACE_EV        value  : EVT_BEGIN
	 *   target : receiver rank              size   : bytes sent
	 *   tag    : message tag                commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_SENDRECV_REPLACE_EV, EVT_BEGIN, ReceiverRank, SentSize, *sendtag, c_comm, EMPTY);

	f_status_ptr = (MPI_F_STATUS_IGNORE == status) ? f_status : status;

	CtoF77(pmpi_sendrecv_replace) (buf, count, type, dest, sendtag, source, recvtag, comm, f_status_ptr, ierr);

	PMPI_Status_f2c (f_status_ptr, &c_status);
	getCommDataFromStatus (&c_status, c_type, c_comm, MPI_GROUP_NULL, &ReceivedSize, &Tag, &SenderRank);

	/*
	 *   event  : SENDRECV_REPLACE_EV        value  : EVT_END
	 *   target : sender rank                size   : bytes received
	 *   tag    : message tag                commid : communicator id
	 *   aux    : ---
	 */
	TRACE_MPIEVENT (TIME, MPI_SENDRECV_REPLACE_EV, EVT_END, SenderRank, ReceivedSize, Tag, c_comm, EMPTY);

	// MPI stats
	updateStats_P2P(global_mpi_stats, SenderRank, ReceivedSize, SentSize);
}

#endif /* defined(FORTRAN_SYMBOLS) */

