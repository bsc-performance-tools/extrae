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

#ifndef MPI_WRAPPER_P2P_F_DEFINED
#define MPI_WRAPPER_P2P_F_DEFINED

#if !defined(MPI_SUPPORT)
# error "This should not be included"
#endif

#include <config.h>

#ifdef HAVE_MPI_H
# include <mpi.h>
#endif
#include "extrae_mpif.h"

/* Fortran Wrappers */

#if defined(FORTRAN_SYMBOLS)

#if (defined(COMBINED_SYMBOLS) && defined(MPI_C_CONTAINS_FORTRAN_MPI_INIT) || \
     !defined(COMBINED_SYMBOLS))
void PMPI_Init_Wrapper (MPI_Fint *ierror);
#endif

void PMPI_BSend_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror);

void PMPI_SSend_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror);

void PMPI_RSend_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror);

void PMPI_Send_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierror);

void PMPI_IBSend_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);

void PMPI_ISend_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);

void PMPI_ISSend_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);

void PMPI_IRSend_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);

void PMPI_Recv_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *status, 
   MPI_Fint *ierror);

void PMPI_IRecv_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);

void PMPI_Probe_Wrapper (MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm,
	MPI_Fint *status, MPI_Fint *ierror);

void PMPI_IProbe_Wrapper (MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm,
	MPI_Fint *flag, MPI_Fint *status, MPI_Fint *ierror);

void PMPI_Test_Wrapper (MPI_Fint *request, MPI_Fint *flag, MPI_Fint *status,
	MPI_Fint *ierror);

void PMPI_TestAll_Wrapper (MPI_Fint * count, MPI_Fint array_of_requests[],
	MPI_Fint *flag, MPI_Fint array_of_statuses[][SIZEOF_MPI_STATUS], MPI_Fint * ierror);

void PMPI_TestAny_Wrapper (MPI_Fint *count, MPI_Fint array_of_requests[],
	MPI_Fint *index, MPI_Fint *flag, MPI_Fint *status, MPI_Fint *ierror);

void PMPI_TestSome_Wrapper (MPI_Fint *incount, MPI_Fint array_of_requests[],
	MPI_Fint *outcount, MPI_Fint array_of_indices[],
	MPI_Fint array_of_statuses[][SIZEOF_MPI_STATUS], MPI_Fint *ierror);

void PMPI_Wait_Wrapper (MPI_Fint *request, MPI_Fint *status, MPI_Fint *ierror);

void PMPI_WaitAll_Wrapper (MPI_Fint * count, MPI_Fint array_of_requests[],
	MPI_Fint array_of_statuses[][SIZEOF_MPI_STATUS], MPI_Fint * ierror);

void PMPI_WaitAny_Wrapper (MPI_Fint *count, MPI_Fint array_of_requests[],
	MPI_Fint *index, MPI_Fint *status, MPI_Fint *ierror);

void PMPI_WaitSome_Wrapper (MPI_Fint *incount, MPI_Fint array_of_requests[],
	MPI_Fint *outcount, MPI_Fint array_of_indices[],
	MPI_Fint array_of_statuses[][SIZEOF_MPI_STATUS], MPI_Fint *ierror);

void PMPI_Recv_init_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);

void PMPI_Send_init_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);

void PMPI_Bsend_init_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);

void PMPI_Rsend_init_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);

void PMPI_Ssend_init_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *datatype,
	MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request,
	MPI_Fint *ierror);

void MPI_Sendrecv_Fortran_Wrapper (void *sendbuf, MPI_Fint *sendcount,
	MPI_Fint *sendtype, MPI_Fint *dest, MPI_Fint *sendtag, void *recvbuf,
	MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *source, MPI_Fint *recvtag,
	MPI_Fint *comm, MPI_Fint *status, MPI_Fint *ierr);

void MPI_Sendrecv_replace_Fortran_Wrapper (void *buf, MPI_Fint *count, MPI_Fint *type,
	MPI_Fint *dest, MPI_Fint *sendtag, MPI_Fint *source, MPI_Fint *recvtag,
	MPI_Fint *comm, MPI_Fint *status, MPI_Fint *ierr);

#endif /* defined(FORTRAN_SYMBOLS) */

#endif /* MPI_WRAPPER_P2P_C_DEFINED */


