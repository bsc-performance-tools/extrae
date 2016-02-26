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

#if HAVE_STDLIB_H
# include <stdlib.h>
#endif
#if HAVE_STDIO_H
# include <stdio.h>
#endif
#if HAVE_STRING_H
# include <string.h>
#endif
#if HAVE_UNISTD_H
# include <unistd.h>
#endif

#include <list>
#include <string>
#include <iostream>
#include <fstream>

using namespace std; 

#include "commonSnippets.h"

struct MPIroutines_t
{
	const char *name;
	const char language;
	BPatch_function *patch;
};

#define MPIROUTINE_Fortran_T(x) \
	{#x, 'F', NULL}
#define MPIROUTINE_F_T(x,y) \
	MPIROUTINE_Fortran_T(x), \
	MPIROUTINE_Fortran_T(x##_), \
	MPIROUTINE_Fortran_T(x##__), \
	MPIROUTINE_Fortran_T(y)

#define MPIROUTINE_C_T(x)  {#x,'C', NULL}
#define MPIROUTINE_C_T_END {NULL,(char)0, NULL}

static struct MPIroutines_t MPIroutines[] =
	{
		MPIROUTINE_C_T(PMPI_Init),
#if defined(MPI_HAS_INIT_THREAD_C)
		MPIROUTINE_C_T(PMPI_Init_thread),
#endif /* MPI_HAS_INIT_THREAD */
		MPIROUTINE_C_T(PMPI_Finalize),
		MPIROUTINE_C_T(PMPI_Bsend),
		MPIROUTINE_C_T(PMPI_Ssend),
		MPIROUTINE_C_T(PMPI_Rsend),
		MPIROUTINE_C_T(PMPI_Send),
		MPIROUTINE_C_T(PMPI_Bsend_init),
		MPIROUTINE_C_T(PMPI_Ssend_init),
		MPIROUTINE_C_T(PMPI_Rsend_init),
		MPIROUTINE_C_T(PMPI_Send_init),
		MPIROUTINE_C_T(PMPI_Ibsend),
		MPIROUTINE_C_T(PMPI_Issend),
		MPIROUTINE_C_T(PMPI_Irsend),
		MPIROUTINE_C_T(PMPI_Isend),
		MPIROUTINE_C_T(PMPI_Recv),
		MPIROUTINE_C_T(PMPI_Irecv),
		MPIROUTINE_C_T(PMPI_Recv_init),
		MPIROUTINE_C_T(PMPI_Reduce),
		MPIROUTINE_C_T(PMPI_Reduce_scatter),
		MPIROUTINE_C_T(PMPI_Allreduce),
		MPIROUTINE_C_T(PMPI_Barrier),
		MPIROUTINE_C_T(PMPI_Cancel),
		MPIROUTINE_C_T(PMPI_Test),
		MPIROUTINE_C_T(PMPI_Wait),
		MPIROUTINE_C_T(PMPI_Waitall),
		MPIROUTINE_C_T(PMPI_Waitany),
		MPIROUTINE_C_T(PMPI_Waitsome),
		MPIROUTINE_C_T(PMPI_Bcast),
		MPIROUTINE_C_T(PMPI_Alltoall),
		MPIROUTINE_C_T(PMPI_Alltoallv),
		MPIROUTINE_C_T(PMPI_Allgather),
		MPIROUTINE_C_T(PMPI_Allgatherv),
		MPIROUTINE_C_T(PMPI_Gather),
		MPIROUTINE_C_T(PMPI_Gatherv),
		MPIROUTINE_C_T(PMPI_Scatter),
		MPIROUTINE_C_T(PMPI_Scatterv),
		MPIROUTINE_C_T(PMPI_Comm_rank),
		MPIROUTINE_C_T(PMPI_Comm_size),
		MPIROUTINE_C_T(PMPI_Comm_create),
		MPIROUTINE_C_T(PMPI_Comm_free),
		MPIROUTINE_C_T(PMPI_Comm_dup),
		MPIROUTINE_C_T(PMPI_Comm_split),
		MPIROUTINE_C_T(PMPI_Cart_create),
		MPIROUTINE_C_T(PMPI_Cart_sub),
		MPIROUTINE_C_T(PMPI_Start),
		MPIROUTINE_C_T(PMPI_Startall),
		MPIROUTINE_C_T(PMPI_Request_free),
		MPIROUTINE_C_T(PMPI_Scan),
		MPIROUTINE_C_T(PMPI_Sendrecv),
		MPIROUTINE_C_T(PMPI_Sendrecv_replace),
		MPIROUTINE_C_T(PMPI_Request_get_status),
#if defined(MPI_SUPPORTS_MPI_IO)
		MPIROUTINE_C_T(PMPI_File_open),
		MPIROUTINE_C_T(PMPI_File_close),
		MPIROUTINE_C_T(PMPI_File_read),
		MPIROUTINE_C_T(PMPI_File_read_all),
		MPIROUTINE_C_T(PMPI_File_write),
		MPIROUTINE_C_T(PMPI_File_write_all),
		MPIROUTINE_C_T(PMPI_File_read_at),
		MPIROUTINE_C_T(PMPI_File_read_at_all),
		MPIROUTINE_C_T(PMPI_File_write_at),
		MPIROUTINE_C_T(PMPI_File_write_at_all),
#endif /* MPI_SUPPORTS_MPI_IO */
		MPIROUTINE_F_T(pmpi_init, PMPI_INIT),
#if defined(MPI_HAS_INIT_THREAD_F)
		MPIROUTINE_F_T(pmpi_init_thread, PMPI_INIT_THREAD),
#endif /* MPI_HAS_INIT_THREAD_F */
		MPIROUTINE_F_T(pmpi_finalize, PMPI_FINALIZE),
		MPIROUTINE_F_T(pmpi_bsend, PMPI_BSEND),
		MPIROUTINE_F_T(pmpi_ssend, PMPI_SSEND),
		MPIROUTINE_F_T(pmpi_rsend, PMPI_RSEND),
		MPIROUTINE_F_T(pmpi_send, PMPI_SEND),
		MPIROUTINE_F_T(pmpi_bsend_init, PMPI_BSEND_INIT),
		MPIROUTINE_F_T(pmpi_ssend_init, PMPI_SSEND_INIT),
		MPIROUTINE_F_T(pmpi_rsend_init, PMPI_RSEND_INIT),
		MPIROUTINE_F_T(pmpi_send_init, PMPI_SEND_INIT),
		MPIROUTINE_F_T(pmpi_ibsend, PMPI_IBSEND),
		MPIROUTINE_F_T(pmpi_issend, PMPI_ISSEND),
		MPIROUTINE_F_T(pmpi_irsend, PMPI_IRSEND),
		MPIROUTINE_F_T(pmpi_isend, PMPI_ISEND),
		MPIROUTINE_F_T(pmpi_recv, PMPI_RECV),
		MPIROUTINE_F_T(pmpi_irecv, PMPI_IRECV),
		MPIROUTINE_F_T(pmpi_recv_init, PMPI_RECV_INIT),
		MPIROUTINE_F_T(pmpi_reduce, PMPI_REDUCE),
		MPIROUTINE_F_T(pmpi_reduce_scatter, PMPI_REDUCE_SCATTER),
		MPIROUTINE_F_T(pmpi_allreduce, PMPI_ALLREDUCE),
		MPIROUTINE_F_T(pmpi_barrier, PMPI_BARRIER),
		MPIROUTINE_F_T(pmpi_cancel, PMPI_CANCEL),
		MPIROUTINE_F_T(pmpi_test, PMPI_TEST),
		MPIROUTINE_F_T(pmpi_wait, PMPI_WAIT),
		MPIROUTINE_F_T(pmpi_waitall, PMPI_WAITALL),
		MPIROUTINE_F_T(pmpi_waitany, PMPI_WAITANY),
		MPIROUTINE_F_T(pmpi_waitsome, PMPI_WAITSOME),
		MPIROUTINE_F_T(pmpi_bcast, PMPI_BCAST),
		MPIROUTINE_F_T(pmpi_alltoall, PMPI_ALLTOALL),
		MPIROUTINE_F_T(pmpi_alltoallv, PMPI_ALLTOALLV),
		MPIROUTINE_F_T(pmpi_allgather, PMPI_ALLGATHER),
		MPIROUTINE_F_T(pmpi_allgatherv, PMPI_ALLGATHERV),
		MPIROUTINE_F_T(pmpi_gather, PMPI_GATHER),
		MPIROUTINE_F_T(pmpi_gatherv, PMPI_GATHERV),
		MPIROUTINE_F_T(pmpi_scatter, PMPI_SCATTERV),
		MPIROUTINE_F_T(pmpi_scatterv, PMPI_SCATTERV),
		MPIROUTINE_F_T(pmpi_comm_rank, PMPI_COMM_RANK),
		MPIROUTINE_F_T(pmpi_comm_size, PMPI_COMM_SIZE),
		MPIROUTINE_F_T(pmpi_comm_create, PMPI_COMM_CREATE),
		MPIROUTINE_F_T(pmpi_comm_free, PMPI_COMM_FREE),
		MPIROUTINE_F_T(pmpi_comm_dup, PMPI_COMM_DUP),
		MPIROUTINE_F_T(pmpi_comm_split, PMPI_COMM_SPLIT),
		MPIROUTINE_F_T(pmpi_cart_create, PMPI_CART_CREATE),
		MPIROUTINE_F_T(pmpi_cart_sub, PMPI_CART_SUB),
		MPIROUTINE_F_T(pmpi_start, PMPI_START),
		MPIROUTINE_F_T(pmpi_startall, PMPI_STARTALL),
		MPIROUTINE_F_T(pmpi_request_free, PMPI_REQUEST_FREE),
		MPIROUTINE_F_T(pmpi_scan, PMPI_SCAN),
		MPIROUTINE_F_T(pmpi_sendrecv, PMPI_SENDRECV),
		MPIROUTINE_F_T(pmpi_sendrecv_replace, PMPI_SENDRECV_REPLACE),
		MPIROUTINE_F_T(PMPI_Request_get_status, PMPI_REQUEST_GET_STATUS),
#if defined(MPI_SUPPORTS_MPI_IO)
		MPIROUTINE_F_T(pmpi_file_open, PMPI_FILE_OPEN),
		MPIROUTINE_F_T(pmpi_file_close, PMPI_FILE_CLOSE),
		MPIROUTINE_F_T(pmpi_file_read, PMPI_FILE_READ),
		MPIROUTINE_F_T(pmpi_file_read_all, PMPI_FILE_READ_ALL),
		MPIROUTINE_F_T(pmpi_file_write, PMPI_FILE_WRITE),
		MPIROUTINE_F_T(pmpi_file_write_all, PMPI_FILE_WRITE_ALL),
		MPIROUTINE_F_T(pmpi_file_read_at, PMPI_FILE_READ_AT),
		MPIROUTINE_F_T(pmpi_file_read_at_all, PMPI_FILE_READ_AT_ALL),
		MPIROUTINE_F_T(pmpi_file_write_at, PMPI_FILE_WRITE_AT),
		MPIROUTINE_F_T(pmpi_file_write_at_all, PMPI_FILE_WRITE_AT_ALL),
#endif /* MPI_SUPPORTS_MPI_IO */
		MPIROUTINE_C_T_END
	};


static struct MPIroutines_t MPIroutines_probes[] =
	{
		MPIROUTINE_C_T(PMPI_Probe),
		MPIROUTINE_C_T(PMPI_Iprobe),
		MPIROUTINE_F_T(pmpi_probe, PMPI_PROBE),
		MPIROUTINE_F_T(pmpi_iprobe, PMPI_IPROBE),
		MPIROUTINE_C_T_END
	};

void loadMPIPatches (BPatch_image *appImage)
{
	int i;

	cout << PACKAGE_NAME << ": Loading instrumentation MPI patches..." << flush;

	i = 0;
	while (MPIroutines[i].name != NULL)
	{
		string r = ((MPIroutines[i].language=='C')?string("PATCH_"):string("patch_"))+MPIroutines[i].name;
		MPIroutines[i].patch = getRoutine (r, appImage);
		if (NULL == MPIroutines[i].patch)
			cerr << "Unable to find " << MPIroutines[i].name << " in the application image" << endl;
		i++;
	}

	i = 0;
	while (MPIroutines_probes[i].name != NULL)
	{
		string r = ((MPIroutines_probes[i].language=='C')?string("PATCH_"):string("patch_"))+MPIroutines_probes[i].name;
		MPIroutines_probes[i].patch = getRoutine (r, appImage);
		if (NULL == MPIroutines_probes[i].patch)
			cerr << "Unable to find " << MPIroutines_probes[i].name << " in the application image" << endl;
		i++;
	}

	cout << "Done" << endl;
}

BPatch_function * getMPIPatch (char *routine, bool recursenames)
{
	BPatch_function *res = NULL;
	int i;

	/* Look for MPI calls using PMPI_ symbols */
	i = 0;
	while (MPIroutines[i].name != NULL)
	{
		if (!strcmp(MPIroutines[i].name, routine))
		{
			res = MPIroutines[i].patch;
			break;
		}
		i++;
	}

	/* If we haven't find it, check against MPI calls.
	   On some systems, MPI calls are done directly through PMPI symbols
	   and others are done through MPI symbols.
	   Even worse, on MN, one can find both in the same binary
	   This could be a dyninst naming issue on aliased symbols.
	*/
	if (res == NULL && recursenames)
	{
		BPatch_function *f;
		string new_name_lo = string("p")+routine;
		string new_name_up = string("P")+routine;
		f = getMPIPatch ((char*) new_name_lo.c_str(), false);
		if (f != NULL)
			return f;
		f = getMPIPatch ((char*) new_name_up.c_str(), false);
		if (f != NULL)
			return f;
		string sroutine = routine;
		if (sroutine[sroutine.length()-1] == 'f')
		{
			string tmp = sroutine.substr(0, sroutine.length()-1);
			return getMPIPatch ((char*) tmp.c_str(), true);
		}
		else
			return NULL;
	}
	else
		return res;
}

