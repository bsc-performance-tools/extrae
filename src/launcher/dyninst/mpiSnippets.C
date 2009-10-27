/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                 MPItrace                                  *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *                                                             ___           *
 *   +---------+     http:// www.cepba.upc.edu/tools_i.htm    /  __          *
 *   |    o//o |     http:// www.bsc.es                      /  /  _____     *
 *   |   o//o  |                                            /  /  /     \    *
 *   |  o//o   |     E-mail: cepbatools@cepba.upc.edu      (  (  ( B S C )   *
 *   | o//o    |     Phone:          +34-93-401 71 78       \  \  \_____/    *
 *   +---------+     Fax:            +34-93-401 25 77        \  \__          *
 *    C E P B A                                               \___           *
 *                                                                           *
 * This software is subject to the terms of the CEPBA/BSC license agreement. *
 *      You must accept the terms of this license to use this software.      *
 *                                 ---------                                 *
 *                European Center for Parallelism of Barcelona               *
 *                      Barcelona Supercomputing Center                      *
\*****************************************************************************/

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/launcher/dyninst/mpiSnippets.C,v $
 | 
 | @last_commit: $Date: 2009/01/13 16:12:01 $
 | @version:     $Revision: 1.7 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: mpiSnippets.C,v 1.7 2009/01/13 16:12:01 harald Exp $";

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
	char *name;
	char language;
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
#if defined(MPI_HAS_INIT_THREAD)
		MPIROUTINE_C_T(PMPI_Init),
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
#if defined(MPI_COMBINED_C_FORTRAN)
#if defined(MPI_C_CONTAINS_FORTRAN_MPI_INIT)
		MPIROUTINE_F_T(pmpi_init, PMPI_INIT),
#if defined(MPI_HAS_INIT_THREAD)
		MPIROUTINE_F_T(pmpi_init_thread, PMPI_INIT_THREAD),
#endif /* MPI_HAS_INIT_THREAD */
#endif /* MPI_C_CONTAINS_FORTRAN_MPI_INIT */
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
#endif /* MPI_COMBINED_C_FORTRAN */
		MPIROUTINE_C_T_END
	};


static struct MPIroutines_t MPIroutines_probes[] =
	{
		MPIROUTINE_C_T(PMPI_Probe),
		MPIROUTINE_C_T(PMPI_Iprobe),

#if defined(MPI_COMBINED_C_FORTRAN)
		MPIROUTINE_F_T(pmpi_probe, PMPI_PROBE),
		MPIROUTINE_F_T(pmpi_iprobe, PMPI_IPROBE),
#endif /* MPI_COMBINED_C_FORTRAN */

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

BPatch_function * getMPIPatch (char *routine)
{
	BPatch_function *res = NULL;
	int i;

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
	return res;
}

