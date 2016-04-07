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

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif

#include "openshmem_prv_events.h"
#include "openshmem_events.h"
#include "labels.h"

int OPENSHMEM_Present = FALSE;

/******************************************************************************
 **      Function name : Enable_OPENSHMEM_Operation
 **      
 **      Description : 
 ******************************************************************************/

void Enable_OPENSHMEM_Operation (int Op)
{
	UNREFERENCED_PARAMETER(Op);
  OPENSHMEM_Present = TRUE;
}

void WriteEnabled_OPENSHMEM_Operations (FILE * fd)
{
	unsigned u = 0;

        if (OPENSHMEM_Present)
        {
                fprintf (fd, "EVENT_TYPE\n");
                fprintf (fd, "%d    %d    %s\n", 0, OPENSHMEM_BASE_EVENT, "OpenSHMEM calls");
                fprintf (fd, "VALUES\n");
                fprintf (fd, "0 Outside OpenSHMEM\n");

                for (u = 0; u < COUNT_OPENSHMEM_EVENTS; u++)
                                fprintf (fd, "%d %s\n", u+1, GetOPENSHMEMLabel( u ));
                LET_SPACES(fd);
		fprintf(fd, "EVENT_TYPE\n");
		fprintf (fd, "%d    %d    %s\n", 0, OPENSHMEM_SENDBYTES_EV, "OpenSHMEM outgoing bytes");
		LET_SPACES(fd);

		fprintf(fd, "EVENT_TYPE\n");
		fprintf (fd, "%d    %d    %s\n", 0, OPENSHMEM_RECVBYTES_EV, "OpenSHMEM incoming bytes");
		LET_SPACES(fd);
        }
}

