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

#include <stdio.h>

#define MAX_HWC 8
#define HETEROGENEOUS_SUPPORT

typedef struct omp_param_t
{
	unsigned long long param;
} omp_param_t;

typedef struct misc_param_t
{
	unsigned long long param;
} misc_param_t;


typedef struct mpi_param_t
{
	int target;                   /* receiver in send - sender in receive */
	int size;
	int tag;
	int comm;
	int aux;
#if defined(HETEROGENEOUS_SUPPORT)
	int padding[1];
#endif
} mpi_param_t;


typedef union
{
	struct omp_param_t omp_param;
	struct mpi_param_t mpi_param;
	struct misc_param_t misc_param;
} u_param;

/* HSG

  This struct contains the elements of every event that must be recorded.
  The fields must be placed in a such way that the sizeof(event_t) must
  be minimal. Each architecture has it's own preference on the alignament,
  so we must care about the packing of the structure. This is very important
  in the heterogeneous environments.
*/

typedef struct
{
	u_param param;                 /* Parameters of this event              */
	unsigned long long value;      /* Value of this event                   */
	long long time;                /* Timestamp of this event               */
#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)
	long long HWCValues[MAX_HWC];  /* Hardware counters read for this event */
#endif
	int event;                     /* Type of this event                    */
#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)
	int HWCReadSet;                /* Has this event HWC read?              */
#endif
} event_t;

int main (int argc, char *argv[])
{
	event_t e;

	printf ("\nsizeof(event_t) = %d\n\n", sizeof(event_t));
	printf ("@e.param        = %d\n", ((long) &e.param) - ((long) &e));
	printf ("@e.value        = %d\n", ((long) &e.value) - ((long) &e));
	printf ("@e.time         = %d\n", ((long) &e.time) - ((long) &e));
	printf ("@e.HWCValues    = %d\n", ((long) &e.HWCValues) - ((long) &e));
	printf ("@e.event        = %d\n", ((long) &e.event) - ((long) &e));
	printf ("@e.HWCReadSet   = %d\n", ((long) &e.HWCReadSet) - ((long) &e));
	printf ("\n");

	return 0;
}

