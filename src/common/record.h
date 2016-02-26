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

#ifndef RECORD_H_INCLUDED
#define RECORD_H_INCLUDED

#include "common.h"
#include "num_hwc.h"

typedef struct omp_param_t
{
  UINT64 param[2];
} omp_param_t;

typedef struct misc_param_t
{
  UINT64 param;
} misc_param_t;


typedef struct mpi_param_t
{
  INT32 target;
  INT32 size;
  INT32 tag;
  INT32 comm;
  INT64 aux;
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
  UINT64 value;                  /* Value of this event                   */
  UINT64 time;                   /* Timestamp of this event               */
#if 1 || USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)
  long long HWCValues[MAX_HWC];      /* Hardware counters read for this event */
#endif
  INT32 event;                   /* Type of this event                    */
#if 1 || USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)
  INT32 HWCReadSet;              /* Marks which set of counters was read, if any */
#endif
} event_t;


#define EVT_SIZE  sizeof(event_t)

#define Get_EvTime(ptr)          (ptr == NULL ? 0 : ptr->time)
#define Get_EvEvent(ptr)         ((ptr)->event)
#define Get_EvValue(ptr)         ((ptr)->value)
#define Get_EvTarget(ptr)        ((ptr)->param.mpi_param.target)
#define Get_EvSize(ptr)          ((ptr)->param.mpi_param.size)
#define Get_EvTag(ptr)           ((ptr)->param.mpi_param.tag)
#define Get_EvComm(ptr)          ((ptr)->param.mpi_param.comm)
#define Get_EvAux(ptr)           ((ptr)->param.mpi_param.aux)
#define Get_EvParam(ptr)         ((ptr)->param.omp_param.param[0])
#define Get_EvNParam(ptr,i)      ((ptr)->param.omp_param.param[i])
#define Get_EvMiscParam(ptr)     ((ptr)->param.misc_param.param)
#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)
# define Get_EvHWCRead(ptr)      (((ptr)->HWCReadSet != 0) ? 1 : 0) /* 0 = not read, >0 = set_id + 1 */

# define Get_EvHWCSet(ptr)       (((ptr)->HWCReadSet > 0) ? ((ptr)->HWCReadSet - 1) : ((((ptr)->HWCReadSet)*(-1)) - 1) )

# define Get_EvHWCVal(ptr)       ((ptr)->HWCValues)

# define Reset_EvHWCs(ptr)                        \
{                                                 \
  if ((ptr)->HWCReadSet > 0)                      \
  {                                               \
    (ptr)->HWCReadSet = (ptr)->HWCReadSet * (-1); \
  }                                               \
}

# define Check_EvHWCsReset(ptr) ((ptr)->HWCReadSet < 0 ? 1 : 0)

# define Get_EvHWC(ptr, cnt) (Check_EvHWCsReset(ptr) ? 0 : (ptr)->HWCValues[cnt])

#endif /* USE_HARDWARE_COUNTERS */

#endif /* RECORD_H_INCLUDED */

