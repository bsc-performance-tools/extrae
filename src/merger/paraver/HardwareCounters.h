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

#ifndef _HARDWARE_COUNTERS_H
#define _HARDWARE_COUNTERS_H

#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)

#include "num_hwc.h"
#include "hwc_version.h"
#include "record.h"
#if defined(PAPI_COUNTERS)
# include "papi.h"
# include "papiStdEventDefs.h"
# if defined(PAPIv3)
#  include "MurmurHash2.h"
# endif
#endif

/**
 * local_hwc_data_t stores the translations from local counter id's 
 * (PAPI/PMAPI codes assigned to each thread) into global id's 
 * (final paraver type). This information is indexed by ptask for 
 * direct translation while processing mpit data. The counter
 * information (name, description) can be accessed through the 
 * global_hwc_data_t structure using the translated 'global_id' 
 * as search key.
 */
typedef struct hwc_id_t
{
  int ptask;
  int local_id;
  int global_id;
} hwc_id_t;

typedef struct ptask_hwc_t
{
  hwc_id_t *local_to_global;
  int num_counters;
} ptask_hwc_t;

typedef struct local_hwc_data_t
{
  ptask_hwc_t *ptask_counters;
  int num_ptasks;
} local_hwc_data_t;


/** 
 * global_hwc_data_t stores a list of unique counters from all 
 * ptasks and threads. Those marked as 'used' appear in mpit data 
 * and their labels have to appear in the final PCF. 
 */
typedef struct hwc_info_t
{
  char *name;
  char *description;
  int global_id;
  int used;
} hwc_info_t;

typedef struct global_hwc_data_t
{
  hwc_info_t *counters_info;
  int num_counters;
} global_hwc_data_t;


#if defined(PAPI_COUNTERS)
int check_if_uncore_in_PFM(char *event_name);
#endif

/* Macro per calcular el tipus que cal assignar al comptador.
 * Els comptadors a linux son numeros aixi: 0x800000XX
 * Els comptadors a linux(natiu) son numeros aixi: 0x4000XXXX
 * per tant, nomes cal tenir en compte els bytes mes baixos per
 * no tenir numeros inmensos. */

/* Uncore and native counters do not have a consistent numbering scheme, 
 * as the assigned identifiers vary depending on the order of addition of 
 * the counters to the PAPI EventSet. Therefore, different executions reading
 * different counters will reuse the same id's. To assign univocal id's, 
 * we compute the global id based on the counter name rather than the local PAPI id's.
 * For PAPI presets, we still use the PAPI local id as we used to do before.
 * When the sym is not present, thus the counter name is unknown, we use the 
 * old method based on the local PAPI id's, which may have collisions.
 */ 
#if defined(PAPI_COUNTERS)
# if defined(PAPIv2)
#  define GET_PARAVER_CODE_FOR_HWC(x, name) (HWC_BASE_PAPI_PRESET + (x & 0x000000FF))
#  define LEGACY_HWC_COUNTER_TYPE(x) (HWC_BASE_PAPI_PRESET + (x & 0x000000FF))
# elif defined(PAPIv3)
#  define HASH(name) ((name != NULL) ? (MurmurHash2(name, strlen(name), 0) % (1000000)) : 0)
#  define IS_UNCORE(x) check_if_uncore_in_PFM(x)
#  define GET_PARAVER_CODE_FOR_HWC(x, name) \
	(IS_PRESET(x) ? (HWC_BASE_PAPI_PRESET + (x & 0x0000FFFF)) : \
	                (IS_UNCORE(name) ? (HWC_BASE_PAPI_UNCORE + HASH(name)) : \
	                                   (HWC_BASE_PAPI_NATIVE + HASH(name))))
#  define LEGACY_HWC_COUNTER_TYPE(x) \
	(IS_PRESET(x) ? HWC_BASE_PAPI_PRESET : HWC_BASE_PAPI_NATIVE) + (x & 0x0000FFFF)
# endif
#elif defined(PMAPI_COUNTERS)
# define GET_PARAVER_CODE_FOR_HWC(x, name) (HWC_BASE_PMAPI + x)
# define LEGACY_HWC_COUNTER_TYPE(x) (HWC_BASE_PMAPI + x)
#elif defined(L4STAT)
# define HWC_L4STAT_BASE HWC_BASE_PAPI_PRESET
# define LEGACY_HWC_COUNTER_TYPE(x) HWC_L4STAT_BASE + ((x & 0x000000FF))
# define GET_PARAVER_CODE_FOR_HWC(x, name) LEGACY_HWC_COUNTER_TYPE(x)
#endif

void HardwareCounters_AssignGlobalID (int ptask, int local_id, char *definition);
int HardwareCounters_LocalToGlobalID (int ptask, int local_id);
int HardwareCounters_GetUsed (hwc_info_t ***used_counters);

void HardwareCounters_NewSetDefinition (int ptask, int task, int thread, int newSet, long long *HWCIds);
int HardwareCounters_Change (int ptask, int task, int thread, unsigned long long change_time, int newSet, unsigned int *outtypes, unsigned long long *outvalues);
void HardwareCounters_SetOverflow (int ptask, int task, int thread, event_t *Event);
int HardwareCounters_Emit (int ptask, int task, int thread,
	unsigned long long time, event_t * Event,
	unsigned int *outtype, unsigned long long *outvalue,
	int absolute);
void HardwareCounters_Show (const event_t * Event, int ncounters);

#if defined(PARALLEL_MERGE)

#define resize(buffer, size)                            \
{                                                       \
        buffer ## _ ## size += size;                    \
        buffer = xrealloc(buffer, buffer ## _ ## size); \
}

void Share_HWC_Before_Processing_MPITS (int rank);
void Share_HWC_After_Processing_MPITS (int rank);

#endif /* PARALLEL_MERGE */

#endif /* USE_HARDWARE_COUNTERS || HETEROGENEOUS_SUPPORT */

#endif /* _HARDWARE_COUNTERS_H */
