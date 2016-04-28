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

#ifndef __OMPT_TARGET_H__
#define __OMPT_TARGET_H__

#define EXTRAE_OMPT_TARGET_BUFFER_SIZE 1000

/* Struct that contains information for each OMPT device */
typedef struct
{
  int ompt_device_id;                 // OMPT's device identifier (from 0 to n-1)
  ompt_function_lookup_t lookup;      // Lookup pointer to retrieve the OMPT API calls for the device
  ompt_target_device_t *device_ptr;   // Pointer to the device context

  int extrae_thread_id;               // Extrae's logical thread (the row in Paraver) 
  long long latency;                  // Latency that has to be applied to the timestamps of the records produced by this device
} extrae_device_info_t;

int ompt_target_initialize(ompt_function_lookup_t lookup);

void ompt_target_finalize();

#endif /* __OMPT_TARGET_H__ */
