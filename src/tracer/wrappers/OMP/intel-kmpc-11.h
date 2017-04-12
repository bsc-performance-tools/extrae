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

#ifndef INTEL_KMPC_11_WRAPPER_H_
#define INTEL_KMPC_11_WRAPPER_H_

struct __kmpv_location_t                                                        
{                                                                               
  int reserved_1;                                                               
  int flags;                                                                    
  int reserved_2;                                                               
  int reserved_3;                                                               
  char *location;                                                               
};                                                                              
                                                                                
struct __kmp_task_t                                                             
{                                                                               
  void *shareds;                                                                
  void *routine;                                                                
  int part_id;                                                                  
};                                                                              

void helper__kmpc_taskloop_substitute(int arg, void *wrap_task, int helper_id);

extern void (*__kmpc_fork_call_real)(void*,int,void*,...);

int _extrae_intel_kmpc_init (int rank);

#endif
