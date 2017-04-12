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

#ifndef GNU_LIBGOMP_WRAPPER_H_
#define GNU_LIBGOMP_WRAPPER_H_

/*
 * Several data helpers to temporarily store the pointers to the real 
 * outlined functions or tasks, and the real data arguments, while we inject 
 * a fake task that will replace the original one to emit instrumentation 
 * events, and then retrieve the original pointers through the helpers to 
 * end up calling the real functions.
 */
struct parallel_helper_t                                                        
{                                                                               
  void (*fn)(void *);                                                           
  void *data;                                                                   
};                                                                              
                                                                                
struct task_helper_t                                                            
{                                                                               
  void (*fn)(void *);                                                           
  void *data;                                                                   
  void *buf;                                                                    
  long long counter;                                                            
};                                                                              
                                                                                
struct helpers_queue_t                                                          
{                                                                               
  struct parallel_helper_t *queue;                                              
  int current_helper;                                                             
  int max_helpers;                                                                
};                                                                              

int _extrae_gnu_libgomp_init (int rank);

#endif /* GNU_LIBGOMP_WRAPPER_H_ */
