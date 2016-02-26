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

#ifndef PARAVER_STATE_H
#define PARAVER_STATE_H

#include "file_set.h"

unsigned int Push_State (unsigned int new_state, unsigned int ptask, unsigned int task, unsigned int thread);
unsigned int Pop_State (unsigned int old_state, unsigned int ptask, unsigned int task, unsigned int thread);
unsigned int Pop_Until (unsigned int until_state, unsigned int ptask, unsigned int task, unsigned int thread);
unsigned int Switch_State (unsigned int state, int condition, unsigned int ptask, unsigned int task, unsigned int thread);
unsigned int Top_State (unsigned int ptask, unsigned int task, unsigned int thread);
void Dump_States_Stack (unsigned int ptask, unsigned int task, unsigned int thread);
int State_Excluded (unsigned int state);
void Initialize_Trace_Mode_States (unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread, int mode);
void Initialize_States (FileSet_t * fset);
void Finalize_States (FileSet_t * fset, unsigned long long current_time);

int Get_Last_State (void);

#endif
