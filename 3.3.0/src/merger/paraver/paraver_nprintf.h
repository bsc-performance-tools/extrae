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

unsigned nprintf_paraver_comm (char *buffer, 
	unsigned long long cpu_s, unsigned long long ptask_s,
	unsigned long long task_s, unsigned long long thread_s,
	unsigned long long log_s, unsigned long long phy_s,
	unsigned long long cpu_r, unsigned long long ptask_r,
	unsigned long long task_r, unsigned long long thread_r,
	unsigned long long log_r, unsigned long long phy_r,
	unsigned long long size, unsigned long long tag);

unsigned nprintf_paraver_event_type_value (char *buffer,
	unsigned long long type, unsigned long long value);

unsigned nprintf_paraver_event_head (char *buffer,
	unsigned long long cpu, unsigned long long ptask,
	unsigned long long task, unsigned long long thread,
	unsigned long long time);

unsigned nprintf_paraver_state (char *buffer,
	unsigned long long cpu, unsigned long long ptask,
	unsigned long long task, unsigned long long thread,
	unsigned long long ini_time, unsigned long long end_time,
	unsigned long long state);
