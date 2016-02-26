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

unsigned nprintf_paraver_comm (char *buffer, 
	unsigned long long cpu_s, unsigned long long ptask_s,
	unsigned long long task_s, unsigned long long thread_s,
	unsigned long long log_s, unsigned long long phy_s,
	unsigned long long cpu_r, unsigned long long ptask_r,
	unsigned long long task_r, unsigned long long thread_r,
	unsigned long long log_r, unsigned long long phy_r,
	unsigned long long size, unsigned long long tag)
{
	unsigned index, index2, start;
	char lbuffer[32];

	/* Put type */
	buffer[0] = '3';
	buffer[1] = ':';
	start = 2;

	/* Put cpu_s */
	index2 = index = 0;
	while (cpu_s >= 10)
	{
		lbuffer[index] = (cpu_s%10)+(char) '0'; 
		cpu_s = cpu_s / 10;
		index++;
	}
	lbuffer[index] = cpu_s + (char) '0';
	index++;
	for (; index2 < index; index2++)
			buffer[index2+start] = lbuffer[index-index2-1];
	buffer[index2+start] = ':';
	start += index2+1;

	/* Put ptask_s */
	index2 = index = 0;
	while (ptask_s >= 10)
	{
		lbuffer[index] = (ptask_s%10)+(char) '0'; 
		ptask_s = ptask_s / 10;
		index++;
	}
	lbuffer[index] = ptask_s + (char) '0';
	index++;
	for (; index2 < index; index2++)
			buffer[start+index2] = lbuffer[index-index2-1];
	buffer[start+index2] = ':';
	start += index2+1;

	/* Put task_s */
	index2 = index = 0;
	while (task_s >= 10)
	{
		lbuffer[index] = (task_s%10)+(char) '0'; 
		task_s = task_s / 10;
		index++;
	}
	lbuffer[index] = task_s + (char) '0';
	index++;
	for (; index2 < index; index2++)
			buffer[start+index2] = lbuffer[index-index2-1];
	buffer[start+index2] = ':';
	start += index2+1;

	/* Put thread_s */
	index2 = index = 0;
	while (thread_s >= 10)
	{
		lbuffer[index] = (thread_s%10)+(char) '0'; 
		thread_s = thread_s / 10;
		index++;
	}
	lbuffer[index] = thread_s + (char) '0';
	index++;
	for (; index2 < index; index2++)
			buffer[start+index2] = lbuffer[index-index2-1];
	buffer[start+index2] = ':';
	start += index2+1;

	/* Put log_s */
	index2 = index = 0;
	while (log_s >= 10)
	{
		lbuffer[index] = (log_s%10)+(char) '0'; 
		log_s = log_s / 10;
		index++;
	}
	lbuffer[index] = log_s + (char) '0';
	index++;
	for (; index2 < index; index2++)
			buffer[start+index2] = lbuffer[index-index2-1];
	buffer[start+index2] = ':';
	start += index2+1;

	/* Put phy_s */
	index2 = index = 0;
	while (phy_s >= 10)
	{
		lbuffer[index] = (phy_s%10)+(char) '0'; 
		phy_s = phy_s / 10;
		index++;
	}
	lbuffer[index] = phy_s + (char) '0';
	index++;
	for (; index2 < index; index2++)
			buffer[start+index2] = lbuffer[index-index2-1];
	buffer[start+index2] = ':';
	start += index2+1;

	/* Put cpu_r */
	index2 = index = 0;
	while (cpu_r >= 10)
	{
		lbuffer[index] = (cpu_r%10)+(char) '0'; 
		cpu_r = cpu_r / 10;
		index++;
	}
	lbuffer[index] = cpu_r + (char) '0';
	index++;
	for (; index2 < index; index2++)
			buffer[index2+start] = lbuffer[index-index2-1];
	buffer[index2+start] = ':';
	start += index2+1;

	/* Put ptask_r */
	index2 = index = 0;
	while (ptask_r >= 10)
	{
		lbuffer[index] = (ptask_r%10)+(char) '0'; 
		ptask_r = ptask_r / 10;
		index++;
	}
	lbuffer[index] = ptask_r + (char) '0';
	index++;
	for (; index2 < index; index2++)
			buffer[start+index2] = lbuffer[index-index2-1];
	buffer[start+index2] = ':';
	start += index2+1;

	/* Put task_r */
	index2 = index = 0;
	while (task_r >= 10)
	{
		lbuffer[index] = (task_r%10)+(char) '0'; 
		task_r = task_r / 10;
		index++;
	}
	lbuffer[index] = task_r + (char) '0';
	index++;
	for (; index2 < index; index2++)
			buffer[start+index2] = lbuffer[index-index2-1];
	buffer[start+index2] = ':';
	start += index2+1;

	/* Put thread_r */
	index2 = index = 0;
	while (thread_r >= 10)
	{
		lbuffer[index] = (thread_r%10)+(char) '0'; 
		thread_r = thread_r / 10;
		index++;
	}
	lbuffer[index] = thread_r + (char) '0';
	index++;
	for (; index2 < index; index2++)
			buffer[start+index2] = lbuffer[index-index2-1];
	buffer[start+index2] = ':';
	start += index2+1;

	/* Put log_r */
	index2 = index = 0;
	while (log_r >= 10)
	{
		lbuffer[index] = (log_r%10)+(char) '0'; 
		log_r = log_r / 10;
		index++;
	}
	lbuffer[index] = log_r + (char) '0';
	index++;
	for (; index2 < index; index2++)
			buffer[start+index2] = lbuffer[index-index2-1];
	buffer[start+index2] = ':';
	start += index2+1;

	/* Put phy_r */
	index2 = index = 0;
	while (phy_r >= 10)
	{
		lbuffer[index] = (phy_r%10)+(char) '0'; 
		phy_r = phy_r / 10;
		index++;
	}
	lbuffer[index] = phy_r + (char) '0';
	index++;
	for (; index2 < index; index2++)
			buffer[start+index2] = lbuffer[index-index2-1];
	buffer[start+index2] = ':';
	start += index2+1;

	/* Put size */
	index2 = index = 0;
	while (size >= 10)
	{
		lbuffer[index] = (size%10)+(char) '0'; 
		size = size / 10;
		index++;
	}
	lbuffer[index] = size + (char) '0';
	index++;
	for (; index2 < index; index2++)
			buffer[start+index2] = lbuffer[index-index2-1];
	buffer[start+index2] = ':';
	start += index2+1;

	/* Put tag */
	index2 = index = 0;
	while (tag >= 10)
	{
		lbuffer[index] = (tag%10)+(char) '0'; 
		tag = tag / 10;
		index++;
	}
	lbuffer[index] = tag + (char) '0';
	index++;
	for (; index2 < index; index2++)
			buffer[start+index2] = lbuffer[index-index2-1];
	buffer[start+index2] = '\n';
	buffer[start+index2+1] = (char) 0;
	start += index2+1;

	return start;
}

unsigned nprintf_paraver_event_type_value (char *buffer,
	unsigned long long type, unsigned long long value)
{
	unsigned index, index2, start;
	char lbuffer[32];

	/* Put two-dots */
	buffer[0] = ':';
	start = 1;

	/* Put type */
	index2 = index = 0;
	while (type >= 10)
	{
		lbuffer[index] = (type%10)+(char) '0'; 
		type = type / 10;
		index++;
	}
	lbuffer[index] = type + (char) '0';
	index++;
	for (; index2 < index; index2++)
			buffer[index2+start] = lbuffer[index-index2-1];
	buffer[index2+start] = ':';
	start += index2+1;

	/* Put value */
	index2 = index = 0;
	while (value >= 10)
	{
		lbuffer[index] = (value%10)+(char) '0'; 
		value = value / 10;
		index++;
	}
	lbuffer[index] = value + (char) '0';
	index++;
	for (; index2 < index; index2++)
			buffer[start+index2] = lbuffer[index-index2-1];
	buffer[start+index2] = (char) 0;
	start += index2;

	return start;
}

unsigned nprintf_paraver_event_head (char *buffer,
	unsigned long long cpu, unsigned long long ptask,
	unsigned long long task, unsigned long long thread,
	unsigned long long time)
{
	unsigned index, index2, start;
	char lbuffer[32];

	/* Put type */
	buffer[0] = '2';
	buffer[1] = ':';
	start = 2;

	/* Put cpu */
	index2 = index = 0;
	while (cpu >= 10)
	{
		lbuffer[index] = (cpu%10)+(char) '0'; 
		cpu = cpu / 10;
		index++;
	}
	lbuffer[index] = cpu + (char) '0';
	index++;
	for (; index2 < index; index2++)
			buffer[index2+start] = lbuffer[index-index2-1];
	buffer[index2+start] = ':';
	start += index2+1;

	/* Put ptask */
	index2 = index = 0;
	while (ptask >= 10)
	{
		lbuffer[index] = (ptask%10)+(char) '0'; 
		ptask = ptask / 10;
		index++;
	}
	lbuffer[index] = ptask + (char) '0';
	index++;
	for (; index2 < index; index2++)
			buffer[start+index2] = lbuffer[index-index2-1];
	buffer[start+index2] = ':';
	start += index2+1;

	/* Put task */
	index2 = index = 0;
	while (task >= 10)
	{
		lbuffer[index] = (task%10)+(char) '0'; 
		task = task / 10;
		index++;
	}
	lbuffer[index] = task + (char) '0';
	index++;
	for (; index2 < index; index2++)
			buffer[start+index2] = lbuffer[index-index2-1];
	buffer[start+index2] = ':';
	start += index2+1;

	/* Put thread */
	index2 = index = 0;
	while (thread >= 10)
	{
		lbuffer[index] = (thread%10)+(char) '0'; 
		thread = thread / 10;
		index++;
	}
	lbuffer[index] = thread + (char) '0';
	index++;
	for (; index2 < index; index2++)
			buffer[start+index2] = lbuffer[index-index2-1];
	buffer[start+index2] = ':';
	start += index2+1;

	/* Put thread */
	index2 = index = 0;
	while (time >= 10)
	{
		lbuffer[index] = (time%10)+(char) '0'; 
		time = time / 10;
		index++;
	}
	lbuffer[index] = time + (char) '0';
	index++;
	for (; index2 < index; index2++)
			buffer[start+index2] = lbuffer[index-index2-1];
	buffer[start+index2] = (char) 0;
	start += index2;

	return start;
}

unsigned nprintf_paraver_state (char *buffer,
	unsigned long long cpu, unsigned long long ptask,
	unsigned long long task, unsigned long long thread,
	unsigned long long ini_time, unsigned long long end_time,
	unsigned long long state)
{
	unsigned index, index2, start;
	char lbuffer[32];

	/* Put type */
	buffer[0] = '1';
	buffer[1] = ':';
	start = 2;

	/* Put cpu */
	index2 = index = 0;
	while (cpu >= 10)
	{
		lbuffer[index] = (cpu%10)+(char) '0'; 
		cpu = cpu / 10;
		index++;
	}
	lbuffer[index] = cpu + (char) '0';
	index++;
	for (; index2 < index; index2++)
			buffer[index2+start] = lbuffer[index-index2-1];
	buffer[index2+start] = ':';
	start += index2+1;

	/* Put ptask */
	index2 = index = 0;
	while (ptask >= 10)
	{
		lbuffer[index] = (ptask%10)+(char) '0'; 
		ptask = ptask / 10;
		index++;
	}
	lbuffer[index] = ptask + (char) '0';
	index++;
	for (; index2 < index; index2++)
			buffer[start+index2] = lbuffer[index-index2-1];
	buffer[start+index2] = ':';
	start += index2+1;

	/* Put task */
	index2 = index = 0;
	while (task >= 10)
	{
		lbuffer[index] = (task%10)+(char) '0'; 
		task = task / 10;
		index++;
	}
	lbuffer[index] = task + (char) '0';
	index++;
	for (; index2 < index; index2++)
			buffer[start+index2] = lbuffer[index-index2-1];
	buffer[start+index2] = ':';
	start += index2+1;

	/* Put thread */
	index2 = index = 0;
	while (thread >= 10)
	{
		lbuffer[index] = (thread%10)+(char) '0'; 
		thread = thread / 10;
		index++;
	}
	lbuffer[index] = thread + (char) '0';
	index++;
	for (; index2 < index; index2++)
			buffer[start+index2] = lbuffer[index-index2-1];
	buffer[start+index2] = ':';
	start += index2+1;

	/* Put ini_time */
	index2 = index = 0;
	while (ini_time >= 10)
	{
		lbuffer[index] = (ini_time%10)+(char) '0'; 
		ini_time = ini_time / 10;
		index++;
	}
	lbuffer[index] = ini_time + (char) '0';
	index++;
	for (; index2 < index; index2++)
			buffer[start+index2] = lbuffer[index-index2-1];
	buffer[start+index2] = ':';
	start += index2+1;

	/* Put end_time */
	index2 = index = 0;
	while (end_time >= 10)
	{
		lbuffer[index] = (end_time%10)+(char) '0'; 
		end_time = end_time / 10;
		index++;
	}
	lbuffer[index] = end_time + (char) '0';
	index++;
	for (; index2 < index; index2++)
			buffer[start+index2] = lbuffer[index-index2-1];
	buffer[start+index2] = ':';
	start += index2+1;

	/* Put state */
	index2 = index = 0;
	while (state >= 10)
	{
		lbuffer[index] = (state%10)+(char) '0'; 
		state = state / 10;
		index++;
	}
	lbuffer[index] = state + (char) '0';
	index++;
	for (; index2 < index; index2++)
			buffer[start+index2] = lbuffer[index-index2-1];
	buffer[start+index2] = '\n';
	buffer[start+index2+1] = (char) 0;
	start += index2+1;

	return start;
}
