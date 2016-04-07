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

#ifndef EXTRAE_INTEL_PEBS_TYPES_H_INCLUDED
#define EXTRAE_INTEL_PEBS_TYPES_H_INCLUDED

typedef enum e_PEBS_MemoryHierarchy_Level
{
	PEBS_MEMORYHIERARCHY_UNCACHEABLE_IO = 0,
	PEBS_MEMORYHIERARCHY_MEM_LVL_L1 = 1,
	PEBS_MEMORYHIERARCHY_MEM_LVL_LFB = 2,
	PEBS_MEMORYHIERARCHY_MEM_LVL_L2 = 3,
	PEBS_MEMORYHIERARCHY_MEM_LVL_L3 = 4,
	PEBS_MEMORYHIERARCHY_MEM_LVL_RCACHE_1HOP = 5,
	PEBS_MEMORYHIERARCHY_MEM_LVL_RCACHE_2HOP = 6,
	PEBS_MEMORYHIERARCHY_MEM_LVL_LOCAL_RAM = 7,
	PEBS_MEMORYHIERARCHY_MEM_LVL_REMOTE_RAM_1HOP = 8,
	PEBS_MEMORYHIERARCHY_MEM_LVL_REMOTE_RAM_2HOP = 9
} PEBS_MemoryHierarchy_Level;

typedef enum e_PEBS_MemoryTLB_Level
{
	PEBS_MEMORYHIERARCHY_TLB_OTHER = 0,
	PEBS_MEMORYHIERARCHY_TLB_L1 = 1,
	PEBS_MEMORYHIERARCHY_TLB_L2 = 2
} PEBS_MemoryTLB_Level;

typedef enum e_PEBS_MemoryHierarchy_HitOrMiss
{
	PEBS_MEMORYHIERARCHY_UNKNOWN = 0,
	PEBS_MEMORYHIERARCHY_HIT = 1,
	PEBS_MEMORYHIERARCHY_MISS = 2
} PEBS_MemoryHierarchy_HitOrMiss;

#endif /* EXTRAE_INTEL_PEBS_TYPES_H_INCLUDED */

