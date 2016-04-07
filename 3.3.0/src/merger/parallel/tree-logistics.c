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

#include "tree-logistics.h"

int tree_pow (int base, int exp)
{
	int i;
	int res;
	
	for (res = 1, i = 0; i < exp; i++)
		res = res * base;

	return res;
}

/*
  tree_TaskHaveWork returns TRUE if the taskid has to work on the current
  depth (tree_depth) of a tree with tree_fanout wide.
*/
int tree_TaskHaveWork (int taskid, int tree_fanout, int tree_depth)
{
	return (taskid % tree_pow (tree_fanout, tree_depth)) == 0;
}

/*
  tree_MasterOfSubtree returns TRUE if the taskid is the master (root) of the
  tree on the current depth (tree_depth) of a tree with tree_fanout wide.
*/
int tree_MasterOfSubtree (int taskid, int tree_fanout, int tree_depth)
{
	return (taskid % tree_pow (tree_fanout, tree_depth+1)) == 0;
}

int tree_myMaster (int taskid, int tree_fanout, int tree_depth)
{
	return (taskid / tree_pow (tree_fanout, 1+tree_depth)) * tree_pow (tree_fanout, 1+tree_depth);
}

int tree_MaxDepth (int ntasks, int tree_fanout)
{
	int max_depth = 0;

	while (ntasks > tree_pow(tree_fanout, max_depth))
		max_depth++;

	return max_depth;
}

