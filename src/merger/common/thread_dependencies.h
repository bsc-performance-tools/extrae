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

#ifndef THREAD_DEPENDENCIES_H_INCLUDED
#define THREAD_DEPENDENCIES_H_INCLUDED

struct ThreadDependencies_st;

struct ThreadDependencies_st * ThreadDependency_create (void);
void ThreadDependency_add (struct ThreadDependencies_st *td,
	const void *dependency_data);

typedef int (*ThreadDepedendencyProcessor_ifMatchSetPredecessor)(
	const void *dependency_event, void *userdata, void **predecessordata);
void ThreadDependency_processAll_ifMatchSetPredecessor (
	struct ThreadDependencies_st *td,
	ThreadDepedendencyProcessor_ifMatchSetPredecessor cb,
	void *userdata);

typedef int (*ThreadDepedendencyProcessor_ifMatchDelete)(
	const void *dependency_event,
	const void *predecessor_event,
	const void *userdata);
void ThreadDependency_processAll_ifMatchDelete (
	struct ThreadDependencies_st *td,
	ThreadDepedendencyProcessor_ifMatchDelete cb,
	const void *userdata);

#endif /* THREAD_DEPENDENCIES_H_INCLUDED */

