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

#ifndef ADDRESS_SPACE_H_INCLUDED
#define ADDRESS_SPACE_H_INCLUDED

#include "common.h"

struct AddressSpace_st;

struct AddressSpace_st* AddressSpace_create (void);

void AddressSpace_add (struct AddressSpace_st *as, uint64_t AddressBegin,
	uint64_t AddressEnd, uint64_t *CallerAddresses,
	uint32_t CallerType);

void AddressSpace_remove (struct AddressSpace_st *as, uint64_t AddressBegin);

int AddressSpace_search (struct AddressSpace_st *as, uint64_t Address,
	uint64_t **CallerAddresses, uint32_t *CallerType);

#endif /* ADDRESS_SPACE_H_INCLUDED */

