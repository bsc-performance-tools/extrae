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

#define _GNU_SOURCE
#include "common.h"

#define DBG

#ifdef HAVE_STDIO_H
# include <stdio.h>
# define DBG fprintf(stderr, "Captured %s\n", __func__);
#endif

#include "gpi_wrapper.h"

gaspi_return_t
gaspi_proc_init(gaspi_timeout_t timeout_ms)
{
	DBG
	
	int ret;

	Extrae_GPI_Init_Entry();
	ret = pgaspi_proc_init(timeout_ms);
	Extrae_GPI_Init_Exit();

	return ret;
}

gaspi_return_t
gaspi_proc_term(gaspi_timeout_t timeout_ms)
{
	DBG
	
	int ret;

	Extrae_GPI_Term_Entry();
	ret = pgaspi_proc_term(timeout_ms);
	Extrae_GPI_Term_Exit();

	return ret;
}

gaspi_return_t
gaspi_segment_create(const gaspi_segment_id_t segment_id,
    const gaspi_size_t size, const gaspi_group_t group,
    const gaspi_timeout_t timeout_ms, const gaspi_alloc_t alloc_policy)
{
	DBG
	
	int ret;

	ret = pgaspi_segment_create(segment_id, size, group, timeout_ms, alloc_policy);

	return ret;
}

gaspi_return_t
gaspi_write(const gaspi_segment_id_t segment_id_local,
    const gaspi_offset_t offset_local, const gaspi_rank_t rank,
    const gaspi_segment_id_t segment_id_remote,
    const gaspi_offset_t offset_remote, const gaspi_size_t size,
    const gaspi_queue_id_t queue, const gaspi_timeout_t timeout_ms)
{
	DBG

	int ret;
	
	ret = pgaspi_write(segment_id_local, offset_local, rank, segment_id_remote, offset_remote, size, queue, timeout_ms);

	return ret;
}
