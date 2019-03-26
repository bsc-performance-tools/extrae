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

#pragma once

#include <PGASPI.h>

gaspi_return_t	gaspi_proc_init(const gaspi_timeout_t timeout_ms);

gaspi_return_t	gaspi_proc_term(const gaspi_timeout_t timeout_ms);

gaspi_return_t	gaspi_segment_create(const gaspi_segment_id_t segment_id,
				    const gaspi_size_t size, const gaspi_group_t group,
				    const gaspi_timeout_t timeout_ms,
				    const gaspi_alloc_t alloc_policy);

gaspi_return_t	gaspi_write(const gaspi_segment_id_t segment_id_local,
				    const gaspi_offset_t offset_local, const gaspi_rank_t rank,
				    const gaspi_segment_id_t segment_id_remote,
				    const gaspi_offset_t offset_remote, const gaspi_size_t size,
				    const gaspi_queue_id_t queue,
				    const gaspi_timeout_t timeout_ms);
