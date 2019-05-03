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

void Extrae_set_trace_GPI(int _trace);
int  Extrae_get_trace_GPI();

void Extrae_set_trace_GPI_HWC(int _trace);
int  Extrae_get_trace_GPI_HWC();


void Probe_GPI_init_Entry();
void Probe_GPI_init_Exit();

void Probe_GPI_term_Entry();
void Probe_GPI_term_Exit();

void Probe_GPI_connect_Entry(const gaspi_rank_t _rank);
void Probe_GPI_connect_Exit();

void Probe_GPI_disconnect_Entry(const gaspi_rank_t _rank);
void Probe_GPI_disconnect_Exit();

void Probe_GPI_group_create_Entry();
void Probe_GPI_group_create_Exit(const gaspi_group_t *_group);

void Probe_GPI_group_add_Entry(
    const gaspi_group_t _group,
    const gaspi_rank_t  _rank);
void Probe_GPI_group_add_Exit();

void Probe_GPI_group_commit_Entry(
    const gaspi_group_t   _group,
    const gaspi_timeout_t _timeout);
void Probe_GPI_group_commit_Exit();

void Probe_GPI_barrier_Entry();
void Probe_GPI_barrier_Exit();

void Probe_GPI_segment_create_Entry(
    const gaspi_segment_id_t _segment_id,
    const gaspi_size_t       _size,
    const gaspi_group_t      _group);
void Probe_GPI_segment_create_Exit();

void Probe_GPI_write_Entry();
void Probe_GPI_write_Exit();

void Probe_GPI_allreduce_Entry(
    const gaspi_number_t   _num,
    const gaspi_datatype_t _datatyp,
    const gaspi_group_t    _group);
void Probe_GPI_allreduce_Exit();
