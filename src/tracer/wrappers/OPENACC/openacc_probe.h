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

void Probe_OPENACC_device_init_start(int);
void Probe_OPENACC_device_init_end(int);
void Probe_OPENACC_device_shutdown_start(int);
void Probe_OPENACC_device_shutdown_end(int);
void Probe_OPENACC_enter_data_start(int);
void Probe_OPENACC_enter_data_end(int);
void Probe_OPENACC_exit_data_start(int);
void Probe_OPENACC_exit_data_end(int);
void Probe_OPENACC_create(int);
void Probe_OPENACC_delete(int);
void Probe_OPENACC_alloc(int);
void Probe_OPENACC_free(int);
void Probe_OPENACC_update_start(int);
void Probe_OPENACC_update_end(int);
void Probe_OPENACC_compute_construct_start(int);
void Probe_OPENACC_compute_construct_end(int);
void Probe_OPENACC_enqueue_launch_start(int);
void Probe_OPENACC_enqueue_launch_end(int);
void Probe_OPENACC_enqueue_upload_start(int);
void Probe_OPENACC_enqueue_upload_end(int);
void Probe_OPENACC_enqueue_download_start(int);
void Probe_OPENACC_enqueue_download_end(int);
void Probe_OPENACC_wait_start(int);
void Probe_OPENACC_wait_end(int);
