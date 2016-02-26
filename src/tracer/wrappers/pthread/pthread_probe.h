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

#ifndef PTHREAD_PROBE_H_INCLUDED
#define PTHREAD_PROBE_H_INCLUDED

void Extrae_pthread_instrument_locks (int value);
int Extrae_get_pthread_instrument_locks (void);

void Probe_pthread_Create_Entry (void *p);
void Probe_pthread_Create_Exit (void);
void Probe_pthread_Join_Entry (void);
void Probe_pthread_Join_Exit (void);
void Probe_pthread_Detach_Entry (void);
void Probe_pthread_Detach_Exit (void);

void Probe_pthread_Function_Entry (void *p);
void Probe_pthread_Function_Exit (void);

void Probe_pthread_Exit_Entry(void);

void Probe_pthread_rwlock_lockwr_Entry (void *p);
void Probe_pthread_rwlock_lockwr_Exit (void *p);
void Probe_pthread_rwlock_lockrd_Entry (void *p);
void Probe_pthread_rwlock_lockrd_Exit (void *p);
void Probe_pthread_rwlock_unlock_Entry (void *p);
void Probe_pthread_rwlock_unlock_Exit (void *p);

void Probe_pthread_mutex_lock_Entry (void *p);
void Probe_pthread_mutex_lock_Exit (void *p);
void Probe_pthread_mutex_unlock_Entry (void *p);
void Probe_pthread_mutex_unlock_Exit (void *p);

void Probe_pthread_cond_signal_Entry (void *p);
void Probe_pthread_cond_signal_Exit (void *p);
void Probe_pthread_cond_broadcast_Entry (void *p);
void Probe_pthread_cond_broadcast_Exit (void *p);
void Probe_pthread_cond_wait_Entry (void *p);
void Probe_pthread_cond_wait_Exit (void *p);

void Probe_pthread_Barrier_Wait_Entry (void);
void Probe_pthread_Barrier_Wait_Exit (void);

#endif
