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

#include "common.h"

#if !defined(HAVE__SYNC_ADD_AND_FETCH) || !defined(HAVE__SYNC_AND_AND_FETCH)
#if !defined(HAVE_PTHREAD_H)
#error "Pthread library is needed to avoid race condition in atomic counters"
#endif
#include <pthread.h>
#endif

typedef struct {
	long long value;
#if !defined(HAVE__SYNC_ADD_AND_FETCH) || !defined(HAVE__SYNC_AND_AND_FETCH)
	pthread_mutex_t mtx;
#endif
} xtr_atomic_counter_t;

#if defined(HAVE__SYNC_ADD_AND_FETCH) && defined(HAVE__SYNC_AND_AND_FETCH)

# define ATOMIC_COUNTER_INITIALIZER(counter, initial_value)        \
  static xtr_atomic_counter_t counter = {                          \
    initial_value                                                  \
  };

# define ATOMIC_COUNTER_INCREMENT(result, counter, increment)      \
{                                                                  \
  result = __sync_add_and_fetch(&(counter.value), increment);      \
}                                                                  \

# define ATOMIC_COUNTER_RESET(counter)                             \
{                                                                  \
  __sync_add_and_and(&(counter.value), 0);                         \
}

#else /* !defined(HAVE__SYNC_ADD_AND_FETCH) || !defined(HAVE__SYNC_AND_AND_FETCH) */

# define ATOMIC_COUNTER_INITIALIZER(counter, initial_value)        \
  static xtr_atomic_counter_t counter = {                          \
    initial_value,                                                 \
    PTHREAD_MUTEX_INITIALIZER                                      \
  };

# define ATOMIC_COUNTER_INCREMENT(result, counter, increment)      \
{                                                                  \
  pthread_mutex_lock (&(counter.mtx));                             \
  counter.value += increment;                                      \
  result = counter.value;                                          \
  pthread_mutex_unlock (&(counter.mtx));                           \
}

# define ATOMIC_COUNTER_RESET(counter)                             \
{                                                                  \
  pthread_mutex_lock (&(counter.mtx));                             \
  counter.value = 0;                                               \
  pthread_mutex_unlock (&(counter.mtx));                           \
}

#endif /* defined(HAVE__SYNC_ADD_AND_FETCH) && defined(HAVE__SYNC_AND_AND_FETCH) */

