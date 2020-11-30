#ifndef __PTHREAD_REDIRECT_H__
#define __PTHREAD_REDIRECT_H__

#include "common.h"

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#ifdef HAVE_PTHREAD_H
# include <pthread.h>
#endif
#ifdef HAVE_DLFCN_H
# define __USE_GNU
# include <dlfcn.h>
# undef  __USE_GNU
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_TIME_H
# include <time.h>
#endif

#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h>

extern int (*pthread_create_real)(pthread_t*,const pthread_attr_t*,void *(*) (void *),void*);
extern int (*pthread_join_real)(pthread_t,void**);
extern int (*pthread_detach_real)(pthread_t);
extern void (*pthread_exit_real)(void*);
extern int (*pthread_barrier_wait_real)(pthread_barrier_t *barrier);

extern int (*pthread_mutex_lock_real)(pthread_mutex_t*);
extern int (*pthread_mutex_trylock_real)(pthread_mutex_t*);
extern int (*pthread_mutex_timedlock_real)(pthread_mutex_t*,const struct timespec *);
extern int (*pthread_mutex_unlock_real)(pthread_mutex_t*);

extern int (*pthread_cond_broadcast_real)(pthread_cond_t*);
extern int (*pthread_cond_timedwait_real)(pthread_cond_t*,pthread_mutex_t*,const struct timespec *);
extern int (*pthread_cond_signal_real)(pthread_cond_t*);
extern int (*pthread_cond_wait_real)(pthread_cond_t*,pthread_mutex_t*);

extern int (*pthread_rwlock_rdlock_real)(pthread_rwlock_t *);
extern int (*pthread_rwlock_tryrdlock_real)(pthread_rwlock_t *);
extern int (*pthread_rwlock_timedrdlock_real)(pthread_rwlock_t *, const struct timespec *);
extern int (*pthread_rwlock_wrlock_real)(pthread_rwlock_t *);
extern int (*pthread_rwlock_trywrlock_real)(pthread_rwlock_t *);
extern int (*pthread_rwlock_timedwrlock_real)(pthread_rwlock_t *, const struct timespec *);
extern int (*pthread_rwlock_unlock_real)(pthread_rwlock_t *);

int GetpThreadIdentifier (void);

// ============================= Locking helper ===================================
void mtx_lock_caller(pthread_mutex_t* lock, char* name, char const * caller_name);

void mtx_unlock_caller(pthread_mutex_t* lock, char* name, char const * caller_name);

void mtx_rw_wrlock_caller(pthread_rwlock_t* lock, char* name, char const * caller_name);

void mtx_rw_rdlock_caller(pthread_rwlock_t* lock, char* name, char const * caller_name);

void mtx_rw_unlock_caller(pthread_rwlock_t* lock, char* name, char const * caller_name);

#ifndef mtx_lock
#define mtx_lock(mtx) mtx_lock_caller(mtx, #mtx, __func__)
#endif

#ifndef mtx_unlock
#define mtx_unlock(mtx) mtx_unlock_caller(mtx, #mtx, __func__)
#endif

#ifndef mtx_rw_wrlock
#define mtx_rw_wrlock(mtx) mtx_rw_wrlock_caller(mtx, #mtx, __func__)
#endif

#ifndef mtx_rw_rdlock
#define mtx_rw_rdlock(mtx) mtx_rw_rdlock_caller(mtx, #mtx, __func__)
#endif

#ifndef mtx_rw_unlock
#define mtx_rw_unlock(mtx) mtx_rw_unlock_caller(mtx, #mtx, __func__)
#endif

#endif