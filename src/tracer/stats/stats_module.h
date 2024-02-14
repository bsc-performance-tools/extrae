#pragma once

#include "common.h"
#include "stats_types.h"
#include "OMP/omp_stats.h"

#if defined(MPI_SUPPORT)
# include "MPI/mpi_stats.h"
#else
# define MPI_BURST_STATS_COUNT 0
#endif

#define STATS_SIZE_PER_GROUP MAX(MPI_BURST_STATS_COUNT, OMP_BURST_STATS_COUNT)


/**
 * These are the routines that every runtime statistic has to implement
*/
typedef struct stats_vtable
{
  void (*reset) (int threadid, xtr_stats_t * self );
  void (*copyto) (int threadid, xtr_stats_t * self, struct xtr_stats_t * dest) ;
  xtr_stats_t * (*dup) (xtr_stats_t * self);
  void (*subtract) (int threadid, xtr_stats_t * self, xtr_stats_t * subtrahend, xtr_stats_t * destination);
  int (*get_positive_values_and_ids) ( int threadid, xtr_stats_t * self, INT32 * out_type, UINT64 * out_values);
  stats_info_t * (*get_ids_and_descriptions) ( void );
  void (*realloc) (xtr_stats_t * self, int new_num_threads);
  void (*free) (xtr_stats_t * self);
  int nevents;
}stats_vtable_st;


xtr_stats_t ** xtr_stats_initialize( void );

void xtr_stats_realloc (xtr_stats_t **stats, int old_num_threads, int new_num_threads );

void xtr_stats_change_nthreads(int old_num_threads, int new_num_threads);

void xtr_stats_reset(xtr_stats_t **stats_obj);

xtr_stats_t **xtr_stats_dup(xtr_stats_t **stats_obj);

void xtr_stats_copyto(xtr_stats_t **stats_obj_from, xtr_stats_t **stats_obj_to);

int xtr_stats_get_category(xtr_stats_t *stats_obj);

void xtr_stats_get_values(int threadid, xtr_stats_t **stats_obj, int out_num[], INT32 out_ids[][STATS_SIZE_PER_GROUP], UINT64 out_values[][STATS_SIZE_PER_GROUP]);

stats_info_t **xtr_stats_get_description_table(void);

void xtr_stats_subtract (int threadid, xtr_stats_t **minuend, xtr_stats_t **subtrahend, xtr_stats_t **destination );

void xtr_stats_free (xtr_stats_t **stats );

void xtr_stats_finalize();

void xtr_print_debug_stats ( int tid );
