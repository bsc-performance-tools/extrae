#define _GNU_SOURCE
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

#include "common.h"

#include <math.h>
#include <unistd.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "wrapper.h"
#include "openshmem_wrappers.h"
#include "openshmem_probes.h"
#include "openshmem_events.h"
#include "utils.h"

/****************************************\
 ***     POINTERS TO REAL SYMBOLS     ***
\****************************************/

static void (*start_pes_real) (int npes) = NULL;
static int (*shmem_my_pe_real) (void) = NULL;
static int (*_my_pe_real) (void) = NULL;
static int (*shmem_n_pes_real) (void) = NULL;
static int (*_num_pes_real) (void) = NULL;
static int (*shmem_pe_accessible_real) (int pe) = NULL;
static int (*shmem_addr_accessible_real) (void *addr, int pe) = NULL;
static void * (*shmem_ptr_real) (void *target, int pe) = NULL;
static void * (*shmalloc_real) (size_t size) = NULL;
static void (*shfree_real) (void *ptr) = NULL;
static void * (*shrealloc_real) (void *ptr, size_t size) = NULL;
static void * (*shmemalign_real) (size_t alignment, size_t size) = NULL;
static void (*shmem_double_put_real) (double *target, const double *source, size_t len, int pe) = NULL;
static void (*shmem_float_put_real) (float *target, const float *source, size_t len, int pe) = NULL;
static void (*shmem_int_put_real) (int *target, const int *source, size_t len, int pe) = NULL;
static void (*shmem_long_put_real) (long *target, const long *source, size_t len, int pe) = NULL;
static void (*shmem_longdouble_put_real) (long double *target, const long double *source, size_t len,int pe) = NULL;
static void (*shmem_longlong_put_real) (long long *target, const long long *source, size_t len, int pe) = NULL;
static void (*shmem_put32_real) (void *target, const void *source, size_t len, int pe) = NULL;
static void (*shmem_put64_real) (void *target, const void *source, size_t len, int pe) = NULL;
static void (*shmem_put128_real) (void *target, const void *source, size_t len, int pe) = NULL;
static void (*shmem_putmem_real) (void *target, const void *source, size_t len, int pe) = NULL;
static void (*shmem_short_put_real) (short*target, const short*source, size_t len, int pe) = NULL;
static void (*shmem_char_p_real) (char *addr, char value, int pe) = NULL;
static void (*shmem_short_p_real) (short *addr, short value, int pe) = NULL;
static void (*shmem_int_p_real) (int *addr, int value, int pe) = NULL;
static void (*shmem_long_p_real) (long *addr, long value, int pe) = NULL;
static void (*shmem_longlong_p_real) (long long *addr, long long value, int pe) = NULL;
static void (*shmem_float_p_real) (float *addr, float value, int pe) = NULL;
static void (*shmem_double_p_real) (double *addr, double value, int pe) = NULL;
static void (*shmem_longdouble_p_real) (long double *addr, long double value, int pe) = NULL;
static void (*shmem_double_iput_real) (double *target, const double *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe) = NULL;
static void (*shmem_float_iput_real) (float *target, const float *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe) = NULL;
static void (*shmem_int_iput_real) (int *target, const int *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe) = NULL;
static void (*shmem_iput32_real) (void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe) = NULL;
static void (*shmem_iput64_real) (void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe) = NULL;
static void (*shmem_iput128_real) (void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe) = NULL;
static void (*shmem_long_iput_real) (long *target, const long *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe) = NULL;
static void (*shmem_longdouble_iput_real) (long double *target, const long double *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe) = NULL;
static void (*shmem_longlong_iput_real) (long long *target, const long long *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe) = NULL;
static void (*shmem_short_iput_real) (short *target, const short *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe) = NULL;
static void (*shmem_double_get_real) (double *target, const double *source, size_t nelems, int pe) = NULL;
static void (*shmem_float_get_real) (float *target, const float *source, size_t nelems, int pe) = NULL;
static void (*shmem_get32_real) (void *target, const void *source, size_t nelems, int pe) = NULL;
static void (*shmem_get64_real) (void *target, const void *source, size_t nelems, int pe) = NULL;
static void (*shmem_get128_real) (void *target, const void *source, size_t nelems, int pe) = NULL;
static void (*shmem_getmem_real) (void *target, const void *source, size_t nelems, int pe) = NULL;
static void (*shmem_int_get_real) (int *target, const int *source, size_t nelems, int pe) = NULL;
static void (*shmem_long_get_real) (long *target, const long *source, size_t nelems, int pe) = NULL;
static void (*shmem_longdouble_get_real) (long double *target, const long double *source, size_t nelems, int pe) = NULL;
static void (*shmem_longlong_get_real) (long long *target, const long long *source, size_t nelems, int pe) = NULL;
static void (*shmem_short_get_real) (short *target, const short *source, size_t nelems, int pe) = NULL;
static char (*shmem_char_g_real) (char *addr, int pe) = NULL;
static short (*shmem_short_g_real) (short *addr, int pe) = NULL;
static int (*shmem_int_g_real) (int *addr, int pe) = NULL;
static long (*shmem_long_g_real) (long *addr, int pe) = NULL;
static long long (*shmem_longlong_g_real) (long long *addr, int pe) = NULL;
static float (*shmem_float_g_real) (float *addr, int pe) = NULL;
static double (*shmem_double_g_real) (double *addr, int pe) = NULL;
static long double (*shmem_longdouble_g_real) (long double *addr, int pe) = NULL;
static void (*shmem_double_iget_real) (double *target, const double *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe) = NULL;
static void (*shmem_float_iget_real) (float *target, const float *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe) = NULL;
static void (*shmem_iget32_real) (void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe) = NULL;
static void (*shmem_iget64_real) (void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe) = NULL;
static void (*shmem_iget128_real) (void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe) = NULL;
static void (*shmem_int_iget_real) (int *target, const int *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe) = NULL;
static void (*shmem_long_iget_real) (long *target, const long *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe) = NULL;
static void (*shmem_longdouble_iget_real) (long double *target, const long double *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe) = NULL;
static void (*shmem_longlong_iget_real) (long long *target, const long long *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe) = NULL;
static void (*shmem_short_iget_real) (short *target, const short *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe) = NULL;
static void (*shmem_int_add_real) (int *target, int value, int pe) = NULL;
static void (*shmem_long_add_real) (long *target, long value, int pe) = NULL;
static void (*shmem_longlong_add_real) (long long *target, long long value, int pe) = NULL;
static int (*shmem_int_cswap_real) (int *target, int cond, int value, int pe) = NULL;
static long (*shmem_long_cswap_real) (long *target, long cond, long value, int pe) = NULL;
static long long (*shmem_longlong_cswap_real) (long long *target, long long cond, long long value, int pe) = NULL;
static double (*shmem_double_swap_real) (double *target, double value, int pe) = NULL;
static float (*shmem_float_swap_real) (float *target, float value, int pe) = NULL;
static int (*shmem_int_swap_real) (int *target, int value, int pe) = NULL;
static long (*shmem_long_swap_real) (long *target, long value, int pe) = NULL;
static long long (*shmem_longlong_swap_real) (long long *target, long long value, int pe) = NULL;
static long (*shmem_swap_real) (long *target, long value, int pe) = NULL;
static int (*shmem_int_finc_real) (int *target, int pe) = NULL;
static long (*shmem_long_finc_real) (long *target, int pe) = NULL;
static long long (*shmem_longlong_finc_real) (long long *target, int pe) = NULL;
static void (*shmem_int_inc_real) (int *target, int pe) = NULL;
static void (*shmem_long_inc_real) (long *target, int pe) = NULL;
static void (*shmem_longlong_inc_real) (long long *target, int pe) = NULL;
static int (*shmem_int_fadd_real) (int *target, int value, int pe) = NULL;
static long (*shmem_long_fadd_real) (long *target, long value, int pe) = NULL;
static long long (*shmem_longlong_fadd_real) (long long *target, long long value, int pe) = NULL;
static void (*shmem_barrier_all_real) (void) = NULL;
static void (*shmem_barrier_real) (int PE_start, int logPE_stride, int PE_size, long *pSync) = NULL;
static void (*shmem_broadcast32_real) (void *target, const void *source, size_t nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long *pSync) = NULL;
static void (*shmem_broadcast64_real) (void *target, const void *source, size_t nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long *pSync) = NULL;
static void (*shmem_collect32_real) (void *target, const void *source, size_t nelems, int PE_start, int logPE_stride, int PE_size, long *pSync) = NULL;
static void (*shmem_collect64_real) (void *target, const void *source, size_t nelems, int PE_start, int logPE_stride, int PE_size, long *pSync) = NULL;
static void (*shmem_fcollect32_real) (void *target, const void *source, size_t nelems, int PE_start, int logPE_stride, int PE_size, long *pSync) = NULL;
static void (*shmem_fcollect64_real) (void *target, const void *source, size_t nelems, int PE_start, int logPE_stride, int PE_size, long *pSync) = NULL;
static void (*shmem_int_and_to_all_real) (int *target, int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync) = NULL;
static void (*shmem_long_and_to_all_real) (long *target, long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync) = NULL;
static void (*shmem_longlong_and_to_all_real) (long long *target, long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync) = NULL;
static void (*shmem_short_and_to_all_real) (short *target, short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync) = NULL;
static void (*shmem_double_max_to_all_real) (double *target, double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, double *pWrk, long *pSync) = NULL;
static void (*shmem_float_max_to_all_real) (float *target, float *source, int nreduce, int PE_start, int logPE_stride, int PE_size, float *pWrk, long *pSync) = NULL;
static void (*shmem_int_max_to_all_real) (int *target, int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync) = NULL;
static void (*shmem_long_max_to_all_real) (long *target, long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync) = NULL;
static void (*shmem_longdouble_max_to_all_real) (long double *target, long double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double *pWrk, long *pSync) = NULL;
static void (*shmem_longlong_max_to_all_real) (long long *target, long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync) = NULL;
static void (*shmem_short_max_to_all_real) (short *target, short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync) = NULL;
static void (*shmem_double_min_to_all_real) (double *target, double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, double *pWrk, long *pSync) = NULL;
static void (*shmem_int_wait_real) (int *ivar, int cmp_value) = NULL;
static void (*shmem_int_wait_until_real) (int *ivar, int cmp, int cmp_value) = NULL;
static void (*shmem_long_wait_real) (long *ivar, long cmp_value) = NULL;
static void (*shmem_long_wait_until_real) (long *ivar, int cmp, long cmp_value) = NULL;
static void (*shmem_longlong_wait_real) (long long *ivar, long long cmp_value) = NULL;
static void (*shmem_longlong_wait_until_real) (long long *ivar, int cmp, long long cmp_value) = NULL;
static void (*shmem_short_wait_real) (short *ivar, short cmp_value) = NULL;
static void (*shmem_short_wait_until_real) (short *ivar, int cmp, short cmp_value) = NULL;
static void (*shmem_wait_real) (long *ivar, long cmp_value) = NULL;
static void (*shmem_wait_until_real) (long *ivar, int cmp, long cmp_value) = NULL;
static void (*shmem_fence_real) (void) = NULL;
static void (*shmem_quiet_real) (void) = NULL;
static void (*shmem_clear_lock_real) (long *lock) = NULL;
static void (*shmem_set_lock_real) (long *lock) = NULL;
static int (*shmem_test_lock_real) (long *lock) = NULL;
static void (*shmem_clear_cache_inv_real) (void) = NULL;
static void (*shmem_set_cache_inv_real) (void) = NULL;
static void (*shmem_clear_cache_line_inv_real) (void *target) = NULL;
static void (*shmem_set_cache_line_inv_real) (void *target) = NULL;
static void (*shmem_udcflush_real) (void) = NULL;
static void (*shmem_udcflush_line_real) (void *target) = NULL;

static unsigned Extrae_OPENSHMEM_NumTasks(void)
{
  static int run = FALSE;
  static unsigned ntasks = 0;

  if (!run)
  {
    ntasks = shmem_n_pes_real();
    run    = TRUE;
  }
  return ntasks;
}

static unsigned Extrae_OPENSHMEM_TaskID(void)
{
  static int run = FALSE;
  static unsigned rank;

  if (!run)
  {
    rank = _my_pe_real();
    run  = TRUE;
  }
  return rank;
}

static void Extrae_OPENSHMEM_Barrier(void)
{
  shmem_barrier_all_real();
}

static void Extrae_OPENSHMEM_Finalize(void)
{
  return;
}

#include "auto_fini.h"
void shmem_finalize()
{
  Extrae_auto_library_fini();
}

static char * OPENSHMEM_Distribute_XML_File (int rank, int world_size, char *origen)
{
  char hostname[1024];
  long *pSync = NULL;
  char *result_file = NULL;
  char *target_file = NULL;
  int has_hostname = FALSE;
  int* file_size_source;
  int* file_size_target;
  int fd;
  void *storage;
  int storage_size = 0;
  int i;

  pSync = (long *) shmalloc_real( sizeof(long) * _SHMEM_BCAST_SYNC_SIZE );
  file_size_source = (int *) shmalloc_real( sizeof(int) );
  file_size_target = (int *) shmalloc_real( sizeof(int) );

  *file_size_source = 0;
  *file_size_target = 0;

  has_hostname = gethostname(hostname, 1024 - 1) == 0;

  /* If no other tasks are running, just return the same file */
  if (world_size == 1)
  {
    /* Copy the filename */
    result_file = strdup (origen);
    if (result_file == NULL)
    {
      fprintf (stderr, PACKAGE_NAME": Cannot obtain memory for the XML file!\n");
      exit (0);
    }
    return result_file;
  }

  for (i=0; i < _SHMEM_BCAST_SYNC_SIZE; i++) {
    pSync[i] = _SHMEM_SYNC_VALUE;
  }
  shmem_barrier_all_real(); /* Wait for all PEs to initialize pSync */

  shmem_broadcast32_real(file_size_target, file_size_source, 1, 0, 0, 0, world_size, pSync);

  if (rank == 0)
  {
                /* Copy the filename */
                result_file = (char*) malloc ((strlen(origen)+1)*sizeof(char));
                if (result_file == NULL)
                {
                        fprintf (stderr, PACKAGE_NAME": Cannot obtain memory for the XML file!\n");
                        exit (0);
                }
                memset (result_file, 0, (strlen(origen)+1)*sizeof(char));
                strncpy (result_file, origen, strlen(origen));

                /* Open the file */
                fd = open (result_file, O_RDONLY);

                /* If open succeeds, read the size of the file */
		if (fd != -1)
		{
		    *file_size_source = lseek (fd, 0, SEEK_END);
		    lseek (fd, 0, SEEK_SET);
		}

                /* Send the size */
                shmem_broadcast32_real(file_size_target, file_size_source, 1, 0, 0, 0, world_size, pSync);

		if (fd < 0 || *file_size_source == 0)
		{
		    fprintf (stderr, PACKAGE_NAME": Cannot open XML configuration file (%s)!\n", result_file);
		    exit (0);
		}

                storage_size = sizeof(int) * (int)(ceil( (float)*file_size_source / (float)sizeof(int) ));


                /* Allocate & Read the file */
                storage = (void *) shmalloc_real (storage_size * sizeof(char));
                if (storage == NULL)
                {
                        fprintf (stderr, PACKAGE_NAME": Cannot obtain memory for the XML distribution!\n");
                        exit (0);
                }
                if (*file_size_source != read (fd, (char *)storage, *file_size_source))
                {
                        fprintf (stderr, PACKAGE_NAME": Unable to read XML file for its distribution on host %s\n", has_hostname?hostname:"unknown");
                        exit (0);
                }

                /* Send the file */
                shmem_broadcast32_real( storage, storage, storage_size/4, 0, 0, 0, world_size, pSync );

                /* Close the file */
                close (fd);
                shfree_real (storage);

                return result_file;
  }
  else  
  {
                /* Receive the size */
                shmem_broadcast32_real(file_size_target, file_size_source, 1, 0, 0, 0, world_size, pSync);

		if (*file_size_target <= 0)
		{
		    exit (0);
		}

                storage_size = sizeof(int) * (int)(ceil( (float)*file_size_target / (float)sizeof(int) ));
                storage = (char*) shmalloc_real (storage_size * sizeof(char));
                if (storage == NULL)
                {
                        fprintf (stderr, PACKAGE_NAME": Cannot obtain memory for the XML distribution!\n");
                        exit (0);
                }

                /* Build the temporal file pattern */
                if (getenv("TMPDIR"))
                {
                        int len = 14 + strlen(getenv("TMPDIR")) + 1;

                        /* If TMPDIR exists but points to non-existent directory, create it */
                        if (!directory_exists (getenv("TMPDIR")))
                                mkdir_recursive (getenv("TMPDIR"));

                        /* 14 is the length from /XMLFileXXXXXX */
                        result_file = (char*) malloc (len * sizeof(char));
                        snprintf (result_file, len, "%s/XMLFileXXXXXX", getenv ("TMPDIR"));
                }
                else
                {
                        /* 13 is the length from XMLFileXXXXXX */
                        result_file = (char*) malloc ((13+1)*sizeof(char));
                        sprintf (result_file, "XMLFileXXXXXX");
                }

                /* Create the temporal file */
                fd = mkstemp (result_file);

                /* Receive the file */
                shmem_broadcast32_real( storage, storage, storage_size/4, 0, 0, 0, world_size, pSync );

                if (*file_size_target != write (fd, (char *)storage, *file_size_target))
                {
                        fprintf (stderr, PACKAGE_NAME": Unable to write XML file for its distribution (%s) - host %s\n", result_file, has_hostname?hostname:"unknown");
                        perror("write");
                        exit (0);
                }

                /* Close the file, free and return it! */
                close (fd);
                shfree_real (storage);

                return result_file;

  }

  shfree_real( pSync );
  shfree_real( file_size_source );
  shfree_real( file_size_target );

  return NULL;
}

char **TasksNodes = NULL;

static void OPENSHMEM_Gather_Nodes_Info (void)
{
  int i = 0;
  int hostname_length = 1024;
  int num_tasks = Extrae_get_num_tasks();
  void *hostname;
  void *all_hostnames;

  hostname = (void *) shmalloc_real( hostname_length * sizeof(char) );
  all_hostnames = (void *) shmalloc_real( hostname_length * sizeof(char) * num_tasks );
  bzero( hostname, hostname_length * sizeof(char) );
  bzero( all_hostnames, hostname_length * sizeof(char) * num_tasks );

  /* Get processor name */
  gethostname((char *)hostname, hostname_length - 1);

  /* Change spaces " " into underscores "_" (BLG nodes use to have spaces in their names) */
  for (i = 0; i < hostname_length; i++)
    if (' ' == ((char *)hostname)[i])
      ((char *)hostname)[i] = '_';

  /* Share information among all tasks */
  long *psync;
  psync = (long *) shmalloc_real (sizeof(long) * _SHMEM_COLLECT_SYNC_SIZE);
  for(i=0;i<_SHMEM_COLLECT_SYNC_SIZE;i++)
    psync[i] = _SHMEM_SYNC_VALUE;
  shmem_barrier_all_real();

  shmem_fcollect32_real(all_hostnames, hostname, (hostname_length/4), 0, 0, num_tasks, psync);

  /* Store the information in a global array */
  TasksNodes = (char **)malloc (num_tasks * sizeof(char *));
  if (TasksNodes == NULL)
  {
    fprintf (stderr, ": Fatal error! Cannot allocate memory for nodes info\n");
    exit (-1);
  }

  for(i=0; i<num_tasks; i++)
  {
    TasksNodes[i] = (char *)malloc(hostname_length * sizeof(char));

    if (TasksNodes[i] == NULL)
    {
      fprintf (stderr, ": Fatal error! Cannot allocate memory for node info %u\n", i);
      exit (-1);
    }
    strncpy (TasksNodes[i], &(((char *)all_hostnames)[i * hostname_length]), hostname_length);
  }

  shfree_real( psync );
  shfree_real( hostname );
  shfree_real( all_hostnames );

  return;
}

void OPENSHMEM_remove_file_list (int all)
{
        char tmpname[1024];

        if (all || (!all && TASKID == 0))
        {
                sprintf (tmpname, "%s/%s%s", final_dir, appl_name, EXT_MPITS);
                unlink (tmpname);
        }
}

char *MpitsFileName = NULL;
#include "threadinfo.h"

int OPENSHMEM_Generate_Task_File_list( char **node_list )
{
	int i;
	unsigned ret;
	int *pid;
	int *task_id;
	int *num_threads;
        int  num_tasks;
        int *all_pids;
        int *all_task_ids;
        int *all_num_threads;
	char *thread_names_source;
	char *thread_names_target;
	char tmpname[1024];

        num_tasks   = Extrae_get_num_tasks();
	pid         = (int *) shmalloc_real( sizeof(int) );
	task_id     = (int *) shmalloc_real( sizeof(int) );
	num_threads = (int *) shmalloc_real( sizeof(int) );
	thread_names_source = (char *) shmalloc_real( Backend_getMaximumOfThreads()*THREAD_INFO_NAME_LEN*sizeof(char) );
	thread_names_target = (char *) shmalloc_real( Backend_getMaximumOfThreads()*THREAD_INFO_NAME_LEN*sizeof(char) );

	*pid = getpid();
	*task_id = TASKID;
	*num_threads = Backend_getMaximumOfThreads();

	all_pids        = (int *) shmalloc_real( num_tasks * sizeof(int) );
	all_task_ids    = (int *) shmalloc_real( num_tasks * sizeof(int) );
	all_num_threads = (int *) shmalloc_real( num_tasks * sizeof(int) );

	/* Share information among all tasks */
	long *psync;
	psync = (long *) shmalloc_real (sizeof(long) * _SHMEM_COLLECT_SYNC_SIZE);
	for(i=0;i<_SHMEM_COLLECT_SYNC_SIZE;i++)
	psync[i] = _SHMEM_SYNC_VALUE;
	shmem_barrier_all_real();

        /* Share PID and number of threads of each task */
	shmem_fcollect32_real(all_pids, pid, 1, 0, 0, num_tasks, psync);
	shmem_fcollect32_real(all_task_ids, task_id, 1, 0, 0, num_tasks, psync);
	shmem_fcollect32_real(all_num_threads, num_threads, 1, 0, 0, num_tasks, psync);

	if (TASKID != 0)
	{
                if (thread_names_source == NULL)
                {
                        fprintf (stderr, "Fatal error! Cannot allocate memory to transfer thread names\n");
                        exit (-1);
                }
                for (i = 0; i < Backend_getMaximumOfThreads(); i++)
                        memcpy (&thread_names_source[i*THREAD_INFO_NAME_LEN], Extrae_get_thread_name(i), THREAD_INFO_NAME_LEN);
	}

	shmem_barrier_all_real();

	if (TASKID == 0)	
	{
		int fd;
		int thid; 

		sprintf (tmpname, "%s/%s.mpits", final_dir, appl_name);
		MpitsFileName = strdup( tmpname );
		fd = open (MpitsFileName, O_RDWR | O_CREAT | O_TRUNC, 0644);
		if (fd == -1)
		{
			return -1;
		}
		for (i = 0; i < num_tasks; i ++)
		{
			char tmpline[2048];
			int PID = all_pids[i];
			int TID = all_task_ids[i];
			int NTHREADS = all_num_threads[i];

			if (i == 0)
			{
                                /* If Im processing MASTER, I know my threads and their names */
                                for (thid = 0; thid < NTHREADS; thid++)
                                {
                                        FileName_PTT(tmpname, Get_FinalDir(TID), appl_name, node_list[i], PID, TID, thid, EXT_MPIT);
                                        sprintf (tmpline, "%s named %s\n", tmpname, Extrae_get_thread_name(thid));
                                        ret = write (fd, tmpline, strlen (tmpline));
                                        if (ret != strlen (tmpline))
                                        {
                                                close (fd);
                                                return -1;
                                        }
                                }
			}
			else
			{
                                /* If Im not processing MASTER, I have to ask for threads and their names */
				shmem_getmem_real(thread_names_target, thread_names_source, Backend_getMaximumOfThreads()*THREAD_INFO_NAME_LEN*sizeof(char), i);
                                for (thid = 0; thid < NTHREADS; thid++)
                                {
                                        FileName_PTT(tmpname, Get_FinalDir(TID), appl_name, node_list[i], PID, TID, thid, EXT_MPIT);
                                        sprintf (tmpline, "%s named %s\n", tmpname, &thread_names_target[thid*THREAD_INFO_NAME_LEN]);
                                        ret = write (fd, tmpline, strlen (tmpline));
                                        if (ret != strlen (tmpline))
                                        {
                                                close (fd);
                                                return -1;
                                        }
                                }
			}
		}
		close (fd);
	}

	shfree_real( psync );
	shfree_real( pid );
	shfree_real( task_id );
	shfree_real( num_threads );
	shfree_real( all_pids );
	shfree_real( all_task_ids );
	shfree_real( all_num_threads );
	shfree_real( thread_names_source );
	shfree_real( thread_names_target );
	return 0;
}

/****************************************\
 ***          SYMBOLS HOOK-UP         ***
\****************************************/

static void Get_OPENSHMEM_Hook_Points (int rank)
{
#if defined(PIC)
# if defined(__APPLE__)
# error "Search for the appropriate library to load the symbols dynamically"
# else
  void *lib = RTLD_NEXT;
# endif /* __APPLE__ */

  /* Obtain @ for start_pes */
  start_pes_real = 
    (void (*)(int))
    dlsym( lib, "start_pes" );
  if (start_pes_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find start_pes in DSOs!!\n");

  /* Obtain @ for shmem_my_pe */
  shmem_my_pe_real = 
    (int (*)(void))
    dlsym( lib, "shmem_my_pe" );
  if (shmem_my_pe_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_my_pe in DSOs!!\n");

  /* Obtain @ for _my_pe */
  _my_pe_real = 
    (int (*)(void))
    dlsym( lib, "_my_pe" );
  if (_my_pe_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find _my_pe in DSOs!!\n");

  /* Obtain @ for shmem_n_pes */
  shmem_n_pes_real = 
    (int (*)(void))
    dlsym( lib, "shmem_n_pes" );
  if (shmem_n_pes_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_n_pes in DSOs!!\n");

  /* Obtain @ for _num_pes */
  _num_pes_real = 
    (int (*)(void))
    dlsym( lib, "_num_pes" );
  if (_num_pes_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find _num_pes in DSOs!!\n");

  /* Obtain @ for shmem_pe_accessible */
  shmem_pe_accessible_real = 
    (int (*)(int))
    dlsym( lib, "shmem_pe_accessible" );
  if (shmem_pe_accessible_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_pe_accessible in DSOs!!\n");

  /* Obtain @ for shmem_addr_accessible */
  shmem_addr_accessible_real = 
    (int (*)(void *, int))
    dlsym( lib, "shmem_addr_accessible" );
  if (shmem_addr_accessible_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_addr_accessible in DSOs!!\n");

  /* Obtain @ for shmem_ptr */
  shmem_ptr_real = 
    (void * (*)(void *, int))
    dlsym( lib, "shmem_ptr" );
  if (shmem_ptr_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_ptr in DSOs!!\n");

  /* Obtain @ for shmalloc */
  shmalloc_real = 
    (void * (*)(size_t))
    dlsym( lib, "shmalloc" );
  if (shmalloc_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmalloc in DSOs!!\n");

  /* Obtain @ for shfree */
  shfree_real = 
    (void (*)(void *))
    dlsym( lib, "shfree" );
  if (shfree_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shfree in DSOs!!\n");

  /* Obtain @ for shrealloc */
  shrealloc_real = 
    (void * (*)(void *, size_t))
    dlsym( lib, "shrealloc" );
  if (shrealloc_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shrealloc in DSOs!!\n");

  /* Obtain @ for shmemalign */
  shmemalign_real = 
    (void * (*)(size_t, size_t))
    dlsym( lib, "shmemalign" );
  if (shmemalign_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmemalign in DSOs!!\n");

  /* Obtain @ for shmem_double_put */
  shmem_double_put_real = 
    (void (*)(double *, const double *, size_t, int))
    dlsym( lib, "shmem_double_put" );
  if (shmem_double_put_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_double_put in DSOs!!\n");

  /* Obtain @ for shmem_float_put */
  shmem_float_put_real = 
    (void (*)(float *, const float *, size_t, int))
    dlsym( lib, "shmem_float_put" );
  if (shmem_float_put_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_float_put in DSOs!!\n");

  /* Obtain @ for shmem_int_put */
  shmem_int_put_real = 
    (void (*)(int *, const int *, size_t, int))
    dlsym( lib, "shmem_int_put" );
  if (shmem_int_put_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_int_put in DSOs!!\n");

  /* Obtain @ for shmem_long_put */
  shmem_long_put_real = 
    (void (*)(long *, const long *, size_t, int))
    dlsym( lib, "shmem_long_put" );
  if (shmem_long_put_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_long_put in DSOs!!\n");

  /* Obtain @ for shmem_longdouble_put */
  shmem_longdouble_put_real = 
    (void (*)(long double *, const long double *, size_t, int))
    dlsym( lib, "shmem_longdouble_put" );
  if (shmem_longdouble_put_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_longdouble_put in DSOs!!\n");

  /* Obtain @ for shmem_longlong_put */
  shmem_longlong_put_real = 
    (void (*)(long long *, const long long *, size_t, int))
    dlsym( lib, "shmem_longlong_put" );
  if (shmem_longlong_put_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_longlong_put in DSOs!!\n");

  /* Obtain @ for shmem_put32 */
  shmem_put32_real = 
    (void (*)(void *, const void *, size_t, int))
    dlsym( lib, "shmem_put32" );
  if (shmem_put32_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_put32 in DSOs!!\n");

  /* Obtain @ for shmem_put64 */
  shmem_put64_real = 
    (void (*)(void *, const void *, size_t, int))
    dlsym( lib, "shmem_put64" );
  if (shmem_put64_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_put64 in DSOs!!\n");

  /* Obtain @ for shmem_put128 */
  shmem_put128_real = 
    (void (*)(void *, const void *, size_t, int))
    dlsym( lib, "shmem_put128" );
  if (shmem_put128_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_put128 in DSOs!!\n");

  /* Obtain @ for shmem_putmem */
  shmem_putmem_real = 
    (void (*)(void *, const void *, size_t, int))
    dlsym( lib, "shmem_putmem" );
  if (shmem_putmem_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_putmem in DSOs!!\n");

  /* Obtain @ for shmem_short_put */
  shmem_short_put_real = 
    (void (*)(short*, const short*, size_t, int))
    dlsym( lib, "shmem_short_put" );
  if (shmem_short_put_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_short_put in DSOs!!\n");

  /* Obtain @ for shmem_char_p */
  shmem_char_p_real = 
    (void (*)(char *, char, int))
    dlsym( lib, "shmem_char_p" );
  if (shmem_char_p_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_char_p in DSOs!!\n");

  /* Obtain @ for shmem_short_p */
  shmem_short_p_real = 
    (void (*)(short *, short, int))
    dlsym( lib, "shmem_short_p" );
  if (shmem_short_p_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_short_p in DSOs!!\n");

  /* Obtain @ for shmem_int_p */
  shmem_int_p_real = 
    (void (*)(int *, int, int))
    dlsym( lib, "shmem_int_p" );
  if (shmem_int_p_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_int_p in DSOs!!\n");

  /* Obtain @ for shmem_long_p */
  shmem_long_p_real = 
    (void (*)(long *, long, int))
    dlsym( lib, "shmem_long_p" );
  if (shmem_long_p_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_long_p in DSOs!!\n");

  /* Obtain @ for shmem_longlong_p */
  shmem_longlong_p_real = 
    (void (*)(long long *, long long, int))
    dlsym( lib, "shmem_longlong_p" );
  if (shmem_longlong_p_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_longlong_p in DSOs!!\n");

  /* Obtain @ for shmem_float_p */
  shmem_float_p_real = 
    (void (*)(float *, float, int))
    dlsym( lib, "shmem_float_p" );
  if (shmem_float_p_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_float_p in DSOs!!\n");

  /* Obtain @ for shmem_double_p */
  shmem_double_p_real = 
    (void (*)(double *, double, int))
    dlsym( lib, "shmem_double_p" );
  if (shmem_double_p_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_double_p in DSOs!!\n");

  /* Obtain @ for shmem_longdouble_p */
  shmem_longdouble_p_real = 
    (void (*)(long double *, long double, int))
    dlsym( lib, "shmem_longdouble_p" );
  if (shmem_longdouble_p_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_longdouble_p in DSOs!!\n");

  /* Obtain @ for shmem_double_iput */
  shmem_double_iput_real = 
    (void (*)(double *, const double *, ptrdiff_t, ptrdiff_t, size_t, int))
    dlsym( lib, "shmem_double_iput" );
  if (shmem_double_iput_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_double_iput in DSOs!!\n");

  /* Obtain @ for shmem_float_iput */
  shmem_float_iput_real = 
    (void (*)(float *, const float *, ptrdiff_t, ptrdiff_t, size_t, int))
    dlsym( lib, "shmem_float_iput" );
  if (shmem_float_iput_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_float_iput in DSOs!!\n");

  /* Obtain @ for shmem_int_iput */
  shmem_int_iput_real = 
    (void (*)(int *, const int *, ptrdiff_t, ptrdiff_t, size_t, int))
    dlsym( lib, "shmem_int_iput" );
  if (shmem_int_iput_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_int_iput in DSOs!!\n");

  /* Obtain @ for shmem_iput32 */
  shmem_iput32_real = 
    (void (*)(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int))
    dlsym( lib, "shmem_iput32" );
  if (shmem_iput32_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_iput32 in DSOs!!\n");

  /* Obtain @ for shmem_iput64 */
  shmem_iput64_real = 
    (void (*)(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int))
    dlsym( lib, "shmem_iput64" );
  if (shmem_iput64_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_iput64 in DSOs!!\n");

  /* Obtain @ for shmem_iput128 */
  shmem_iput128_real = 
    (void (*)(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int))
    dlsym( lib, "shmem_iput128" );
  if (shmem_iput128_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_iput128 in DSOs!!\n");

  /* Obtain @ for shmem_long_iput */
  shmem_long_iput_real = 
    (void (*)(long *, const long *, ptrdiff_t, ptrdiff_t, size_t, int))
    dlsym( lib, "shmem_long_iput" );
  if (shmem_long_iput_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_long_iput in DSOs!!\n");

  /* Obtain @ for shmem_longdouble_iput */
  shmem_longdouble_iput_real = 
    (void (*)(long double *, const long double *, ptrdiff_t, ptrdiff_t, size_t, int))
    dlsym( lib, "shmem_longdouble_iput" );
  if (shmem_longdouble_iput_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_longdouble_iput in DSOs!!\n");

  /* Obtain @ for shmem_longlong_iput */
  shmem_longlong_iput_real = 
    (void (*)(long long *, const long long *, ptrdiff_t, ptrdiff_t, size_t, int))
    dlsym( lib, "shmem_longlong_iput" );
  if (shmem_longlong_iput_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_longlong_iput in DSOs!!\n");

  /* Obtain @ for shmem_short_iput */
  shmem_short_iput_real = 
    (void (*)(short *, const short *, ptrdiff_t, ptrdiff_t, size_t, int))
    dlsym( lib, "shmem_short_iput" );
  if (shmem_short_iput_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_short_iput in DSOs!!\n");

  /* Obtain @ for shmem_double_get */
  shmem_double_get_real = 
    (void (*)(double *, const double *, size_t, int))
    dlsym( lib, "shmem_double_get" );
  if (shmem_double_get_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_double_get in DSOs!!\n");

  /* Obtain @ for shmem_float_get */
  shmem_float_get_real = 
    (void (*)(float *, const float *, size_t, int))
    dlsym( lib, "shmem_float_get" );
  if (shmem_float_get_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_float_get in DSOs!!\n");

  /* Obtain @ for shmem_get32 */
  shmem_get32_real = 
    (void (*)(void *, const void *, size_t, int))
    dlsym( lib, "shmem_get32" );
  if (shmem_get32_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_get32 in DSOs!!\n");

  /* Obtain @ for shmem_get64 */
  shmem_get64_real = 
    (void (*)(void *, const void *, size_t, int))
    dlsym( lib, "shmem_get64" );
  if (shmem_get64_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_get64 in DSOs!!\n");

  /* Obtain @ for shmem_get128 */
  shmem_get128_real = 
    (void (*)(void *, const void *, size_t, int))
    dlsym( lib, "shmem_get128" );
  if (shmem_get128_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_get128 in DSOs!!\n");

  /* Obtain @ for shmem_getmem */
  shmem_getmem_real = 
    (void (*)(void *, const void *, size_t, int))
    dlsym( lib, "shmem_getmem" );
  if (shmem_getmem_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_getmem in DSOs!!\n");

  /* Obtain @ for shmem_int_get */
  shmem_int_get_real = 
    (void (*)(int *, const int *, size_t, int))
    dlsym( lib, "shmem_int_get" );
  if (shmem_int_get_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_int_get in DSOs!!\n");

  /* Obtain @ for shmem_long_get */
  shmem_long_get_real = 
    (void (*)(long *, const long *, size_t, int))
    dlsym( lib, "shmem_long_get" );
  if (shmem_long_get_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_long_get in DSOs!!\n");

  /* Obtain @ for shmem_longdouble_get */
  shmem_longdouble_get_real = 
    (void (*)(long double *, const long double *, size_t, int))
    dlsym( lib, "shmem_longdouble_get" );
  if (shmem_longdouble_get_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_longdouble_get in DSOs!!\n");

  /* Obtain @ for shmem_longlong_get */
  shmem_longlong_get_real = 
    (void (*)(long long *, const long long *, size_t, int))
    dlsym( lib, "shmem_longlong_get" );
  if (shmem_longlong_get_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_longlong_get in DSOs!!\n");

  /* Obtain @ for shmem_short_get */
  shmem_short_get_real = 
    (void (*)(short *, const short *, size_t, int))
    dlsym( lib, "shmem_short_get" );
  if (shmem_short_get_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_short_get in DSOs!!\n");

  /* Obtain @ for shmem_char_g */
  shmem_char_g_real = 
    (char (*)(char *, int))
    dlsym( lib, "shmem_char_g" );
  if (shmem_char_g_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_char_g in DSOs!!\n");

  /* Obtain @ for shmem_short_g */
  shmem_short_g_real = 
    (short (*)(short *, int))
    dlsym( lib, "shmem_short_g" );
  if (shmem_short_g_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_short_g in DSOs!!\n");

  /* Obtain @ for shmem_int_g */
  shmem_int_g_real = 
    (int (*)(int *, int))
    dlsym( lib, "shmem_int_g" );
  if (shmem_int_g_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_int_g in DSOs!!\n");

  /* Obtain @ for shmem_long_g */
  shmem_long_g_real = 
    (long (*)(long *, int))
    dlsym( lib, "shmem_long_g" );
  if (shmem_long_g_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_long_g in DSOs!!\n");

  /* Obtain @ for shmem_longlong_g */
  shmem_longlong_g_real = 
    (long long (*)(long long *, int))
    dlsym( lib, "shmem_longlong_g" );
  if (shmem_longlong_g_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_longlong_g in DSOs!!\n");

  /* Obtain @ for shmem_float_g */
  shmem_float_g_real = 
    (float (*)(float *, int))
    dlsym( lib, "shmem_float_g" );
  if (shmem_float_g_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_float_g in DSOs!!\n");

  /* Obtain @ for shmem_double_g */
  shmem_double_g_real = 
    (double (*)(double *, int))
    dlsym( lib, "shmem_double_g" );
  if (shmem_double_g_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_double_g in DSOs!!\n");

  /* Obtain @ for shmem_longdouble_g */
  shmem_longdouble_g_real = 
    (long double (*)(long double *, int))
    dlsym( lib, "shmem_longdouble_g" );
  if (shmem_longdouble_g_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_longdouble_g in DSOs!!\n");

  /* Obtain @ for shmem_double_iget */
  shmem_double_iget_real = 
    (void (*)(double *, const double *, ptrdiff_t, ptrdiff_t, size_t, int))
    dlsym( lib, "shmem_double_iget" );
  if (shmem_double_iget_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_double_iget in DSOs!!\n");

  /* Obtain @ for shmem_float_iget */
  shmem_float_iget_real = 
    (void (*)(float *, const float *, ptrdiff_t, ptrdiff_t, size_t, int))
    dlsym( lib, "shmem_float_iget" );
  if (shmem_float_iget_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_float_iget in DSOs!!\n");

  /* Obtain @ for shmem_iget32 */
  shmem_iget32_real = 
    (void (*)(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int))
    dlsym( lib, "shmem_iget32" );
  if (shmem_iget32_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_iget32 in DSOs!!\n");

  /* Obtain @ for shmem_iget64 */
  shmem_iget64_real = 
    (void (*)(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int))
    dlsym( lib, "shmem_iget64" );
  if (shmem_iget64_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_iget64 in DSOs!!\n");

  /* Obtain @ for shmem_iget128 */
  shmem_iget128_real = 
    (void (*)(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int))
    dlsym( lib, "shmem_iget128" );
  if (shmem_iget128_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_iget128 in DSOs!!\n");

  /* Obtain @ for shmem_int_iget */
  shmem_int_iget_real = 
    (void (*)(int *, const int *, ptrdiff_t, ptrdiff_t, size_t, int))
    dlsym( lib, "shmem_int_iget" );
  if (shmem_int_iget_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_int_iget in DSOs!!\n");

  /* Obtain @ for shmem_long_iget */
  shmem_long_iget_real = 
    (void (*)(long *, const long *, ptrdiff_t, ptrdiff_t, size_t, int))
    dlsym( lib, "shmem_long_iget" );
  if (shmem_long_iget_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_long_iget in DSOs!!\n");

  /* Obtain @ for shmem_longdouble_iget */
  shmem_longdouble_iget_real = 
    (void (*)(long double *, const long double *, ptrdiff_t, ptrdiff_t, size_t, int))
    dlsym( lib, "shmem_longdouble_iget" );
  if (shmem_longdouble_iget_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_longdouble_iget in DSOs!!\n");

  /* Obtain @ for shmem_longlong_iget */
  shmem_longlong_iget_real = 
    (void (*)(long long *, const long long *, ptrdiff_t, ptrdiff_t, size_t, int))
    dlsym( lib, "shmem_longlong_iget" );
  if (shmem_longlong_iget_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_longlong_iget in DSOs!!\n");

  /* Obtain @ for shmem_short_iget */
  shmem_short_iget_real = 
    (void (*)(short *, const short *, ptrdiff_t, ptrdiff_t, size_t, int))
    dlsym( lib, "shmem_short_iget" );
  if (shmem_short_iget_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_short_iget in DSOs!!\n");

  /* Obtain @ for shmem_int_add */
  shmem_int_add_real = 
    (void (*)(int *, int, int))
    dlsym( lib, "shmem_int_add" );
  if (shmem_int_add_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_int_add in DSOs!!\n");

  /* Obtain @ for shmem_long_add */
  shmem_long_add_real = 
    (void (*)(long *, long, int))
    dlsym( lib, "shmem_long_add" );
  if (shmem_long_add_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_long_add in DSOs!!\n");

  /* Obtain @ for shmem_longlong_add */
  shmem_longlong_add_real = 
    (void (*)(long long *, long long, int))
    dlsym( lib, "shmem_longlong_add" );
  if (shmem_longlong_add_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_longlong_add in DSOs!!\n");

  /* Obtain @ for shmem_int_cswap */
  shmem_int_cswap_real = 
    (int (*)(int *, int, int, int))
    dlsym( lib, "shmem_int_cswap" );
  if (shmem_int_cswap_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_int_cswap in DSOs!!\n");

  /* Obtain @ for shmem_long_cswap */
  shmem_long_cswap_real = 
    (long (*)(long *, long, long, int))
    dlsym( lib, "shmem_long_cswap" );
  if (shmem_long_cswap_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_long_cswap in DSOs!!\n");

  /* Obtain @ for shmem_longlong_cswap */
  shmem_longlong_cswap_real = 
    (long long (*)(long long *, long long, long long, int))
    dlsym( lib, "shmem_longlong_cswap" );
  if (shmem_longlong_cswap_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_longlong_cswap in DSOs!!\n");

  /* Obtain @ for shmem_double_swap */
  shmem_double_swap_real = 
    (double (*)(double *, double, int))
    dlsym( lib, "shmem_double_swap" );
  if (shmem_double_swap_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_double_swap in DSOs!!\n");

  /* Obtain @ for shmem_float_swap */
  shmem_float_swap_real = 
    (float (*)(float *, float, int))
    dlsym( lib, "shmem_float_swap" );
  if (shmem_float_swap_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_float_swap in DSOs!!\n");

  /* Obtain @ for shmem_int_swap */
  shmem_int_swap_real = 
    (int (*)(int *, int, int))
    dlsym( lib, "shmem_int_swap" );
  if (shmem_int_swap_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_int_swap in DSOs!!\n");

  /* Obtain @ for shmem_long_swap */
  shmem_long_swap_real = 
    (long (*)(long *, long, int))
    dlsym( lib, "shmem_long_swap" );
  if (shmem_long_swap_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_long_swap in DSOs!!\n");

  /* Obtain @ for shmem_longlong_swap */
  shmem_longlong_swap_real = 
    (long long (*)(long long *, long long, int))
    dlsym( lib, "shmem_longlong_swap" );
  if (shmem_longlong_swap_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_longlong_swap in DSOs!!\n");

  /* Obtain @ for shmem_swap */
  shmem_swap_real = 
    (long (*)(long *, long, int))
    dlsym( lib, "shmem_swap" );
  if (shmem_swap_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_swap in DSOs!!\n");

  /* Obtain @ for shmem_int_finc */
  shmem_int_finc_real = 
    (int (*)(int *, int))
    dlsym( lib, "shmem_int_finc" );
  if (shmem_int_finc_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_int_finc in DSOs!!\n");

  /* Obtain @ for shmem_long_finc */
  shmem_long_finc_real = 
    (long (*)(long *, int))
    dlsym( lib, "shmem_long_finc" );
  if (shmem_long_finc_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_long_finc in DSOs!!\n");

  /* Obtain @ for shmem_longlong_finc */
  shmem_longlong_finc_real = 
    (long long (*)(long long *, int))
    dlsym( lib, "shmem_longlong_finc" );
  if (shmem_longlong_finc_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_longlong_finc in DSOs!!\n");

  /* Obtain @ for shmem_int_inc */
  shmem_int_inc_real = 
    (void (*)(int *, int))
    dlsym( lib, "shmem_int_inc" );
  if (shmem_int_inc_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_int_inc in DSOs!!\n");

  /* Obtain @ for shmem_long_inc */
  shmem_long_inc_real = 
    (void (*)(long *, int))
    dlsym( lib, "shmem_long_inc" );
  if (shmem_long_inc_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_long_inc in DSOs!!\n");

  /* Obtain @ for shmem_longlong_inc */
  shmem_longlong_inc_real = 
    (void (*)(long long *, int))
    dlsym( lib, "shmem_longlong_inc" );
  if (shmem_longlong_inc_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_longlong_inc in DSOs!!\n");

  /* Obtain @ for shmem_int_fadd */
  shmem_int_fadd_real = 
    (int (*)(int *, int, int))
    dlsym( lib, "shmem_int_fadd" );
  if (shmem_int_fadd_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_int_fadd in DSOs!!\n");

  /* Obtain @ for shmem_long_fadd */
  shmem_long_fadd_real = 
    (long (*)(long *, long, int))
    dlsym( lib, "shmem_long_fadd" );
  if (shmem_long_fadd_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_long_fadd in DSOs!!\n");

  /* Obtain @ for shmem_longlong_fadd */
  shmem_longlong_fadd_real = 
    (long long (*)(long long *, long long, int))
    dlsym( lib, "shmem_longlong_fadd" );
  if (shmem_longlong_fadd_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_longlong_fadd in DSOs!!\n");

  /* Obtain @ for shmem_barrier_all */
  shmem_barrier_all_real = 
    (void (*)(void))
    dlsym( lib, "shmem_barrier_all" );
  if (shmem_barrier_all_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_barrier_all in DSOs!!\n");

  /* Obtain @ for shmem_barrier */
  shmem_barrier_real = 
    (void (*)(int, int, int, long *))
    dlsym( lib, "shmem_barrier" );
  if (shmem_barrier_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_barrier in DSOs!!\n");

  /* Obtain @ for shmem_broadcast32 */
  shmem_broadcast32_real = 
    (void (*)(void *, const void *, size_t, int, int, int, int, long *))
    dlsym( lib, "shmem_broadcast32" );
  if (shmem_broadcast32_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_broadcast32 in DSOs!!\n");

  /* Obtain @ for shmem_broadcast64 */
  shmem_broadcast64_real = 
    (void (*)(void *, const void *, size_t, int, int, int, int, long *))
    dlsym( lib, "shmem_broadcast64" );
  if (shmem_broadcast64_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_broadcast64 in DSOs!!\n");

  /* Obtain @ for shmem_collect32 */
  shmem_collect32_real = 
    (void (*)(void *, const void *, size_t, int, int, int, long *))
    dlsym( lib, "shmem_collect32" );
  if (shmem_collect32_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_collect32 in DSOs!!\n");

  /* Obtain @ for shmem_collect64 */
  shmem_collect64_real = 
    (void (*)(void *, const void *, size_t, int, int, int, long *))
    dlsym( lib, "shmem_collect64" );
  if (shmem_collect64_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_collect64 in DSOs!!\n");

  /* Obtain @ for shmem_fcollect32 */
  shmem_fcollect32_real = 
    (void (*)(void *, const void *, size_t, int, int, int, long *))
    dlsym( lib, "shmem_fcollect32" );
  if (shmem_fcollect32_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_fcollect32 in DSOs!!\n");

  /* Obtain @ for shmem_fcollect64 */
  shmem_fcollect64_real = 
    (void (*)(void *, const void *, size_t, int, int, int, long *))
    dlsym( lib, "shmem_fcollect64" );
  if (shmem_fcollect64_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_fcollect64 in DSOs!!\n");

  /* Obtain @ for shmem_int_and_to_all */
  shmem_int_and_to_all_real = 
    (void (*)(int *, int *, int, int, int, int, int *, long *))
    dlsym( lib, "shmem_int_and_to_all" );
  if (shmem_int_and_to_all_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_int_and_to_all in DSOs!!\n");

  /* Obtain @ for shmem_long_and_to_all */
  shmem_long_and_to_all_real = 
    (void (*)(long *, long *, int, int, int, int, long *, long *))
    dlsym( lib, "shmem_long_and_to_all" );
  if (shmem_long_and_to_all_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_long_and_to_all in DSOs!!\n");

  /* Obtain @ for shmem_longlong_and_to_all */
  shmem_longlong_and_to_all_real = 
    (void (*)(long long *, long long *, int, int, int, int, long long *, long *))
    dlsym( lib, "shmem_longlong_and_to_all" );
  if (shmem_longlong_and_to_all_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_longlong_and_to_all in DSOs!!\n");

  /* Obtain @ for shmem_short_and_to_all */
  shmem_short_and_to_all_real = 
    (void (*)(short *, short *, int, int, int, int, short *, long *))
    dlsym( lib, "shmem_short_and_to_all" );
  if (shmem_short_and_to_all_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_short_and_to_all in DSOs!!\n");

  /* Obtain @ for shmem_double_max_to_all */
  shmem_double_max_to_all_real = 
    (void (*)(double *, double *, int, int, int, int, double *, long *))
    dlsym( lib, "shmem_double_max_to_all" );
  if (shmem_double_max_to_all_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_double_max_to_all in DSOs!!\n");

  /* Obtain @ for shmem_float_max_to_all */
  shmem_float_max_to_all_real = 
    (void (*)(float *, float *, int, int, int, int, float *, long *))
    dlsym( lib, "shmem_float_max_to_all" );
  if (shmem_float_max_to_all_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_float_max_to_all in DSOs!!\n");

  /* Obtain @ for shmem_int_max_to_all */
  shmem_int_max_to_all_real = 
    (void (*)(int *, int *, int, int, int, int, int *, long *))
    dlsym( lib, "shmem_int_max_to_all" );
  if (shmem_int_max_to_all_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_int_max_to_all in DSOs!!\n");

  /* Obtain @ for shmem_long_max_to_all */
  shmem_long_max_to_all_real = 
    (void (*)(long *, long *, int, int, int, int, long *, long *))
    dlsym( lib, "shmem_long_max_to_all" );
  if (shmem_long_max_to_all_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_long_max_to_all in DSOs!!\n");

  /* Obtain @ for shmem_longdouble_max_to_all */
  shmem_longdouble_max_to_all_real = 
    (void (*)(long double *, long double *, int, int, int, int, long double *, long *))
    dlsym( lib, "shmem_longdouble_max_to_all" );
  if (shmem_longdouble_max_to_all_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_longdouble_max_to_all in DSOs!!\n");

  /* Obtain @ for shmem_longlong_max_to_all */
  shmem_longlong_max_to_all_real = 
    (void (*)(long long *, long long *, int, int, int, int, long long *, long *))
    dlsym( lib, "shmem_longlong_max_to_all" );
  if (shmem_longlong_max_to_all_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_longlong_max_to_all in DSOs!!\n");

  /* Obtain @ for shmem_short_max_to_all */
  shmem_short_max_to_all_real = 
    (void (*)(short *, short *, int, int, int, int, short *, long *))
    dlsym( lib, "shmem_short_max_to_all" );
  if (shmem_short_max_to_all_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_short_max_to_all in DSOs!!\n");

  /* Obtain @ for shmem_double_min_to_all */
  shmem_double_min_to_all_real = 
    (void (*)(double *, double *, int, int, int, int, double *, long *))
    dlsym( lib, "shmem_double_min_to_all" );
  if (shmem_double_min_to_all_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_double_min_to_all in DSOs!!\n");

  /* Obtain @ for shmem_int_wait */
  shmem_int_wait_real = 
    (void (*)(int *, int))
    dlsym( lib, "shmem_int_wait" );
  if (shmem_int_wait_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_int_wait in DSOs!!\n");

  /* Obtain @ for shmem_int_wait_until */
  shmem_int_wait_until_real = 
    (void (*)(int *, int, int))
    dlsym( lib, "shmem_int_wait_until" );
  if (shmem_int_wait_until_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_int_wait_until in DSOs!!\n");

  /* Obtain @ for shmem_long_wait */
  shmem_long_wait_real = 
    (void (*)(long *, long))
    dlsym( lib, "shmem_long_wait" );
  if (shmem_long_wait_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_long_wait in DSOs!!\n");

  /* Obtain @ for shmem_long_wait_until */
  shmem_long_wait_until_real = 
    (void (*)(long *, int, long))
    dlsym( lib, "shmem_long_wait_until" );
  if (shmem_long_wait_until_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_long_wait_until in DSOs!!\n");

  /* Obtain @ for shmem_longlong_wait */
  shmem_longlong_wait_real = 
    (void (*)(long long *, long long))
    dlsym( lib, "shmem_longlong_wait" );
  if (shmem_longlong_wait_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_longlong_wait in DSOs!!\n");

  /* Obtain @ for shmem_longlong_wait_until */
  shmem_longlong_wait_until_real = 
    (void (*)(long long *, int, long long))
    dlsym( lib, "shmem_longlong_wait_until" );
  if (shmem_longlong_wait_until_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_longlong_wait_until in DSOs!!\n");

  /* Obtain @ for shmem_short_wait */
  shmem_short_wait_real = 
    (void (*)(short *, short))
    dlsym( lib, "shmem_short_wait" );
  if (shmem_short_wait_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_short_wait in DSOs!!\n");

  /* Obtain @ for shmem_short_wait_until */
  shmem_short_wait_until_real = 
    (void (*)(short *, int, short))
    dlsym( lib, "shmem_short_wait_until" );
  if (shmem_short_wait_until_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_short_wait_until in DSOs!!\n");

  /* Obtain @ for shmem_wait */
  shmem_wait_real = 
    (void (*)(long *, long))
    dlsym( lib, "shmem_wait" );
  if (shmem_wait_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_wait in DSOs!!\n");

  /* Obtain @ for shmem_wait_until */
  shmem_wait_until_real = 
    (void (*)(long *, int, long))
    dlsym( lib, "shmem_wait_until" );
  if (shmem_wait_until_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_wait_until in DSOs!!\n");

  /* Obtain @ for shmem_fence */
  shmem_fence_real = 
    (void (*)(void))
    dlsym( lib, "shmem_fence" );
  if (shmem_fence_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_fence in DSOs!!\n");

  /* Obtain @ for shmem_quiet */
  shmem_quiet_real = 
    (void (*)(void))
    dlsym( lib, "shmem_quiet" );
  if (shmem_quiet_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_quiet in DSOs!!\n");

  /* Obtain @ for shmem_clear_lock */
  shmem_clear_lock_real = 
    (void (*)(long *))
    dlsym( lib, "shmem_clear_lock" );
  if (shmem_clear_lock_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_clear_lock in DSOs!!\n");

  /* Obtain @ for shmem_set_lock */
  shmem_set_lock_real = 
    (void (*)(long *))
    dlsym( lib, "shmem_set_lock" );
  if (shmem_set_lock_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_set_lock in DSOs!!\n");

  /* Obtain @ for shmem_test_lock */
  shmem_test_lock_real = 
    (int (*)(long *))
    dlsym( lib, "shmem_test_lock" );
  if (shmem_test_lock_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_test_lock in DSOs!!\n");

  /* Obtain @ for shmem_clear_cache_inv */
  shmem_clear_cache_inv_real = 
    (void (*)(void))
    dlsym( lib, "shmem_clear_cache_inv" );
  if (shmem_clear_cache_inv_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_clear_cache_inv in DSOs!!\n");

  /* Obtain @ for shmem_set_cache_inv */
  shmem_set_cache_inv_real = 
    (void (*)(void))
    dlsym( lib, "shmem_set_cache_inv" );
  if (shmem_set_cache_inv_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_set_cache_inv in DSOs!!\n");

  /* Obtain @ for shmem_clear_cache_line_inv */
  shmem_clear_cache_line_inv_real = 
    (void (*)(void *))
    dlsym( lib, "shmem_clear_cache_line_inv" );
  if (shmem_clear_cache_line_inv_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_clear_cache_line_inv in DSOs!!\n");

  /* Obtain @ for shmem_set_cache_line_inv */
  shmem_set_cache_line_inv_real = 
    (void (*)(void *))
    dlsym( lib, "shmem_set_cache_line_inv" );
  if (shmem_set_cache_line_inv_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_set_cache_line_inv in DSOs!!\n");

  /* Obtain @ for shmem_udcflush */
  shmem_udcflush_real = 
    (void (*)(void))
    dlsym( lib, "shmem_udcflush" );
  if (shmem_udcflush_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_udcflush in DSOs!!\n");

  /* Obtain @ for shmem_udcflush_line */
  shmem_udcflush_line_real = 
    (void (*)(void *))
    dlsym( lib, "shmem_udcflush_line" );
  if (shmem_udcflush_line_real == NULL && rank == 0)
    fprintf(stderr, PACKAGE_NAME": Unable to find shmem_udcflush_line in DSOs!!\n");

#endif /* PIC */
}

static void Initialize_Extrae_Stuff()
{
  iotimer_t shmem_init_start_time, shmem_init_end_time;

  Extrae_set_ApplicationIsSHMEM (TRUE);
  Extrae_Allocate_Task_Bitmap( Extrae_OPENSHMEM_NumTasks() );
  Extrae_set_taskid_function( Extrae_OPENSHMEM_TaskID );
  Extrae_set_numtasks_function( Extrae_OPENSHMEM_NumTasks );
  Extrae_set_barrier_tasks_function ( Extrae_OPENSHMEM_Barrier );
  Extrae_set_finalize_task_function ( Extrae_OPENSHMEM_Finalize );

  if (Extrae_is_initialized_Wrapper() == EXTRAE_NOT_INITIALIZED)
  {
    int res;
    char *config_file = getenv ("EXTRAE_CONFIG_FILE");
  
    if (config_file == NULL)
    {
      config_file = getenv ("MPTRACE_CONFIG_FILE");
    }
    
    Extrae_set_initial_TASKID (TASKID);
    Extrae_set_is_initialized (EXTRAE_INITIALIZED_SHMEM_INIT);
    
    if (config_file != NULL)
    {
      /* Obtain a localized copy *except for the master process* */
      config_file = OPENSHMEM_Distribute_XML_File (TASKID, Extrae_get_num_tasks(), config_file);
    }

    /* Initialize the backend */
    res = Backend_preInitialize (TASKID, Extrae_get_num_tasks(), config_file, FALSE);
    if (!res) return;
    
    /* Remove the local copy only if we're not the master */
    if (TASKID != 0)
      unlink (config_file);
    free (config_file);
  }
  else  
  {
    Backend_updateTaskID ();
  }

  OPENSHMEM_Gather_Nodes_Info();

  if (Extrae_is_initialized_Wrapper() == EXTRAE_INITIALIZED_EXTRAE_INIT)
    OPENSHMEM_remove_file_list (TRUE);

  OPENSHMEM_Generate_Task_File_list (TasksNodes);

  shmem_init_start_time = TIME;

  Extrae_barrier_tasks();
  Extrae_barrier_tasks();
  Extrae_barrier_tasks(); 

  initTracingTime = shmem_init_end_time = TIME;

  if (!Backend_postInitialize (TASKID, Extrae_get_num_tasks(), 0, shmem_init_start_time, shmem_init_end_time, TasksNodes))
     return;
}

/****************************************\
 ***           INJECTED CODE          ***
\****************************************/

void start_pes (int npes)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %d\n", TASKID, npes);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: start_pes_real at %p\n", TASKID, start_pes_real);
#endif

  if (start_pes_real != NULL)
  {
    Backend_Enter_Instrumentation(2);
    start_pes_real(npes);
    atexit (shmem_finalize);
    Initialize_Extrae_Stuff();
    PROBE_start_pes_ENTRY(npes);
    PROBE_start_pes_EXIT();
    Backend_Leave_Instrumentation();
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error start_pes was not hooked!\n");
    exit(-1);
  }
}

int shmem_my_pe (void)
{
  int res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_my_pe_real at %p\n", TASKID, shmem_my_pe_real);
#endif

  if (shmem_my_pe_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_my_pe_ENTRY();
    res = shmem_my_pe_real();
    PROBE_shmem_my_pe_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_my_pe_real != NULL)
  {
    res = shmem_my_pe_real();
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_my_pe was not hooked!\n");
    exit(-1);
  }
  return res;
}

int _my_pe (void)
{
  int res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: _my_pe_real at %p\n", TASKID, _my_pe_real);
#endif

  if (_my_pe_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE__my_pe_ENTRY();
    res = _my_pe_real();
    PROBE__my_pe_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (_my_pe_real != NULL)
  {
    res = _my_pe_real();
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error _my_pe was not hooked!\n");
    exit(-1);
  }
  return res;
}

int shmem_n_pes (void)
{
  int res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_n_pes_real at %p\n", TASKID, shmem_n_pes_real);
#endif

  if (shmem_n_pes_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_n_pes_ENTRY();
    res = shmem_n_pes_real();
    PROBE_shmem_n_pes_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_n_pes_real != NULL)
  {
    res = shmem_n_pes_real();
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_n_pes was not hooked!\n");
    exit(-1);
  }
  return res;
}

int _num_pes (void)
{
  int res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: _num_pes_real at %p\n", TASKID, _num_pes_real);
#endif

  if (_num_pes_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE__num_pes_ENTRY();
    res = _num_pes_real();
    PROBE__num_pes_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (_num_pes_real != NULL)
  {
    res = _num_pes_real();
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error _num_pes was not hooked!\n");
    exit(-1);
  }
  return res;
}

int shmem_pe_accessible (int pe)
{
  int res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %d\n", TASKID, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_pe_accessible_real at %p\n", TASKID, shmem_pe_accessible_real);
#endif

  if (shmem_pe_accessible_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_pe_accessible_ENTRY(pe);
    res = shmem_pe_accessible_real(pe);
    PROBE_shmem_pe_accessible_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_pe_accessible_real != NULL)
  {
    res = shmem_pe_accessible_real(pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_pe_accessible was not hooked!\n");
    exit(-1);
  }
  return res;
}

int shmem_addr_accessible (void *addr, int pe)
{
  int res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %d\n", TASKID, addr, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_addr_accessible_real at %p\n", TASKID, shmem_addr_accessible_real);
#endif

  if (shmem_addr_accessible_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_addr_accessible_ENTRY(addr, pe);
    res = shmem_addr_accessible_real(addr, pe);
    PROBE_shmem_addr_accessible_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_addr_accessible_real != NULL)
  {
    res = shmem_addr_accessible_real(addr, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_addr_accessible was not hooked!\n");
    exit(-1);
  }
  return res;
}

void * shmem_ptr (void *target, int pe)
{
  void * res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %d\n", TASKID, target, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_ptr_real at %p\n", TASKID, shmem_ptr_real);
#endif

  if (shmem_ptr_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_ptr_ENTRY(target, pe);
    res = shmem_ptr_real(target, pe);
    PROBE_shmem_ptr_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_ptr_real != NULL)
  {
    res = shmem_ptr_real(target, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_ptr was not hooked!\n");
    exit(-1);
  }
  return res;
}

void * shmalloc (size_t size)
{
  void * res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %d\n", TASKID, size);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmalloc_real at %p\n", TASKID, shmalloc_real);
#endif

  if (shmalloc_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmalloc_ENTRY(size);
    res = shmalloc_real(size);
    PROBE_shmalloc_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmalloc_real != NULL)
  {
    res = shmalloc_real(size);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmalloc was not hooked!\n");
    exit(-1);
  }
  return res;
}

void shfree (void *ptr)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p\n", TASKID, ptr);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shfree_real at %p\n", TASKID, shfree_real);
#endif

  if (shfree_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shfree_ENTRY(ptr);
    shfree_real(ptr);
    PROBE_shfree_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shfree_real != NULL)
  {
    shfree_real(ptr);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shfree was not hooked!\n");
    exit(-1);
  }
}

void * shrealloc (void *ptr, size_t size)
{
  void * res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %d\n", TASKID, ptr, size);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shrealloc_real at %p\n", TASKID, shrealloc_real);
#endif

  if (shrealloc_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shrealloc_ENTRY(ptr, size);
    res = shrealloc_real(ptr, size);
    PROBE_shrealloc_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shrealloc_real != NULL)
  {
    res = shrealloc_real(ptr, size);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shrealloc was not hooked!\n");
    exit(-1);
  }
  return res;
}

void * shmemalign (size_t alignment, size_t size)
{
  void * res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %d %d\n", TASKID, alignment, size);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmemalign_real at %p\n", TASKID, shmemalign_real);
#endif

  if (shmemalign_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmemalign_ENTRY(alignment, size);
    res = shmemalign_real(alignment, size);
    PROBE_shmemalign_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmemalign_real != NULL)
  {
    res = shmemalign_real(alignment, size);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmemalign was not hooked!\n");
    exit(-1);
  }
  return res;
}

void shmem_double_put (double *target, const double *source, size_t len, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d\n", TASKID, target, source, len, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_double_put_real at %p\n", TASKID, shmem_double_put_real);
#endif

  if (shmem_double_put_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_double_put_ENTRY(target, source, len, pe);
    shmem_double_put_real(target, source, len, pe);
    PROBE_shmem_double_put_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_double_put_real != NULL)
  {
    shmem_double_put_real(target, source, len, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_double_put was not hooked!\n");
    exit(-1);
  }
}

void shmem_float_put (float *target, const float *source, size_t len, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d\n", TASKID, target, source, len, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_float_put_real at %p\n", TASKID, shmem_float_put_real);
#endif

  if (shmem_float_put_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_float_put_ENTRY(target, source, len, pe);
    shmem_float_put_real(target, source, len, pe);
    PROBE_shmem_float_put_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_float_put_real != NULL)
  {
    shmem_float_put_real(target, source, len, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_float_put was not hooked!\n");
    exit(-1);
  }
}

void shmem_int_put (int *target, const int *source, size_t len, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d\n", TASKID, target, source, len, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_int_put_real at %p\n", TASKID, shmem_int_put_real);
#endif

  if (shmem_int_put_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_int_put_ENTRY(target, source, len, pe);
    shmem_int_put_real(target, source, len, pe);
    PROBE_shmem_int_put_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_int_put_real != NULL)
  {
    shmem_int_put_real(target, source, len, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_int_put was not hooked!\n");
    exit(-1);
  }
}

void shmem_long_put (long *target, const long *source, size_t len, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d\n", TASKID, target, source, len, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_long_put_real at %p\n", TASKID, shmem_long_put_real);
#endif

  if (shmem_long_put_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_long_put_ENTRY(target, source, len, pe);
    shmem_long_put_real(target, source, len, pe);
    PROBE_shmem_long_put_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_long_put_real != NULL)
  {
    shmem_long_put_real(target, source, len, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_long_put was not hooked!\n");
    exit(-1);
  }
}

void shmem_longdouble_put (long double *target, const long double *source, size_t len,int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d\n", TASKID, target, source, len, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_longdouble_put_real at %p\n", TASKID, shmem_longdouble_put_real);
#endif

  if (shmem_longdouble_put_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_longdouble_put_ENTRY(target, source, len, pe);
    shmem_longdouble_put_real(target, source, len, pe);
    PROBE_shmem_longdouble_put_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_longdouble_put_real != NULL)
  {
    shmem_longdouble_put_real(target, source, len, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_longdouble_put was not hooked!\n");
    exit(-1);
  }
}

void shmem_longlong_put (long long *target, const long long *source, size_t len, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d\n", TASKID, target, source, len, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_longlong_put_real at %p\n", TASKID, shmem_longlong_put_real);
#endif

  if (shmem_longlong_put_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_longlong_put_ENTRY(target, source, len, pe);
    shmem_longlong_put_real(target, source, len, pe);
    PROBE_shmem_longlong_put_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_longlong_put_real != NULL)
  {
    shmem_longlong_put_real(target, source, len, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_longlong_put was not hooked!\n");
    exit(-1);
  }
}

void shmem_put32 (void *target, const void *source, size_t len, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d\n", TASKID, target, source, len, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_put32_real at %p\n", TASKID, shmem_put32_real);
#endif

  if (shmem_put32_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_put32_ENTRY(target, source, len, pe);
    shmem_put32_real(target, source, len, pe);
    PROBE_shmem_put32_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_put32_real != NULL)
  {
    shmem_put32_real(target, source, len, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_put32 was not hooked!\n");
    exit(-1);
  }
}

void shmem_put64 (void *target, const void *source, size_t len, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d\n", TASKID, target, source, len, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_put64_real at %p\n", TASKID, shmem_put64_real);
#endif

  if (shmem_put64_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_put64_ENTRY(target, source, len, pe);
    shmem_put64_real(target, source, len, pe);
    PROBE_shmem_put64_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_put64_real != NULL)
  {
    shmem_put64_real(target, source, len, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_put64 was not hooked!\n");
    exit(-1);
  }
}

void shmem_put128 (void *target, const void *source, size_t len, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d\n", TASKID, target, source, len, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_put128_real at %p\n", TASKID, shmem_put128_real);
#endif

  if (shmem_put128_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_put128_ENTRY(target, source, len, pe);
    shmem_put128_real(target, source, len, pe);
    PROBE_shmem_put128_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_put128_real != NULL)
  {
    shmem_put128_real(target, source, len, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_put128 was not hooked!\n");
    exit(-1);
  }
}

void shmem_putmem (void *target, const void *source, size_t len, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d\n", TASKID, target, source, len, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_putmem_real at %p\n", TASKID, shmem_putmem_real);
#endif

  if (shmem_putmem_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_putmem_ENTRY(target, source, len, pe);
    shmem_putmem_real(target, source, len, pe);
    PROBE_shmem_putmem_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_putmem_real != NULL)
  {
    shmem_putmem_real(target, source, len, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_putmem was not hooked!\n");
    exit(-1);
  }
}

void shmem_short_put (short*target, const short*source, size_t len, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d\n", TASKID, target, source, len, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_short_put_real at %p\n", TASKID, shmem_short_put_real);
#endif

  if (shmem_short_put_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_short_put_ENTRY(target, source, len, pe);
    shmem_short_put_real(target, source, len, pe);
    PROBE_shmem_short_put_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_short_put_real != NULL)
  {
    shmem_short_put_real(target, source, len, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_short_put was not hooked!\n");
    exit(-1);
  }
}

void shmem_char_p (char *addr, char value, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %c %d\n", TASKID, addr, value, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_char_p_real at %p\n", TASKID, shmem_char_p_real);
#endif

  if (shmem_char_p_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_char_p_ENTRY(addr, value, pe);
    shmem_char_p_real(addr, value, pe);
    PROBE_shmem_char_p_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_char_p_real != NULL)
  {
    shmem_char_p_real(addr, value, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_char_p was not hooked!\n");
    exit(-1);
  }
}

void shmem_short_p (short *addr, short value, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %h %d\n", TASKID, addr, value, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_short_p_real at %p\n", TASKID, shmem_short_p_real);
#endif

  if (shmem_short_p_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_short_p_ENTRY(addr, value, pe);
    shmem_short_p_real(addr, value, pe);
    PROBE_shmem_short_p_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_short_p_real != NULL)
  {
    shmem_short_p_real(addr, value, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_short_p was not hooked!\n");
    exit(-1);
  }
}

void shmem_int_p (int *addr, int value, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %d %d\n", TASKID, addr, value, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_int_p_real at %p\n", TASKID, shmem_int_p_real);
#endif

  if (shmem_int_p_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_int_p_ENTRY(addr, value, pe);
    shmem_int_p_real(addr, value, pe);
    PROBE_shmem_int_p_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_int_p_real != NULL)
  {
    shmem_int_p_real(addr, value, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_int_p was not hooked!\n");
    exit(-1);
  }
}

void shmem_long_p (long *addr, long value, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %ld %d\n", TASKID, addr, value, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_long_p_real at %p\n", TASKID, shmem_long_p_real);
#endif

  if (shmem_long_p_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_long_p_ENTRY(addr, value, pe);
    shmem_long_p_real(addr, value, pe);
    PROBE_shmem_long_p_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_long_p_real != NULL)
  {
    shmem_long_p_real(addr, value, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_long_p was not hooked!\n");
    exit(-1);
  }
}

void shmem_longlong_p (long long *addr, long long value, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %lld %d\n", TASKID, addr, value, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_longlong_p_real at %p\n", TASKID, shmem_longlong_p_real);
#endif

  if (shmem_longlong_p_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_longlong_p_ENTRY(addr, value, pe);
    shmem_longlong_p_real(addr, value, pe);
    PROBE_shmem_longlong_p_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_longlong_p_real != NULL)
  {
    shmem_longlong_p_real(addr, value, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_longlong_p was not hooked!\n");
    exit(-1);
  }
}

void shmem_float_p (float *addr, float value, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %f %d\n", TASKID, addr, value, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_float_p_real at %p\n", TASKID, shmem_float_p_real);
#endif

  if (shmem_float_p_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_float_p_ENTRY(addr, value, pe);
    shmem_float_p_real(addr, value, pe);
    PROBE_shmem_float_p_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_float_p_real != NULL)
  {
    shmem_float_p_real(addr, value, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_float_p was not hooked!\n");
    exit(-1);
  }
}

void shmem_double_p (double *addr, double value, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %lf %d\n", TASKID, addr, value, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_double_p_real at %p\n", TASKID, shmem_double_p_real);
#endif

  if (shmem_double_p_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_double_p_ENTRY(addr, value, pe);
    shmem_double_p_real(addr, value, pe);
    PROBE_shmem_double_p_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_double_p_real != NULL)
  {
    shmem_double_p_real(addr, value, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_double_p was not hooked!\n");
    exit(-1);
  }
}

void shmem_longdouble_p (long double *addr, long double value, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %lf %d\n", TASKID, addr, value, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_longdouble_p_real at %p\n", TASKID, shmem_longdouble_p_real);
#endif

  if (shmem_longdouble_p_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_longdouble_p_ENTRY(addr, value, pe);
    shmem_longdouble_p_real(addr, value, pe);
    PROBE_shmem_longdouble_p_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_longdouble_p_real != NULL)
  {
    shmem_longdouble_p_real(addr, value, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_longdouble_p was not hooked!\n");
    exit(-1);
  }
}

void shmem_double_iput (double *target, const double *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %p %p %d %d\n", TASKID, target, source, tst, sst, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_double_iput_real at %p\n", TASKID, shmem_double_iput_real);
#endif

  if (shmem_double_iput_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_double_iput_ENTRY(target, source, tst, sst, nelems, pe);
    shmem_double_iput_real(target, source, tst, sst, nelems, pe);
    PROBE_shmem_double_iput_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_double_iput_real != NULL)
  {
    shmem_double_iput_real(target, source, tst, sst, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_double_iput was not hooked!\n");
    exit(-1);
  }
}

void shmem_float_iput (float *target, const float *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %p %p %d %d\n", TASKID, target, source, tst, sst, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_float_iput_real at %p\n", TASKID, shmem_float_iput_real);
#endif

  if (shmem_float_iput_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_float_iput_ENTRY(target, source, tst, sst, nelems, pe);
    shmem_float_iput_real(target, source, tst, sst, nelems, pe);
    PROBE_shmem_float_iput_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_float_iput_real != NULL)
  {
    shmem_float_iput_real(target, source, tst, sst, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_float_iput was not hooked!\n");
    exit(-1);
  }
}

void shmem_int_iput (int *target, const int *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %p %p %d %d\n", TASKID, target, source, tst, sst, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_int_iput_real at %p\n", TASKID, shmem_int_iput_real);
#endif

  if (shmem_int_iput_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_int_iput_ENTRY(target, source, tst, sst, nelems, pe);
    shmem_int_iput_real(target, source, tst, sst, nelems, pe);
    PROBE_shmem_int_iput_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_int_iput_real != NULL)
  {
    shmem_int_iput_real(target, source, tst, sst, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_int_iput was not hooked!\n");
    exit(-1);
  }
}

void shmem_iput32 (void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %p %p %d %d\n", TASKID, target, source, tst, sst, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_iput32_real at %p\n", TASKID, shmem_iput32_real);
#endif

  if (shmem_iput32_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_iput32_ENTRY(target, source, tst, sst, nelems, pe);
    shmem_iput32_real(target, source, tst, sst, nelems, pe);
    PROBE_shmem_iput32_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_iput32_real != NULL)
  {
    shmem_iput32_real(target, source, tst, sst, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_iput32 was not hooked!\n");
    exit(-1);
  }
}

void shmem_iput64 (void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %p %p %d %d\n", TASKID, target, source, tst, sst, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_iput64_real at %p\n", TASKID, shmem_iput64_real);
#endif

  if (shmem_iput64_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_iput64_ENTRY(target, source, tst, sst, nelems, pe);
    shmem_iput64_real(target, source, tst, sst, nelems, pe);
    PROBE_shmem_iput64_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_iput64_real != NULL)
  {
    shmem_iput64_real(target, source, tst, sst, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_iput64 was not hooked!\n");
    exit(-1);
  }
}

void shmem_iput128 (void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %p %p %d %d\n", TASKID, target, source, tst, sst, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_iput128_real at %p\n", TASKID, shmem_iput128_real);
#endif

  if (shmem_iput128_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_iput128_ENTRY(target, source, tst, sst, nelems, pe);
    shmem_iput128_real(target, source, tst, sst, nelems, pe);
    PROBE_shmem_iput128_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_iput128_real != NULL)
  {
    shmem_iput128_real(target, source, tst, sst, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_iput128 was not hooked!\n");
    exit(-1);
  }
}

void shmem_long_iput (long *target, const long *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %p %p %d %d\n", TASKID, target, source, tst, sst, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_long_iput_real at %p\n", TASKID, shmem_long_iput_real);
#endif

  if (shmem_long_iput_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_long_iput_ENTRY(target, source, tst, sst, nelems, pe);
    shmem_long_iput_real(target, source, tst, sst, nelems, pe);
    PROBE_shmem_long_iput_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_long_iput_real != NULL)
  {
    shmem_long_iput_real(target, source, tst, sst, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_long_iput was not hooked!\n");
    exit(-1);
  }
}

void shmem_longdouble_iput (long double *target, const long double *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %p %p %d %d\n", TASKID, target, source, tst, sst, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_longdouble_iput_real at %p\n", TASKID, shmem_longdouble_iput_real);
#endif

  if (shmem_longdouble_iput_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_longdouble_iput_ENTRY(target, source, tst, sst, nelems, pe);
    shmem_longdouble_iput_real(target, source, tst, sst, nelems, pe);
    PROBE_shmem_longdouble_iput_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_longdouble_iput_real != NULL)
  {
    shmem_longdouble_iput_real(target, source, tst, sst, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_longdouble_iput was not hooked!\n");
    exit(-1);
  }
}

void shmem_longlong_iput (long long *target, const long long *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %p %p %d %d\n", TASKID, target, source, tst, sst, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_longlong_iput_real at %p\n", TASKID, shmem_longlong_iput_real);
#endif

  if (shmem_longlong_iput_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_longlong_iput_ENTRY(target, source, tst, sst, nelems, pe);
    shmem_longlong_iput_real(target, source, tst, sst, nelems, pe);
    PROBE_shmem_longlong_iput_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_longlong_iput_real != NULL)
  {
    shmem_longlong_iput_real(target, source, tst, sst, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_longlong_iput was not hooked!\n");
    exit(-1);
  }
}

void shmem_short_iput (short *target, const short *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %p %p %d %d\n", TASKID, target, source, tst, sst, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_short_iput_real at %p\n", TASKID, shmem_short_iput_real);
#endif

  if (shmem_short_iput_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_short_iput_ENTRY(target, source, tst, sst, nelems, pe);
    shmem_short_iput_real(target, source, tst, sst, nelems, pe);
    PROBE_shmem_short_iput_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_short_iput_real != NULL)
  {
    shmem_short_iput_real(target, source, tst, sst, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_short_iput was not hooked!\n");
    exit(-1);
  }
}

void shmem_double_get (double *target, const double *source, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d\n", TASKID, target, source, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_double_get_real at %p\n", TASKID, shmem_double_get_real);
#endif

  if (shmem_double_get_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_double_get_ENTRY(target, source, nelems, pe);
    shmem_double_get_real(target, source, nelems, pe);
    PROBE_shmem_double_get_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_double_get_real != NULL)
  {
    shmem_double_get_real(target, source, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_double_get was not hooked!\n");
    exit(-1);
  }
}

void shmem_float_get (float *target, const float *source, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d\n", TASKID, target, source, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_float_get_real at %p\n", TASKID, shmem_float_get_real);
#endif

  if (shmem_float_get_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_float_get_ENTRY(target, source, nelems, pe);
    shmem_float_get_real(target, source, nelems, pe);
    PROBE_shmem_float_get_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_float_get_real != NULL)
  {
    shmem_float_get_real(target, source, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_float_get was not hooked!\n");
    exit(-1);
  }
}

void shmem_get32 (void *target, const void *source, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d\n", TASKID, target, source, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_get32_real at %p\n", TASKID, shmem_get32_real);
#endif

  if (shmem_get32_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_get32_ENTRY(target, source, nelems, pe);
    shmem_get32_real(target, source, nelems, pe);
    PROBE_shmem_get32_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_get32_real != NULL)
  {
    shmem_get32_real(target, source, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_get32 was not hooked!\n");
    exit(-1);
  }
}

void shmem_get64 (void *target, const void *source, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d\n", TASKID, target, source, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_get64_real at %p\n", TASKID, shmem_get64_real);
#endif

  if (shmem_get64_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_get64_ENTRY(target, source, nelems, pe);
    shmem_get64_real(target, source, nelems, pe);
    PROBE_shmem_get64_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_get64_real != NULL)
  {
    shmem_get64_real(target, source, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_get64 was not hooked!\n");
    exit(-1);
  }
}

void shmem_get128 (void *target, const void *source, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d\n", TASKID, target, source, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_get128_real at %p\n", TASKID, shmem_get128_real);
#endif

  if (shmem_get128_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_get128_ENTRY(target, source, nelems, pe);
    shmem_get128_real(target, source, nelems, pe);
    PROBE_shmem_get128_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_get128_real != NULL)
  {
    shmem_get128_real(target, source, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_get128 was not hooked!\n");
    exit(-1);
  }
}

void shmem_getmem (void *target, const void *source, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d\n", TASKID, target, source, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_getmem_real at %p\n", TASKID, shmem_getmem_real);
#endif

  if (shmem_getmem_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_getmem_ENTRY(target, source, nelems, pe);
    shmem_getmem_real(target, source, nelems, pe);
    PROBE_shmem_getmem_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_getmem_real != NULL)
  {
    shmem_getmem_real(target, source, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_getmem was not hooked!\n");
    exit(-1);
  }
}

void shmem_int_get (int *target, const int *source, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d\n", TASKID, target, source, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_int_get_real at %p\n", TASKID, shmem_int_get_real);
#endif

  if (shmem_int_get_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_int_get_ENTRY(target, source, nelems, pe);
    shmem_int_get_real(target, source, nelems, pe);
    PROBE_shmem_int_get_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_int_get_real != NULL)
  {
    shmem_int_get_real(target, source, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_int_get was not hooked!\n");
    exit(-1);
  }
}

void shmem_long_get (long *target, const long *source, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d\n", TASKID, target, source, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_long_get_real at %p\n", TASKID, shmem_long_get_real);
#endif

  if (shmem_long_get_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_long_get_ENTRY(target, source, nelems, pe);
    shmem_long_get_real(target, source, nelems, pe);
    PROBE_shmem_long_get_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_long_get_real != NULL)
  {
    shmem_long_get_real(target, source, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_long_get was not hooked!\n");
    exit(-1);
  }
}

void shmem_longdouble_get (long double *target, const long double *source, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d\n", TASKID, target, source, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_longdouble_get_real at %p\n", TASKID, shmem_longdouble_get_real);
#endif

  if (shmem_longdouble_get_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_longdouble_get_ENTRY(target, source, nelems, pe);
    shmem_longdouble_get_real(target, source, nelems, pe);
    PROBE_shmem_longdouble_get_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_longdouble_get_real != NULL)
  {
    shmem_longdouble_get_real(target, source, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_longdouble_get was not hooked!\n");
    exit(-1);
  }
}

void shmem_longlong_get (long long *target, const long long *source, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d\n", TASKID, target, source, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_longlong_get_real at %p\n", TASKID, shmem_longlong_get_real);
#endif

  if (shmem_longlong_get_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_longlong_get_ENTRY(target, source, nelems, pe);
    shmem_longlong_get_real(target, source, nelems, pe);
    PROBE_shmem_longlong_get_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_longlong_get_real != NULL)
  {
    shmem_longlong_get_real(target, source, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_longlong_get was not hooked!\n");
    exit(-1);
  }
}

void shmem_short_get (short *target, const short *source, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d\n", TASKID, target, source, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_short_get_real at %p\n", TASKID, shmem_short_get_real);
#endif

  if (shmem_short_get_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_short_get_ENTRY(target, source, nelems, pe);
    shmem_short_get_real(target, source, nelems, pe);
    PROBE_shmem_short_get_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_short_get_real != NULL)
  {
    shmem_short_get_real(target, source, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_short_get was not hooked!\n");
    exit(-1);
  }
}

char shmem_char_g (char *addr, int pe)
{
  char res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %d\n", TASKID, addr, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_char_g_real at %p\n", TASKID, shmem_char_g_real);
#endif

  if (shmem_char_g_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_char_g_ENTRY(addr, pe);
    res = shmem_char_g_real(addr, pe);
    PROBE_shmem_char_g_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_char_g_real != NULL)
  {
    res = shmem_char_g_real(addr, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_char_g was not hooked!\n");
    exit(-1);
  }
  return res;
}

short shmem_short_g (short *addr, int pe)
{
  short res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %d\n", TASKID, addr, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_short_g_real at %p\n", TASKID, shmem_short_g_real);
#endif

  if (shmem_short_g_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_short_g_ENTRY(addr, pe);
    res = shmem_short_g_real(addr, pe);
    PROBE_shmem_short_g_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_short_g_real != NULL)
  {
    res = shmem_short_g_real(addr, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_short_g was not hooked!\n");
    exit(-1);
  }
  return res;
}

int shmem_int_g (int *addr, int pe)
{
  int res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %d\n", TASKID, addr, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_int_g_real at %p\n", TASKID, shmem_int_g_real);
#endif

  if (shmem_int_g_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_int_g_ENTRY(addr, pe);
    res = shmem_int_g_real(addr, pe);
    PROBE_shmem_int_g_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_int_g_real != NULL)
  {
    res = shmem_int_g_real(addr, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_int_g was not hooked!\n");
    exit(-1);
  }
  return res;
}

long shmem_long_g (long *addr, int pe)
{
  long res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %d\n", TASKID, addr, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_long_g_real at %p\n", TASKID, shmem_long_g_real);
#endif

  if (shmem_long_g_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_long_g_ENTRY(addr, pe);
    res = shmem_long_g_real(addr, pe);
    PROBE_shmem_long_g_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_long_g_real != NULL)
  {
    res = shmem_long_g_real(addr, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_long_g was not hooked!\n");
    exit(-1);
  }
  return res;
}

long long shmem_longlong_g (long long *addr, int pe)
{
  long long res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %d\n", TASKID, addr, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_longlong_g_real at %p\n", TASKID, shmem_longlong_g_real);
#endif

  if (shmem_longlong_g_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_longlong_g_ENTRY(addr, pe);
    res = shmem_longlong_g_real(addr, pe);
    PROBE_shmem_longlong_g_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_longlong_g_real != NULL)
  {
    res = shmem_longlong_g_real(addr, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_longlong_g was not hooked!\n");
    exit(-1);
  }
  return res;
}

float shmem_float_g (float *addr, int pe)
{
  float res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %d\n", TASKID, addr, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_float_g_real at %p\n", TASKID, shmem_float_g_real);
#endif

  if (shmem_float_g_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_float_g_ENTRY(addr, pe);
    res = shmem_float_g_real(addr, pe);
    PROBE_shmem_float_g_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_float_g_real != NULL)
  {
    res = shmem_float_g_real(addr, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_float_g was not hooked!\n");
    exit(-1);
  }
  return res;
}

double shmem_double_g (double *addr, int pe)
{
  double res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %d\n", TASKID, addr, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_double_g_real at %p\n", TASKID, shmem_double_g_real);
#endif

  if (shmem_double_g_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_double_g_ENTRY(addr, pe);
    res = shmem_double_g_real(addr, pe);
    PROBE_shmem_double_g_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_double_g_real != NULL)
  {
    res = shmem_double_g_real(addr, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_double_g was not hooked!\n");
    exit(-1);
  }
  return res;
}

long double shmem_longdouble_g (long double *addr, int pe)
{
  long double res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %d\n", TASKID, addr, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_longdouble_g_real at %p\n", TASKID, shmem_longdouble_g_real);
#endif

  if (shmem_longdouble_g_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_longdouble_g_ENTRY(addr, pe);
    res = shmem_longdouble_g_real(addr, pe);
    PROBE_shmem_longdouble_g_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_longdouble_g_real != NULL)
  {
    res = shmem_longdouble_g_real(addr, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_longdouble_g was not hooked!\n");
    exit(-1);
  }
  return res;
}

void shmem_double_iget (double *target, const double *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %p %p %d %d\n", TASKID, target, source, tst, sst, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_double_iget_real at %p\n", TASKID, shmem_double_iget_real);
#endif

  if (shmem_double_iget_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_double_iget_ENTRY(target, source, tst, sst, nelems, pe);
    shmem_double_iget_real(target, source, tst, sst, nelems, pe);
    PROBE_shmem_double_iget_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_double_iget_real != NULL)
  {
    shmem_double_iget_real(target, source, tst, sst, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_double_iget was not hooked!\n");
    exit(-1);
  }
}

void shmem_float_iget (float *target, const float *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %p %p %d %d\n", TASKID, target, source, tst, sst, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_float_iget_real at %p\n", TASKID, shmem_float_iget_real);
#endif

  if (shmem_float_iget_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_float_iget_ENTRY(target, source, tst, sst, nelems, pe);
    shmem_float_iget_real(target, source, tst, sst, nelems, pe);
    PROBE_shmem_float_iget_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_float_iget_real != NULL)
  {
    shmem_float_iget_real(target, source, tst, sst, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_float_iget was not hooked!\n");
    exit(-1);
  }
}

void shmem_iget32 (void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %p %p %d %d\n", TASKID, target, source, tst, sst, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_iget32_real at %p\n", TASKID, shmem_iget32_real);
#endif

  if (shmem_iget32_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_iget32_ENTRY(target, source, tst, sst, nelems, pe);
    shmem_iget32_real(target, source, tst, sst, nelems, pe);
    PROBE_shmem_iget32_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_iget32_real != NULL)
  {
    shmem_iget32_real(target, source, tst, sst, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_iget32 was not hooked!\n");
    exit(-1);
  }
}

void shmem_iget64 (void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %p %p %d %d\n", TASKID, target, source, tst, sst, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_iget64_real at %p\n", TASKID, shmem_iget64_real);
#endif

  if (shmem_iget64_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_iget64_ENTRY(target, source, tst, sst, nelems, pe);
    shmem_iget64_real(target, source, tst, sst, nelems, pe);
    PROBE_shmem_iget64_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_iget64_real != NULL)
  {
    shmem_iget64_real(target, source, tst, sst, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_iget64 was not hooked!\n");
    exit(-1);
  }
}

void shmem_iget128 (void *target, const void *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %p %p %d %d\n", TASKID, target, source, tst, sst, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_iget128_real at %p\n", TASKID, shmem_iget128_real);
#endif

  if (shmem_iget128_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_iget128_ENTRY(target, source, tst, sst, nelems, pe);
    shmem_iget128_real(target, source, tst, sst, nelems, pe);
    PROBE_shmem_iget128_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_iget128_real != NULL)
  {
    shmem_iget128_real(target, source, tst, sst, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_iget128 was not hooked!\n");
    exit(-1);
  }
}

void shmem_int_iget (int *target, const int *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %p %p %d %d\n", TASKID, target, source, tst, sst, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_int_iget_real at %p\n", TASKID, shmem_int_iget_real);
#endif

  if (shmem_int_iget_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_int_iget_ENTRY(target, source, tst, sst, nelems, pe);
    shmem_int_iget_real(target, source, tst, sst, nelems, pe);
    PROBE_shmem_int_iget_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_int_iget_real != NULL)
  {
    shmem_int_iget_real(target, source, tst, sst, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_int_iget was not hooked!\n");
    exit(-1);
  }
}

void shmem_long_iget (long *target, const long *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %p %p %d %d\n", TASKID, target, source, tst, sst, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_long_iget_real at %p\n", TASKID, shmem_long_iget_real);
#endif

  if (shmem_long_iget_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_long_iget_ENTRY(target, source, tst, sst, nelems, pe);
    shmem_long_iget_real(target, source, tst, sst, nelems, pe);
    PROBE_shmem_long_iget_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_long_iget_real != NULL)
  {
    shmem_long_iget_real(target, source, tst, sst, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_long_iget was not hooked!\n");
    exit(-1);
  }
}

void shmem_longdouble_iget (long double *target, const long double *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %p %p %d %d\n", TASKID, target, source, tst, sst, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_longdouble_iget_real at %p\n", TASKID, shmem_longdouble_iget_real);
#endif

  if (shmem_longdouble_iget_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_longdouble_iget_ENTRY(target, source, tst, sst, nelems, pe);
    shmem_longdouble_iget_real(target, source, tst, sst, nelems, pe);
    PROBE_shmem_longdouble_iget_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_longdouble_iget_real != NULL)
  {
    shmem_longdouble_iget_real(target, source, tst, sst, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_longdouble_iget was not hooked!\n");
    exit(-1);
  }
}

void shmem_longlong_iget (long long *target, const long long *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %p %p %d %d\n", TASKID, target, source, tst, sst, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_longlong_iget_real at %p\n", TASKID, shmem_longlong_iget_real);
#endif

  if (shmem_longlong_iget_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_longlong_iget_ENTRY(target, source, tst, sst, nelems, pe);
    shmem_longlong_iget_real(target, source, tst, sst, nelems, pe);
    PROBE_shmem_longlong_iget_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_longlong_iget_real != NULL)
  {
    shmem_longlong_iget_real(target, source, tst, sst, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_longlong_iget was not hooked!\n");
    exit(-1);
  }
}

void shmem_short_iget (short *target, const short *source, ptrdiff_t tst, ptrdiff_t sst, size_t nelems, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %p %p %d %d\n", TASKID, target, source, tst, sst, nelems, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_short_iget_real at %p\n", TASKID, shmem_short_iget_real);
#endif

  if (shmem_short_iget_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_short_iget_ENTRY(target, source, tst, sst, nelems, pe);
    shmem_short_iget_real(target, source, tst, sst, nelems, pe);
    PROBE_shmem_short_iget_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_short_iget_real != NULL)
  {
    shmem_short_iget_real(target, source, tst, sst, nelems, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_short_iget was not hooked!\n");
    exit(-1);
  }
}

void shmem_int_add (int *target, int value, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %d %d\n", TASKID, target, value, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_int_add_real at %p\n", TASKID, shmem_int_add_real);
#endif

  if (shmem_int_add_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_int_add_ENTRY(target, value, pe);
    shmem_int_add_real(target, value, pe);
    PROBE_shmem_int_add_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_int_add_real != NULL)
  {
    shmem_int_add_real(target, value, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_int_add was not hooked!\n");
    exit(-1);
  }
}

void shmem_long_add (long *target, long value, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %ld %d\n", TASKID, target, value, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_long_add_real at %p\n", TASKID, shmem_long_add_real);
#endif

  if (shmem_long_add_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_long_add_ENTRY(target, value, pe);
    shmem_long_add_real(target, value, pe);
    PROBE_shmem_long_add_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_long_add_real != NULL)
  {
    shmem_long_add_real(target, value, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_long_add was not hooked!\n");
    exit(-1);
  }
}

void shmem_longlong_add (long long *target, long long value, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %lld %d\n", TASKID, target, value, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_longlong_add_real at %p\n", TASKID, shmem_longlong_add_real);
#endif

  if (shmem_longlong_add_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_longlong_add_ENTRY(target, value, pe);
    shmem_longlong_add_real(target, value, pe);
    PROBE_shmem_longlong_add_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_longlong_add_real != NULL)
  {
    shmem_longlong_add_real(target, value, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_longlong_add was not hooked!\n");
    exit(-1);
  }
}

int shmem_int_cswap (int *target, int cond, int value, int pe)
{
  int res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %d %d %d\n", TASKID, target, cond, value, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_int_cswap_real at %p\n", TASKID, shmem_int_cswap_real);
#endif

  if (shmem_int_cswap_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_int_cswap_ENTRY(target, cond, value, pe);
    res = shmem_int_cswap_real(target, cond, value, pe);
    PROBE_shmem_int_cswap_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_int_cswap_real != NULL)
  {
    res = shmem_int_cswap_real(target, cond, value, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_int_cswap was not hooked!\n");
    exit(-1);
  }
  return res;
}

long shmem_long_cswap (long *target, long cond, long value, int pe)
{
  long res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %ld %ld %d\n", TASKID, target, cond, value, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_long_cswap_real at %p\n", TASKID, shmem_long_cswap_real);
#endif

  if (shmem_long_cswap_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_long_cswap_ENTRY(target, cond, value, pe);
    res = shmem_long_cswap_real(target, cond, value, pe);
    PROBE_shmem_long_cswap_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_long_cswap_real != NULL)
  {
    res = shmem_long_cswap_real(target, cond, value, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_long_cswap was not hooked!\n");
    exit(-1);
  }
  return res;
}

long long shmem_longlong_cswap (long long *target, long long cond, long long value, int pe)
{
  long long res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %lld %lld %d\n", TASKID, target, cond, value, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_longlong_cswap_real at %p\n", TASKID, shmem_longlong_cswap_real);
#endif

  if (shmem_longlong_cswap_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_longlong_cswap_ENTRY(target, cond, value, pe);
    res = shmem_longlong_cswap_real(target, cond, value, pe);
    PROBE_shmem_longlong_cswap_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_longlong_cswap_real != NULL)
  {
    res = shmem_longlong_cswap_real(target, cond, value, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_longlong_cswap was not hooked!\n");
    exit(-1);
  }
  return res;
}

double shmem_double_swap (double *target, double value, int pe)
{
  double res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %lf %d\n", TASKID, target, value, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_double_swap_real at %p\n", TASKID, shmem_double_swap_real);
#endif

  if (shmem_double_swap_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_double_swap_ENTRY(target, value, pe);
    res = shmem_double_swap_real(target, value, pe);
    PROBE_shmem_double_swap_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_double_swap_real != NULL)
  {
    res = shmem_double_swap_real(target, value, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_double_swap was not hooked!\n");
    exit(-1);
  }
  return res;
}

float shmem_float_swap (float *target, float value, int pe)
{
  float res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %f %d\n", TASKID, target, value, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_float_swap_real at %p\n", TASKID, shmem_float_swap_real);
#endif

  if (shmem_float_swap_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_float_swap_ENTRY(target, value, pe);
    res = shmem_float_swap_real(target, value, pe);
    PROBE_shmem_float_swap_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_float_swap_real != NULL)
  {
    res = shmem_float_swap_real(target, value, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_float_swap was not hooked!\n");
    exit(-1);
  }
  return res;
}

int shmem_int_swap (int *target, int value, int pe)
{
  int res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %d %d\n", TASKID, target, value, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_int_swap_real at %p\n", TASKID, shmem_int_swap_real);
#endif

  if (shmem_int_swap_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_int_swap_ENTRY(target, value, pe);
    res = shmem_int_swap_real(target, value, pe);
    PROBE_shmem_int_swap_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_int_swap_real != NULL)
  {
    res = shmem_int_swap_real(target, value, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_int_swap was not hooked!\n");
    exit(-1);
  }
  return res;
}

long shmem_long_swap (long *target, long value, int pe)
{
  long res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %ld %d\n", TASKID, target, value, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_long_swap_real at %p\n", TASKID, shmem_long_swap_real);
#endif

  if (shmem_long_swap_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_long_swap_ENTRY(target, value, pe);
    res = shmem_long_swap_real(target, value, pe);
    PROBE_shmem_long_swap_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_long_swap_real != NULL)
  {
    res = shmem_long_swap_real(target, value, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_long_swap was not hooked!\n");
    exit(-1);
  }
  return res;
}

long long shmem_longlong_swap (long long *target, long long value, int pe)
{
  long long res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %lld %d\n", TASKID, target, value, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_longlong_swap_real at %p\n", TASKID, shmem_longlong_swap_real);
#endif

  if (shmem_longlong_swap_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_longlong_swap_ENTRY(target, value, pe);
    res = shmem_longlong_swap_real(target, value, pe);
    PROBE_shmem_longlong_swap_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_longlong_swap_real != NULL)
  {
    res = shmem_longlong_swap_real(target, value, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_longlong_swap was not hooked!\n");
    exit(-1);
  }
  return res;
}

long shmem_swap (long *target, long value, int pe)
{
  long res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %ld %d\n", TASKID, target, value, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_swap_real at %p\n", TASKID, shmem_swap_real);
#endif

  if (shmem_swap_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_swap_ENTRY(target, value, pe);
    res = shmem_swap_real(target, value, pe);
    PROBE_shmem_swap_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_swap_real != NULL)
  {
    res = shmem_swap_real(target, value, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_swap was not hooked!\n");
    exit(-1);
  }
  return res;
}

int shmem_int_finc (int *target, int pe)
{
  int res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %d\n", TASKID, target, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_int_finc_real at %p\n", TASKID, shmem_int_finc_real);
#endif

  if (shmem_int_finc_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_int_finc_ENTRY(target, pe);
    res = shmem_int_finc_real(target, pe);
    PROBE_shmem_int_finc_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_int_finc_real != NULL)
  {
    res = shmem_int_finc_real(target, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_int_finc was not hooked!\n");
    exit(-1);
  }
  return res;
}

long shmem_long_finc (long *target, int pe)
{
  long res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %d\n", TASKID, target, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_long_finc_real at %p\n", TASKID, shmem_long_finc_real);
#endif

  if (shmem_long_finc_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_long_finc_ENTRY(target, pe);
    res = shmem_long_finc_real(target, pe);
    PROBE_shmem_long_finc_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_long_finc_real != NULL)
  {
    res = shmem_long_finc_real(target, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_long_finc was not hooked!\n");
    exit(-1);
  }
  return res;
}

long long shmem_longlong_finc (long long *target, int pe)
{
  long long res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %d\n", TASKID, target, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_longlong_finc_real at %p\n", TASKID, shmem_longlong_finc_real);
#endif

  if (shmem_longlong_finc_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_longlong_finc_ENTRY(target, pe);
    res = shmem_longlong_finc_real(target, pe);
    PROBE_shmem_longlong_finc_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_longlong_finc_real != NULL)
  {
    res = shmem_longlong_finc_real(target, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_longlong_finc was not hooked!\n");
    exit(-1);
  }
  return res;
}

void shmem_int_inc (int *target, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %d\n", TASKID, target, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_int_inc_real at %p\n", TASKID, shmem_int_inc_real);
#endif

  if (shmem_int_inc_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_int_inc_ENTRY(target, pe);
    shmem_int_inc_real(target, pe);
    PROBE_shmem_int_inc_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_int_inc_real != NULL)
  {
    shmem_int_inc_real(target, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_int_inc was not hooked!\n");
    exit(-1);
  }
}

void shmem_long_inc (long *target, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %d\n", TASKID, target, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_long_inc_real at %p\n", TASKID, shmem_long_inc_real);
#endif

  if (shmem_long_inc_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_long_inc_ENTRY(target, pe);
    shmem_long_inc_real(target, pe);
    PROBE_shmem_long_inc_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_long_inc_real != NULL)
  {
    shmem_long_inc_real(target, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_long_inc was not hooked!\n");
    exit(-1);
  }
}

void shmem_longlong_inc (long long *target, int pe)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %d\n", TASKID, target, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_longlong_inc_real at %p\n", TASKID, shmem_longlong_inc_real);
#endif

  if (shmem_longlong_inc_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_longlong_inc_ENTRY(target, pe);
    shmem_longlong_inc_real(target, pe);
    PROBE_shmem_longlong_inc_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_longlong_inc_real != NULL)
  {
    shmem_longlong_inc_real(target, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_longlong_inc was not hooked!\n");
    exit(-1);
  }
}

int shmem_int_fadd (int *target, int value, int pe)
{
  int res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %d %d\n", TASKID, target, value, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_int_fadd_real at %p\n", TASKID, shmem_int_fadd_real);
#endif

  if (shmem_int_fadd_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_int_fadd_ENTRY(target, value, pe);
    res = shmem_int_fadd_real(target, value, pe);
    PROBE_shmem_int_fadd_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_int_fadd_real != NULL)
  {
    res = shmem_int_fadd_real(target, value, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_int_fadd was not hooked!\n");
    exit(-1);
  }
  return res;
}

long shmem_long_fadd (long *target, long value, int pe)
{
  long res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %ld %d\n", TASKID, target, value, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_long_fadd_real at %p\n", TASKID, shmem_long_fadd_real);
#endif

  if (shmem_long_fadd_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_long_fadd_ENTRY(target, value, pe);
    res = shmem_long_fadd_real(target, value, pe);
    PROBE_shmem_long_fadd_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_long_fadd_real != NULL)
  {
    res = shmem_long_fadd_real(target, value, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_long_fadd was not hooked!\n");
    exit(-1);
  }
  return res;
}

long long shmem_longlong_fadd (long long *target, long long value, int pe)
{
  long long res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %lld %d\n", TASKID, target, value, pe);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_longlong_fadd_real at %p\n", TASKID, shmem_longlong_fadd_real);
#endif

  if (shmem_longlong_fadd_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_longlong_fadd_ENTRY(target, value, pe);
    res = shmem_longlong_fadd_real(target, value, pe);
    PROBE_shmem_longlong_fadd_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_longlong_fadd_real != NULL)
  {
    res = shmem_longlong_fadd_real(target, value, pe);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_longlong_fadd was not hooked!\n");
    exit(-1);
  }
  return res;
}

void shmem_barrier_all (void)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_barrier_all_real at %p\n", TASKID, shmem_barrier_all_real);
#endif

  if (shmem_barrier_all_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_barrier_all_ENTRY();
    shmem_barrier_all_real();
    PROBE_shmem_barrier_all_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_barrier_all_real != NULL)
  {
    shmem_barrier_all_real();
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_barrier_all was not hooked!\n");
    exit(-1);
  }
}

void shmem_barrier (int PE_start, int logPE_stride, int PE_size, long *pSync)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %d %d %d %p\n", TASKID, PE_start, logPE_stride, PE_size, pSync);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_barrier_real at %p\n", TASKID, shmem_barrier_real);
#endif

  if (shmem_barrier_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_barrier_ENTRY(PE_start, logPE_stride, PE_size, pSync);
    shmem_barrier_real(PE_start, logPE_stride, PE_size, pSync);
    PROBE_shmem_barrier_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_barrier_real != NULL)
  {
    shmem_barrier_real(PE_start, logPE_stride, PE_size, pSync);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_barrier was not hooked!\n");
    exit(-1);
  }
}

void shmem_broadcast32 (void *target, const void *source, size_t nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long *pSync)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d %d %d %d %p\n", TASKID, target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_broadcast32_real at %p\n", TASKID, shmem_broadcast32_real);
#endif

  if (shmem_broadcast32_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_broadcast32_ENTRY(target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync);
    shmem_broadcast32_real(target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync);
    PROBE_shmem_broadcast32_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_broadcast32_real != NULL)
  {
    shmem_broadcast32_real(target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_broadcast32 was not hooked!\n");
    exit(-1);
  }
}

void shmem_broadcast64 (void *target, const void *source, size_t nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long *pSync)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d %d %d %d %p\n", TASKID, target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_broadcast64_real at %p\n", TASKID, shmem_broadcast64_real);
#endif

  if (shmem_broadcast64_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_broadcast64_ENTRY(target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync);
    shmem_broadcast64_real(target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync);
    PROBE_shmem_broadcast64_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_broadcast64_real != NULL)
  {
    shmem_broadcast64_real(target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_broadcast64 was not hooked!\n");
    exit(-1);
  }
}

void shmem_collect32 (void *target, const void *source, size_t nelems, int PE_start, int logPE_stride, int PE_size, long *pSync)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d %d %d %p\n", TASKID, target, source, nelems, PE_start, logPE_stride, PE_size, pSync);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_collect32_real at %p\n", TASKID, shmem_collect32_real);
#endif

  if (shmem_collect32_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_collect32_ENTRY(target, source, nelems, PE_start, logPE_stride, PE_size, pSync);
    shmem_collect32_real(target, source, nelems, PE_start, logPE_stride, PE_size, pSync);
    PROBE_shmem_collect32_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_collect32_real != NULL)
  {
    shmem_collect32_real(target, source, nelems, PE_start, logPE_stride, PE_size, pSync);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_collect32 was not hooked!\n");
    exit(-1);
  }
}

void shmem_collect64 (void *target, const void *source, size_t nelems, int PE_start, int logPE_stride, int PE_size, long *pSync)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d %d %d %p\n", TASKID, target, source, nelems, PE_start, logPE_stride, PE_size, pSync);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_collect64_real at %p\n", TASKID, shmem_collect64_real);
#endif

  if (shmem_collect64_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_collect64_ENTRY(target, source, nelems, PE_start, logPE_stride, PE_size, pSync);
    shmem_collect64_real(target, source, nelems, PE_start, logPE_stride, PE_size, pSync);
    PROBE_shmem_collect64_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_collect64_real != NULL)
  {
    shmem_collect64_real(target, source, nelems, PE_start, logPE_stride, PE_size, pSync);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_collect64 was not hooked!\n");
    exit(-1);
  }
}

void shmem_fcollect32 (void *target, const void *source, size_t nelems, int PE_start, int logPE_stride, int PE_size, long *pSync)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d %d %d %p\n", TASKID, target, source, nelems, PE_start, logPE_stride, PE_size, pSync);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_fcollect32_real at %p\n", TASKID, shmem_fcollect32_real);
#endif

  if (shmem_fcollect32_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_fcollect32_ENTRY(target, source, nelems, PE_start, logPE_stride, PE_size, pSync);
    shmem_fcollect32_real(target, source, nelems, PE_start, logPE_stride, PE_size, pSync);
    PROBE_shmem_fcollect32_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_fcollect32_real != NULL)
  {
    shmem_fcollect32_real(target, source, nelems, PE_start, logPE_stride, PE_size, pSync);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_fcollect32 was not hooked!\n");
    exit(-1);
  }
}

void shmem_fcollect64 (void *target, const void *source, size_t nelems, int PE_start, int logPE_stride, int PE_size, long *pSync)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d %d %d %p\n", TASKID, target, source, nelems, PE_start, logPE_stride, PE_size, pSync);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_fcollect64_real at %p\n", TASKID, shmem_fcollect64_real);
#endif

  if (shmem_fcollect64_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_fcollect64_ENTRY(target, source, nelems, PE_start, logPE_stride, PE_size, pSync);
    shmem_fcollect64_real(target, source, nelems, PE_start, logPE_stride, PE_size, pSync);
    PROBE_shmem_fcollect64_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_fcollect64_real != NULL)
  {
    shmem_fcollect64_real(target, source, nelems, PE_start, logPE_stride, PE_size, pSync);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_fcollect64 was not hooked!\n");
    exit(-1);
  }
}

void shmem_int_and_to_all (int *target, int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d %d %d %p %p\n", TASKID, target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_int_and_to_all_real at %p\n", TASKID, shmem_int_and_to_all_real);
#endif

  if (shmem_int_and_to_all_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_int_and_to_all_ENTRY(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
    shmem_int_and_to_all_real(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
    PROBE_shmem_int_and_to_all_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_int_and_to_all_real != NULL)
  {
    shmem_int_and_to_all_real(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_int_and_to_all was not hooked!\n");
    exit(-1);
  }
}

void shmem_long_and_to_all (long *target, long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d %d %d %p %p\n", TASKID, target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_long_and_to_all_real at %p\n", TASKID, shmem_long_and_to_all_real);
#endif

  if (shmem_long_and_to_all_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_long_and_to_all_ENTRY(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
    shmem_long_and_to_all_real(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
    PROBE_shmem_long_and_to_all_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_long_and_to_all_real != NULL)
  {
    shmem_long_and_to_all_real(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_long_and_to_all was not hooked!\n");
    exit(-1);
  }
}

void shmem_longlong_and_to_all (long long *target, long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d %d %d %p %p\n", TASKID, target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_longlong_and_to_all_real at %p\n", TASKID, shmem_longlong_and_to_all_real);
#endif

  if (shmem_longlong_and_to_all_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_longlong_and_to_all_ENTRY(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
    shmem_longlong_and_to_all_real(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
    PROBE_shmem_longlong_and_to_all_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_longlong_and_to_all_real != NULL)
  {
    shmem_longlong_and_to_all_real(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_longlong_and_to_all was not hooked!\n");
    exit(-1);
  }
}

void shmem_short_and_to_all (short *target, short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d %d %d %p %p\n", TASKID, target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_short_and_to_all_real at %p\n", TASKID, shmem_short_and_to_all_real);
#endif

  if (shmem_short_and_to_all_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_short_and_to_all_ENTRY(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
    shmem_short_and_to_all_real(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
    PROBE_shmem_short_and_to_all_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_short_and_to_all_real != NULL)
  {
    shmem_short_and_to_all_real(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_short_and_to_all was not hooked!\n");
    exit(-1);
  }
}

void shmem_double_max_to_all (double *target, double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, double *pWrk, long *pSync)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d %d %d %p %p\n", TASKID, target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_double_max_to_all_real at %p\n", TASKID, shmem_double_max_to_all_real);
#endif

  if (shmem_double_max_to_all_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_double_max_to_all_ENTRY(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
    shmem_double_max_to_all_real(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
    PROBE_shmem_double_max_to_all_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_double_max_to_all_real != NULL)
  {
    shmem_double_max_to_all_real(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_double_max_to_all was not hooked!\n");
    exit(-1);
  }
}

void shmem_float_max_to_all (float *target, float *source, int nreduce, int PE_start, int logPE_stride, int PE_size, float *pWrk, long *pSync)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d %d %d %p %p\n", TASKID, target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_float_max_to_all_real at %p\n", TASKID, shmem_float_max_to_all_real);
#endif

  if (shmem_float_max_to_all_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_float_max_to_all_ENTRY(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
    shmem_float_max_to_all_real(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
    PROBE_shmem_float_max_to_all_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_float_max_to_all_real != NULL)
  {
    shmem_float_max_to_all_real(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_float_max_to_all was not hooked!\n");
    exit(-1);
  }
}

void shmem_int_max_to_all (int *target, int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d %d %d %p %p\n", TASKID, target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_int_max_to_all_real at %p\n", TASKID, shmem_int_max_to_all_real);
#endif

  if (shmem_int_max_to_all_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_int_max_to_all_ENTRY(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
    shmem_int_max_to_all_real(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
    PROBE_shmem_int_max_to_all_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_int_max_to_all_real != NULL)
  {
    shmem_int_max_to_all_real(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_int_max_to_all was not hooked!\n");
    exit(-1);
  }
}

void shmem_long_max_to_all (long *target, long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d %d %d %p %p\n", TASKID, target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_long_max_to_all_real at %p\n", TASKID, shmem_long_max_to_all_real);
#endif

  if (shmem_long_max_to_all_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_long_max_to_all_ENTRY(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
    shmem_long_max_to_all_real(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
    PROBE_shmem_long_max_to_all_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_long_max_to_all_real != NULL)
  {
    shmem_long_max_to_all_real(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_long_max_to_all was not hooked!\n");
    exit(-1);
  }
}

void shmem_longdouble_max_to_all (long double *target, long double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double *pWrk, long *pSync)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d %d %d %p %p\n", TASKID, target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_longdouble_max_to_all_real at %p\n", TASKID, shmem_longdouble_max_to_all_real);
#endif

  if (shmem_longdouble_max_to_all_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_longdouble_max_to_all_ENTRY(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
    shmem_longdouble_max_to_all_real(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
    PROBE_shmem_longdouble_max_to_all_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_longdouble_max_to_all_real != NULL)
  {
    shmem_longdouble_max_to_all_real(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_longdouble_max_to_all was not hooked!\n");
    exit(-1);
  }
}

void shmem_longlong_max_to_all (long long *target, long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d %d %d %p %p\n", TASKID, target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_longlong_max_to_all_real at %p\n", TASKID, shmem_longlong_max_to_all_real);
#endif

  if (shmem_longlong_max_to_all_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_longlong_max_to_all_ENTRY(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
    shmem_longlong_max_to_all_real(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
    PROBE_shmem_longlong_max_to_all_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_longlong_max_to_all_real != NULL)
  {
    shmem_longlong_max_to_all_real(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_longlong_max_to_all was not hooked!\n");
    exit(-1);
  }
}

void shmem_short_max_to_all (short *target, short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d %d %d %p %p\n", TASKID, target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_short_max_to_all_real at %p\n", TASKID, shmem_short_max_to_all_real);
#endif

  if (shmem_short_max_to_all_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_short_max_to_all_ENTRY(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
    shmem_short_max_to_all_real(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
    PROBE_shmem_short_max_to_all_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_short_max_to_all_real != NULL)
  {
    shmem_short_max_to_all_real(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_short_max_to_all was not hooked!\n");
    exit(-1);
  }
}

void shmem_double_min_to_all (double *target, double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, double *pWrk, long *pSync)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %p %d %d %d %d %p %p\n", TASKID, target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_double_min_to_all_real at %p\n", TASKID, shmem_double_min_to_all_real);
#endif

  if (shmem_double_min_to_all_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_double_min_to_all_ENTRY(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
    shmem_double_min_to_all_real(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
    PROBE_shmem_double_min_to_all_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_double_min_to_all_real != NULL)
  {
    shmem_double_min_to_all_real(target, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_double_min_to_all was not hooked!\n");
    exit(-1);
  }
}

void shmem_int_wait (int *ivar, int cmp_value)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %d\n", TASKID, ivar, cmp_value);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_int_wait_real at %p\n", TASKID, shmem_int_wait_real);
#endif

  if (shmem_int_wait_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_int_wait_ENTRY(ivar, cmp_value);
    shmem_int_wait_real(ivar, cmp_value);
    PROBE_shmem_int_wait_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_int_wait_real != NULL)
  {
    shmem_int_wait_real(ivar, cmp_value);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_int_wait was not hooked!\n");
    exit(-1);
  }
}

void shmem_int_wait_until (int *ivar, int cmp, int cmp_value)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %d %d\n", TASKID, ivar, cmp, cmp_value);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_int_wait_until_real at %p\n", TASKID, shmem_int_wait_until_real);
#endif

  if (shmem_int_wait_until_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_int_wait_until_ENTRY(ivar, cmp, cmp_value);
    shmem_int_wait_until_real(ivar, cmp, cmp_value);
    PROBE_shmem_int_wait_until_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_int_wait_until_real != NULL)
  {
    shmem_int_wait_until_real(ivar, cmp, cmp_value);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_int_wait_until was not hooked!\n");
    exit(-1);
  }
}

void shmem_long_wait (long *ivar, long cmp_value)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %ld\n", TASKID, ivar, cmp_value);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_long_wait_real at %p\n", TASKID, shmem_long_wait_real);
#endif

  if (shmem_long_wait_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_long_wait_ENTRY(ivar, cmp_value);
    shmem_long_wait_real(ivar, cmp_value);
    PROBE_shmem_long_wait_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_long_wait_real != NULL)
  {
    shmem_long_wait_real(ivar, cmp_value);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_long_wait was not hooked!\n");
    exit(-1);
  }
}

void shmem_long_wait_until (long *ivar, int cmp, long cmp_value)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %d %ld\n", TASKID, ivar, cmp, cmp_value);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_long_wait_until_real at %p\n", TASKID, shmem_long_wait_until_real);
#endif

  if (shmem_long_wait_until_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_long_wait_until_ENTRY(ivar, cmp, cmp_value);
    shmem_long_wait_until_real(ivar, cmp, cmp_value);
    PROBE_shmem_long_wait_until_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_long_wait_until_real != NULL)
  {
    shmem_long_wait_until_real(ivar, cmp, cmp_value);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_long_wait_until was not hooked!\n");
    exit(-1);
  }
}

void shmem_longlong_wait (long long *ivar, long long cmp_value)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %lld\n", TASKID, ivar, cmp_value);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_longlong_wait_real at %p\n", TASKID, shmem_longlong_wait_real);
#endif

  if (shmem_longlong_wait_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_longlong_wait_ENTRY(ivar, cmp_value);
    shmem_longlong_wait_real(ivar, cmp_value);
    PROBE_shmem_longlong_wait_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_longlong_wait_real != NULL)
  {
    shmem_longlong_wait_real(ivar, cmp_value);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_longlong_wait was not hooked!\n");
    exit(-1);
  }
}

void shmem_longlong_wait_until (long long *ivar, int cmp, long long cmp_value)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %d %lld\n", TASKID, ivar, cmp, cmp_value);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_longlong_wait_until_real at %p\n", TASKID, shmem_longlong_wait_until_real);
#endif

  if (shmem_longlong_wait_until_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_longlong_wait_until_ENTRY(ivar, cmp, cmp_value);
    shmem_longlong_wait_until_real(ivar, cmp, cmp_value);
    PROBE_shmem_longlong_wait_until_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_longlong_wait_until_real != NULL)
  {
    shmem_longlong_wait_until_real(ivar, cmp, cmp_value);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_longlong_wait_until was not hooked!\n");
    exit(-1);
  }
}

void shmem_short_wait (short *ivar, short cmp_value)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %h\n", TASKID, ivar, cmp_value);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_short_wait_real at %p\n", TASKID, shmem_short_wait_real);
#endif

  if (shmem_short_wait_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_short_wait_ENTRY(ivar, cmp_value);
    shmem_short_wait_real(ivar, cmp_value);
    PROBE_shmem_short_wait_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_short_wait_real != NULL)
  {
    shmem_short_wait_real(ivar, cmp_value);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_short_wait was not hooked!\n");
    exit(-1);
  }
}

void shmem_short_wait_until (short *ivar, int cmp, short cmp_value)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %d %h\n", TASKID, ivar, cmp, cmp_value);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_short_wait_until_real at %p\n", TASKID, shmem_short_wait_until_real);
#endif

  if (shmem_short_wait_until_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_short_wait_until_ENTRY(ivar, cmp, cmp_value);
    shmem_short_wait_until_real(ivar, cmp, cmp_value);
    PROBE_shmem_short_wait_until_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_short_wait_until_real != NULL)
  {
    shmem_short_wait_until_real(ivar, cmp, cmp_value);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_short_wait_until was not hooked!\n");
    exit(-1);
  }
}

void shmem_wait (long *ivar, long cmp_value)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %ld\n", TASKID, ivar, cmp_value);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_wait_real at %p\n", TASKID, shmem_wait_real);
#endif

  if (shmem_wait_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_wait_ENTRY(ivar, cmp_value);
    shmem_wait_real(ivar, cmp_value);
    PROBE_shmem_wait_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_wait_real != NULL)
  {
    shmem_wait_real(ivar, cmp_value);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_wait was not hooked!\n");
    exit(-1);
  }
}

void shmem_wait_until (long *ivar, int cmp, long cmp_value)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p %d %ld\n", TASKID, ivar, cmp, cmp_value);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_wait_until_real at %p\n", TASKID, shmem_wait_until_real);
#endif

  if (shmem_wait_until_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_wait_until_ENTRY(ivar, cmp, cmp_value);
    shmem_wait_until_real(ivar, cmp, cmp_value);
    PROBE_shmem_wait_until_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_wait_until_real != NULL)
  {
    shmem_wait_until_real(ivar, cmp, cmp_value);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_wait_until was not hooked!\n");
    exit(-1);
  }
}

void shmem_fence (void)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_fence_real at %p\n", TASKID, shmem_fence_real);
#endif

  if (shmem_fence_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_fence_ENTRY();
    shmem_fence_real();
    PROBE_shmem_fence_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_fence_real != NULL)
  {
    shmem_fence_real();
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_fence was not hooked!\n");
    exit(-1);
  }
}

void shmem_quiet (void)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_quiet_real at %p\n", TASKID, shmem_quiet_real);
#endif

  if (shmem_quiet_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_quiet_ENTRY();
    shmem_quiet_real();
    PROBE_shmem_quiet_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_quiet_real != NULL)
  {
    shmem_quiet_real();
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_quiet was not hooked!\n");
    exit(-1);
  }
}

void shmem_clear_lock (long *lock)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p\n", TASKID, lock);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_clear_lock_real at %p\n", TASKID, shmem_clear_lock_real);
#endif

  if (shmem_clear_lock_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_clear_lock_ENTRY(lock);
    shmem_clear_lock_real(lock);
    PROBE_shmem_clear_lock_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_clear_lock_real != NULL)
  {
    shmem_clear_lock_real(lock);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_clear_lock was not hooked!\n");
    exit(-1);
  }
}

void shmem_set_lock (long *lock)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p\n", TASKID, lock);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_set_lock_real at %p\n", TASKID, shmem_set_lock_real);
#endif

  if (shmem_set_lock_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_set_lock_ENTRY(lock);
    shmem_set_lock_real(lock);
    PROBE_shmem_set_lock_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_set_lock_real != NULL)
  {
    shmem_set_lock_real(lock);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_set_lock was not hooked!\n");
    exit(-1);
  }
}

int shmem_test_lock (long *lock)
{
  int res = 0;

#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p\n", TASKID, lock);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_test_lock_real at %p\n", TASKID, shmem_test_lock_real);
#endif

  if (shmem_test_lock_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_test_lock_ENTRY(lock);
    res = shmem_test_lock_real(lock);
    PROBE_shmem_test_lock_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_test_lock_real != NULL)
  {
    res = shmem_test_lock_real(lock);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_test_lock was not hooked!\n");
    exit(-1);
  }
  return res;
}

void shmem_clear_cache_inv (void)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_clear_cache_inv_real at %p\n", TASKID, shmem_clear_cache_inv_real);
#endif

  if (shmem_clear_cache_inv_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_clear_cache_inv_ENTRY();
    shmem_clear_cache_inv_real();
    PROBE_shmem_clear_cache_inv_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_clear_cache_inv_real != NULL)
  {
    shmem_clear_cache_inv_real();
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_clear_cache_inv was not hooked!\n");
    exit(-1);
  }
}

void shmem_set_cache_inv (void)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_set_cache_inv_real at %p\n", TASKID, shmem_set_cache_inv_real);
#endif

  if (shmem_set_cache_inv_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_set_cache_inv_ENTRY();
    shmem_set_cache_inv_real();
    PROBE_shmem_set_cache_inv_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_set_cache_inv_real != NULL)
  {
    shmem_set_cache_inv_real();
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_set_cache_inv was not hooked!\n");
    exit(-1);
  }
}

void shmem_clear_cache_line_inv (void *target)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p\n", TASKID, target);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_clear_cache_line_inv_real at %p\n", TASKID, shmem_clear_cache_line_inv_real);
#endif

  if (shmem_clear_cache_line_inv_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_clear_cache_line_inv_ENTRY(target);
    shmem_clear_cache_line_inv_real(target);
    PROBE_shmem_clear_cache_line_inv_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_clear_cache_line_inv_real != NULL)
  {
    shmem_clear_cache_line_inv_real(target);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_clear_cache_line_inv was not hooked!\n");
    exit(-1);
  }
}

void shmem_set_cache_line_inv (void *target)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p\n", TASKID, target);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_set_cache_line_inv_real at %p\n", TASKID, shmem_set_cache_line_inv_real);
#endif

  if (shmem_set_cache_line_inv_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_set_cache_line_inv_ENTRY(target);
    shmem_set_cache_line_inv_real(target);
    PROBE_shmem_set_cache_line_inv_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_set_cache_line_inv_real != NULL)
  {
    shmem_set_cache_line_inv_real(target);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_set_cache_line_inv was not hooked!\n");
    exit(-1);
  }
}

void shmem_udcflush (void)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_udcflush_real at %p\n", TASKID, shmem_udcflush_real);
#endif

  if (shmem_udcflush_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_udcflush_ENTRY();
    shmem_udcflush_real();
    PROBE_shmem_udcflush_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_udcflush_real != NULL)
  {
    shmem_udcflush_real();
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_udcflush was not hooked!\n");
    exit(-1);
  }
}

void shmem_udcflush_line (void *target)
{
#if defined(DEBUG)
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: %p\n", TASKID, target);
  fprintf(stderr, PACKAGE_NAME": DEBUG: %d: shmem_udcflush_line_real at %p\n", TASKID, shmem_udcflush_line_real);
#endif

  if (shmem_udcflush_line_real != NULL && EXTRAE_ON() && !Backend_inInstrumentation(THREADID))
  {
    Backend_Enter_Instrumentation(2);
    PROBE_shmem_udcflush_line_ENTRY(target);
    shmem_udcflush_line_real(target);
    PROBE_shmem_udcflush_line_EXIT();
    Backend_Leave_Instrumentation();
  }
  else if (shmem_udcflush_line_real != NULL)
  {
    shmem_udcflush_line_real(target);
  }
  else
  {
    fprintf(stderr, PACKAGE_NAME": Error shmem_udcflush_line was not hooked!\n");
    exit(-1);
  }
}

/****************************************\
 ***       MODULE INITIALIZATION      ***
\****************************************/

void __attribute__ ((constructor)) Extrae_OPENSHMEM_init(void)
{
  Get_OPENSHMEM_Hook_Points(0);
}

