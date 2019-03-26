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

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STDINT_H
# include <stdint.h>
#endif
#if defined(MPI_HAS_INIT_THREAD_C) || defined(MPI_HAS_INIT_THREAD_F)
# ifdef HAVE_PTHREAD_H
#  include <pthread.h>
# endif
#endif
#include "hash_table.h"

#if !defined(MPI_SUPPORT) /* This shouldn't be compiled if MPI is not used */
# error "This should not be compiled outside MPI bounds"
#endif

#if defined(MPI_HAS_INIT_THREAD_C) || defined(MPI_HAS_INIT_THREAD_F)
pthread_mutex_t hash_lock;
#endif

/*
 *   Initializes hashing table.
 */
void xtr_hash_init (xtr_hash_t * hash)
{
  int i;

#if defined(MPI_HAS_INIT_THREAD_C) || defined(MPI_HAS_INIT_THREAD_F)
  if (pthread_mutex_init(&hash_lock, NULL) != 0)
  {
    fprintf (stderr, PACKAGE_NAME": xtr_hash_init: Mutex initialization failed.\n");
    exit(-1);
  }
#endif

  for (i = 0; i < XTR_HASH_TABLE_SIZE; i++)
    hash->table[i].ovf_link = XTR_HASH_FREE;

  /*
   * All overflow cells are free 
   */
  for (i = 0; i < XTR_HASH_OVERFLOW_SIZE - 1; i++)
    hash->overflow[i].next = i + 1;
  /*
   * Last in the list has different 'next' 
   */
  hash->overflow[XTR_HASH_OVERFLOW_SIZE - 1].next = XTR_HASH_NULL;
  hash->ovf_free = 0;
}

/*
 *   Adds a new hash entry to the hash table. Returns != 0 when there is no
 *  space left in the table.
 */
int xtr_hash_add (xtr_hash_t * hash, const xtr_hash_data_t * data)
{
  int cell, free;
  int rc = FALSE;

#if defined(MPI_HAS_INIT_THREAD_C) || defined(MPI_HAS_INIT_THREAD_F)
  pthread_mutex_lock(&hash_lock);
#endif

  cell = XTR_HASH_FUNCTION (data->key);

  if (hash->table[cell].ovf_link == XTR_HASH_FREE)
  {
    hash->table[cell].ovf_link = XTR_HASH_NULL;     /* No longer free */
    hash->table[cell].data = *data;
  }
  else
  {
    /*
     * Get a free overflow cell 
     */
    if ((free = hash->ovf_free) == XTR_HASH_NULL)
    {
      fprintf (stderr, PACKAGE_NAME": xtr_hash_add: No space left in hash table. Size is %d+%d\n", XTR_HASH_TABLE_SIZE, XTR_HASH_OVERFLOW_SIZE);
      rc = TRUE;
    }
    else
    {
      hash->ovf_free = hash->overflow[free].next;

      /*
       * Insert free into overflow list of cell 
       */
      hash->overflow[free].next = hash->table[cell].ovf_link;
      hash->table[cell].ovf_link = free;
      /*
       * Update user data 
       */
      hash->overflow[free].data = *data;
    }
  }

#if defined(MPI_HAS_INIT_THREAD_C) || defined(MPI_HAS_INIT_THREAD_F)
  pthread_mutex_unlock(&hash_lock);
#endif

  return rc;
}


/*
 *   Searches for key in the hash table. Returns NULL when the key is not
 *  found in the table.
 */
xtr_hash_data_t *xtr_hash_search (const xtr_hash_t * hash, UINT64 key)
{
  int cell, ovf;

  cell = XTR_HASH_FUNCTION (key);

  if (hash->table[cell].ovf_link == XTR_HASH_FREE)
    return NULL;

  if (hash->table[cell].data.key == key)
    return (xtr_hash_data_t *) & hash->table[cell].data;

  /*
   * Look for key in overflow list 
   */
  ovf = hash->table[cell].ovf_link;
  while (ovf != XTR_HASH_NULL)
  {
    if (hash->overflow[ovf].data.key == key)
      return (xtr_hash_data_t *) & hash->overflow[ovf].data;
    ovf = hash->overflow[ovf].next;
  }
  return NULL;
}


/*
 *   Removes entry key in the hash table. Returns != 0 when key is not found in
 *  the table.
 */
int xtr_hash_remove (xtr_hash_t * hash, UINT64 key)
{
  int cell, ovf, prev;
  int rc = FALSE;

#if defined(MPI_HAS_INIT_THREAD_C) || defined(MPI_HAS_INIT_THREAD_F)
  pthread_mutex_lock(&hash_lock);
#endif

  cell = XTR_HASH_FUNCTION (key);

  if (hash->table[cell].ovf_link == XTR_HASH_FREE)
  {
#if SIZEOF_LONG == 8
    fprintf (stderr, PACKAGE_NAME": xtr_hash_remove: Key %08lx not in hash table\n", (long) key);
#elif SIZEOF_LONG == 4
    fprintf (stderr, PACKAGE_NAME": xtr_hash_remove: Key %04x not in hash table\n", (long) key);
#endif
    rc = TRUE;
  }
  else if (hash->table[cell].data.key == key)
  {
    /*
     * Remove the main entry 
     */
    if ((ovf = hash->table[cell].ovf_link) != XTR_HASH_NULL)
    {
      /*
       * Bring 1st overflow to main entry 
       */
      hash->table[cell].data = hash->overflow[ovf].data;
      hash->table[cell].ovf_link = hash->overflow[ovf].next;
      /*
       * Put freed ovf in free list 
       */
      hash->overflow[ovf].next = hash->ovf_free;
      hash->ovf_free = ovf;
    }
    else
    {
      hash->table[cell].ovf_link = XTR_HASH_FREE;   /* Mark as free */
    }
  }
  else
  {
    /*
     * Search key in overflow list 
     */
    ovf = hash->table[cell].ovf_link;
    prev = XTR_HASH_NULL;
    while (ovf != XTR_HASH_NULL && hash->overflow[ovf].data.key != key)
    {
      prev = ovf;
      ovf = hash->overflow[ovf].next;
    }
    if (ovf == XTR_HASH_NULL)       /* Not found */
    {
#if SIZEOF_LONG == 8
      fprintf (stderr, PACKAGE_NAME": xtr_hash_remove: Key %08lx not in hash table\n", (long) key);
#elif SIZEOF_LONG == 4
      fprintf (stderr, PACKAGE_NAME": xtr_hash_remove: Key %04x not in hash table\n", (long) key);
#endif
      rc = TRUE;
    }
    else
    {
      if (prev == XTR_HASH_NULL)
      {
        hash->table[cell].ovf_link = hash->overflow[ovf].next;
      }
      else
      {
        hash->overflow[prev].next = hash->overflow[ovf].next;
      }
      hash->overflow[ovf].next = hash->ovf_free;
      hash->ovf_free = ovf;
    }
  }

#if defined(MPI_HAS_INIT_THREAD_C) || defined(MPI_HAS_INIT_THREAD_F)
  pthread_mutex_unlock(&hash_lock);
#endif

  return rc;
}

