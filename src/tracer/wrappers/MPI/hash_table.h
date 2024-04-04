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

#include <pthread.h>
#ifndef UNIT_TEST
# include "common.h"
#else
typedef unsigned long UINT64;
#endif /* UNIT_TEST */

/***

  This data structure represents a hash table with collision resolution by chaining. Main design features are:
  - The hash function is a simple module operation of the key by the hash size. 
  - The hash elements are stored in a memory pool that is dynamically allocated once, and the update of the elements is done through a linked list.
  - The data associated with the keys is stored in a separate memory pool of configurable size.
  - The hash table can be locked for multithread access.
  - The hash table can be configured to allow duplicate keys.
  - The hash table can be configured to automatically expand when full.
  - TODO: The hash table can be configured to automatically shrink when empty.
  - The elements are stored in LIFO order, so the last element added is the first to be removed.
  - TODO: The hash table can be configured to store the elements in FIFO order.
  - The hash table can be configured to return the data associated with the key when searching, removing or adding duplicates. 

key % hash_size = bucket array that points to a linked list of elements with the same index

buckets          pool_array[0].item_pool
   \                                    \ 
    [ Index_0 ]                          \ _______________  
    [ Index_1 ] ------------------------> |    |    |     |
    [ ...     ]                           | i1 | i2 | ... | i1, i2, ..., iN are chained through next pointer when they fall in the same bucket
                                          |____|____|_____| 
                                         /|    |    |     | 
                                        / |____|____|_____|  
                         pool_first_free  |    |    |     | pool_first_free points to the first free element in (any of) the item pool(s)
                                          |____|____|_____|
                                          |    |    |     | 
                                          |    |    | iN  | N is the hash size. 
                                          |____|____|_____| When iN is reached, pool_array grows with a new pair of item and data pools

                  pool_array[0].data_pool                   data_pool matches the item_pool in size, and stores the data associated with the keys
                                         \                  so i1 points to d1, i2 points to d2, ..., iN points to dN
                                          \_______________ 
                                          |    |    |     |
                                          | d1 | d2 | ... | 
                                          |____|____|_____| 
                                          |    |    |     | 
                                          |____|____|_____|  
                                          |    |    |     | 
                                          |____|____|_____|
                                          |    |    |     | 
                                          |    |    | dN  | 
                                          |____|____|_____|
 

***/

/*
 * Some prime numbers for the hash size
 *
 *         5       11       19       31      127
 *       211      383      631      887     1151
 *      1399     1663     1913     2161     2423
 *      2687     2939     3191     3449     3709
 *      3967     4219     4463     4733     4987
 *      5237     5503     5749     6011     6271
 *      6521     6781     7039     7283     7549
 *      7793     8059     8317     8573     8831
 *      9067     9343     9587     9851    10111
 *     10357    10613    10867    11131    11383
 *     11633    11903    12157    12413    12671
 *     12923    13183    13421    13693    13933
 *     14207    14461    14717    14969    15227
 *     15473    15739    15991    16253    16493
 *     18553    20599    22651    24697    26737
 *     28793    30841    32887    34939    36979
 *     39023    41081    43133    45181    47221
 *     49279    51307    53359    55411    57467
 *     59513    61561    63611    65657    82039
 *     98429   114809   131171   147583   163927
 *    180347   196727   213119   229499   245881
 *    262271   327799   393331   458879   524413
 *    589933   655471   721013   786553   852079
 *    917629   983153
 */
typedef enum {
  XTR_HASH_SIZE_TINY = 55411,
  XTR_HASH_SIZE_SMALL = 114809,
  XTR_HASH_SIZE_MEDIUM = 229499,
  XTR_HASH_SIZE_LARGE = 458879,
  XTR_HASH_SIZE_XLARGE = 917629
} xtr_hash_size;

// Hash creation flags
enum {
  XTR_HASH_NONE = 0,
  XTR_HASH_LOCK = 1 << 0,             // Lock hash for multithread access
  XTR_HASH_ALLOW_DUPLICATES = 1 << 1, // Allow duplicate keys
  XTR_HASH_AUTO_EXPAND = 1 << 2       // Hash automatically expands when full
};


/**
 * xtr_hash_stats
 *
 * Holds usage statistics for debugging purposes.
 * 
 * 'num_adds' counts the number of calls to xtr_hash_add() that successfully insert the element. 
 * 'num_overflows' counts the number of calls to xtr_hash_add() that can't insert the element because hash is full. 
 * 'num_hits' counts the number of calls to xtr_hash_search()/_remove() that find the key.
 * 'num_misses' counts the number of calls to xtr_hash_search()/_remove() that don't find the key. 
 * 'num_deletes' counts the number of calls to xtr_hash_remove() that succesfully delete an element. 
 * 'num_collisions' counts how many additions provoked a collision. 
 * 'num_duplicates' counts how many additions provoked a duplicate.
 */
typedef struct xtr_hash_stats {
  int num_adds;
  int num_overflows;
  int num_hits;
  int num_misses;
  int num_deletes;
  int num_collisions;
  int num_duplicates;
} xtr_hash_stats;

#define xtr_hash_stats_update(hash, metric) \
  { hash->stats.metric++; }


/**
 * xtr_hash_item
 *
 * This structure holds a (key, data) element in the hash.
 * 
 * 'key' is the stored element identifier.
 * 'data' points to the data pool where the data is stored.
 * 'next' points to the next collision in the hash, or NULL if there aren't
 */
typedef struct xtr_hash_item {
  UINT64 key;
  void *data;
  struct xtr_hash_item *next;
} xtr_hash_item;


/**
 * xtr_hash_pool
 * 
 * This structure holds the memory pools for the hash. 
 * Each pool is a dynamically allocated memory region that preallocates all the possible hash elements as a 
 * linked list of items, and a data region to store the data associated with the keys.
 * 
 * 'item_pool' points to the memory pool for the hash items.
 * 'data_pool' points to the memory pool for the data.
 */
typedef struct xtr_hash_pool {
  xtr_hash_item *item_pool;
  void *data_pool;
} xtr_hash_pool;


/**
 * xtr_hash_t
 *
 * This structure holds the hash table.
 * 
 * 'num_buckets' is the size of the hash.
 * 'buckets' points to the array indexed by the hash function to store elements.
 * 'pool_size' is the maximum number of elements that the hash can hold.
 * 'data_size' is the size of the data stored along with the keys.
 * 'num_pools' is the number of memory pools allocated for the hash.
 * 'pool_array' points to an array of memory pools.
 * 'pool_first_free' points to the first free element available in the hash.
 * 'flags' can be set for different creation options.
 * 'lock' holds a mutex activated by flag XTR_HASH_LOCK for safe multithread access. 
 * 'stats' holds usage statistics for this hash.
 */
typedef struct xtr_hash {
  int num_buckets;
  xtr_hash_item **buckets;

  int pool_size;
  int data_size;
  int num_pools;
  xtr_hash_pool *pool_array;
  xtr_hash_item *pool_first_free;

  int flags;
  pthread_rwlock_t lock;
  xtr_hash_stats stats;
} xtr_hash;


// Public functions
xtr_hash *xtr_hash_new(xtr_hash_size hash_size, int data_size, int flags);
void xtr_hash_free(xtr_hash *hash);
int xtr_hash_add(xtr_hash *hash, UINT64 key, void *data_in, void *data_out);
int xtr_hash_remove(xtr_hash *hash, UINT64 key, void *data);
int xtr_hash_search(xtr_hash *hash, UINT64 key, void *data);
void xtr_hash_dump(xtr_hash *hash, void *pretty_print_func_ptr);
void xtr_hash_stats_reset(xtr_hash *hash);
void xtr_hash_stats_dump(xtr_hash *hash);
