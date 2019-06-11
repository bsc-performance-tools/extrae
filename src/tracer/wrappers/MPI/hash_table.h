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
#include <stdio.h>
#include <stdint.h>
#include <mpi.h>

/***

  This data structure represents a hash table with shared overflow. Main design details follow:

  - The first key hashed is stored in the corresponding hash index of the 'Head' array.
  - Empty cells in the head array are marked with XTR_KEY_NOT_HASHED (NULL).
  - Further keys that collide are stored in the 'Collision' list.
  - Each head cell points to its own collision list. 
  - Starting from the head cell, you can iterate all collisions through pointer 'next', as a linked list. 
  - Last element in the collision list points to itself. 
  - Free cells in the collision list are pointed by 'next_free_collision_cell' and linked through the 'next' pointer.
  - Last free cell points to XTR_HASH_FULL (NULL).
  - Collisions are added in LIFO order, i.e. becomes 1st in the collision list (head cell points to the newest collision)
  - When head cells are free'd, first collision gets promoted to the head array.
  - Each cell has a reserved spot in the data_pool to store user data.
  - Initially the data_pool storage is assigned to the head cells (h1, h2...) , then to the collision cells (c1, c2...), in order.
  - If freeing a head cell causes a promotion (move 1st collision to head), the data pointers get switched to avoid memcpy's (see promote_collision)


                  XTR_HASH_FUNCTION(MPI_Request)      +--> XTR_KEY_NOT_HASHED
                          ↘         ↘         ↘       |
                         +------------------------------------
              Head array | S4 |   | R1 |   | T1 |   | X | ...
                         +------------------------------------
                           |         |         ↺         
                           |         |   
                     next  +----+    |   +--------- next_free_collision_cell
                                |    |   |
                                v    v   v
                    +---------------------------+
     Collision list | R2 | R3 | S5 | R6 |   |   |
                    +---------------------------+
                    ↻ ^   |  ^  ↺   |     |  ^ |
                      |   |  |      |     |  | |
                next  +---+  +------+     +--+ +--> XTR_HASH_FULL
                             

                           +--------------------------
                      Head |  |  |  |  |  |  |  | ...
                           +--------------------------
                            |
                            |                   +-----------------------
                     data   |         Collision |  |  |  |  |  |  | ...
              ______________|                   +-----------------------
 data_pool   ↓                             data  ↓
         ↘ +------------------------------------------------------------------------------------+
           | h1 | h2 | ...                         | c1 | c2 | ...                              | 
           +------------------------------------------------------------------------------------+
           <-----    data_size * head_size     ----><-----    data_size * collision_size    -----> 

***/


/*** Defines ***/

// Mark cells in heads array as empty
#define XTR_KEY_NOT_HASHED NULL 

// Mark free list as full
#define XTR_HASH_FULL      NULL

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
typedef enum
{
  XTR_HASH_SIZE_TINY   = 55411,
  XTR_HASH_SIZE_SMALL  = 114809,
  XTR_HASH_SIZE_MEDIUM = 229499,
  XTR_HASH_SIZE_LARGE  = 458879,
  XTR_HASH_SIZE_XLARGE = 917629
} xtr_hash_size_t;

#define XTR_HASH_COLLISION_ARRAY_SIZE(head_array_size) ((head_array_size*15)/100)

// Hash creation flags 
enum
{
  XTR_HASH_NONE = 0,
  XTR_HASH_LOCK = 1 << 0
};


/*** Types, structures & macros ***/

/**
 * xtr_hash_cell_t
 * 
 * This structure holds a (key, data) element in the hash.
 * 'key' is the stored element identifier.
 * 'data' points to the hash data pool where the data is stored.
 * 'next' can have multiple values:
 *   (a) When the cell belongs to the head array:
 *       - XTR_KEY_NOT_HASHED (NULL) indicates the cell is empty
 *       - Points to itself when there are no collisions.
 *       - Points to the next cell in the collision list when there are 1+ collisions.
 *   (b) When the cell belongs to the collision list:
 *       - Points to itself when this is the last collision.
 *       - Points to the next cell in the collision list if there are more collisions.
 *   (c) When the cell belongs to the free list:
 *       - XTR_HASH_FULL (NULL) indicating there are no more free cells.
 *       - Points to the next free cell in the collision list.
 */
typedef struct xtr_hash_cell_t xtr_hash_cell_t;

struct xtr_hash_cell_t
{
 	uintptr_t        key;
	void            *data;
	xtr_hash_cell_t *next;
};

/**
 * xtr_hash_stats_t
 * 
 * Holds usage statistics for debugging purposes.
 * 'num_add' counts the number of calls to xtr_hash_add().
 * 'num_query' counts the number of calls to xtr_hash_query().
 * 'num_fetch' counts the number of calls to xtr_hash_fetch().
 * 'num_collisions' counts how many additions provoked a collision.
 * 'leftovers' counts how many elements are currently in the hash
 *             when xtr_hash_stats_dump() is called.
 */
typedef struct xtr_hash_stats_t
{
	int num_add;
	int num_query;
	int num_fetch;
	int num_collisions;
	int leftovers;
} xtr_hash_stats_t;

/**
 * xtr_hash_t
 * 
 * This structure represents a hash container with shared overflow 
 * and dynamic data storage.
 * 'head' points to the array indexed by the hash function to store elements.
 * 'collision' points to the array where collisions are stored when the 
 *             corresponding 'head' cells are already in use.
 * 'data_size' is the size of the data stored along with the keys.
 * 'data_pool' points to a dynamically allocated memory region where
 *             the data of each cell (either from 'head' or 'collision')
 * 'next_free_collision_cell' points to the first free cell in the collision list.
 * 'flags' can be set for different creation options.
 * 'lock' holds a mutex activated by flag XTR_HASH_LOCK for safe multithread access.
 * 'stats' holds usage statistics for this hash.
 */
typedef struct xtr_hash_t
{
	int              head_size;
	xtr_hash_cell_t *head;
	int              collision_size;
	xtr_hash_cell_t *collision;
	int              data_size;
	void            *data_pool;
	xtr_hash_cell_t *next_free_collision_cell;
	int              flags;
	pthread_rwlock_t lock;
	xtr_hash_stats_t stats;
} xtr_hash_t;

// Applies the hash function to the given key
#define XTR_HASH_FUNCTION(hash, key) (((uintptr_t)(key)) % hash->head_size)

// Return the corresponding cell for the given key (in the head array)
#define XTR_HASH_GET_CELL_FOR_KEY(hash, key) &(hash->head[ XTR_HASH_FUNCTION(hash, key) ])

// Return the corresponding cell for the given hash index (in the head array)
#define XTR_HASH_GET_CELL_FOR_INDEX(hash, i) &(hash->head[ i ]) 

// Check whether there's a key already stored in the given cell (in the head array)
#define XTR_KEY_HASHED(cell) (cell->next != XTR_KEY_NOT_HASHED)

/**
 * BYPASS_CELL
 *
 * Bypass 'current' cell in the collision list by making 'previous' point to 'next'.
 * Special case: 'current' pointing to itself means this is the last element in the collision list.
 *               In this case, 'previous' becomes the new last element.
 */
#define BYPASS_CELL(previous, current) previous->next = (current->next == current ? previous : current->next)

/** 
 * XTR_KEY_HASHED_WITH_COLLISION 
 *
 * Adds a new 'collision' to the given 'head' cell in LIFO order. 
 * To keep LIFO pointers update as follows:
 * - New 'collision' cell points to the last previous collision (if there was).
 * - Corresponding 'head' cell points to the newest 'collision'.
 */
#define XTR_KEY_HASHED_WITH_COLLISION(head, collision) BYPASS_CELL(collision, head); head->next = collision;

/**
 * XTR_KEY_HASHED_WITHOUT_COLLISION 
 *
 * Marks given 'head' array cell as storing 1 key without collisions (pointer to itself indicates no more collisions)
 */
#define XTR_KEY_HASHED_WITHOUT_COLLISION(cell) cell->next = cell;

// Check whether there are more collisions (pointer to itself indicates last collision; NULL either XTR_HASH_FULL or XTR_KEY_NOT_HASHED) 
#define XTR_KEY_HAS_COLLISION(cell) ((cell->next != cell) && (cell->next != NULL))

// Atomic increment of the specified statistic
#define xtr_hash_stats_update(hash, metric)  __sync_fetch_and_add(&(hash->stats.metric), 1);


/*** Prototypes ***/

xtr_hash_t * xtr_hash_new (xtr_hash_size_t hash_size, int data_size, int flags);
void xtr_hash_free(xtr_hash_t *hash);
int xtr_hash_add (xtr_hash_t *hash, uintptr_t key, void *data);
int xtr_hash_query (xtr_hash_t *hash, uintptr_t key, void *data);
int xtr_hash_fetch (xtr_hash_t * hash, uintptr_t key, void *data);
void xtr_hash_dump(xtr_hash_t *hash, void *pretty_print_func_ptr);

void xtr_hash_stats_reset(xtr_hash_t *hash);
void xtr_hash_stats_dump (xtr_hash_t *hash);

