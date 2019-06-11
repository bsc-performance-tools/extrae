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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include "hash_table.h"


/*** Prototypes ***/
static inline int xtr_hash_search (xtr_hash_t *hash, uintptr_t key, xtr_hash_cell_t **previous_out, xtr_hash_cell_t **cell_out) __attribute__((always_inline));

static inline void free_head_cell(xtr_hash_t *hash, xtr_hash_cell_t *cell) __attribute__((always_inline));
static inline void promote_collision(xtr_hash_t *hash, xtr_hash_cell_t *src, xtr_hash_cell_t *dest) __attribute__((always_inline));
static inline void free_collision_cell(xtr_hash_t *hash, xtr_hash_cell_t *previous, xtr_hash_cell_t *cell) __attribute__((always_inline));


/**
 * xtr_hash_new
 * 
 * Allocates a new hash structure.
 *
 * @param data_size Size in bytes of the data that needs to be stored along with the key. 
 * @param flags Creation flags available for the hash structure (see enum in header file).
 *
 * @return a pointer to the new hash structure.
 */
xtr_hash_t * xtr_hash_new (xtr_hash_size_t hash_size, int data_size, int flags)
{
	int i = 0;
	xtr_hash_t *hash = NULL;

	// Allocate memory for the new hash
	if ((hash = malloc(sizeof(xtr_hash_t))) == NULL)
	{
		perror("xtr_hash_new: malloc");
		exit(-1);
	}
	memset(hash, 0, sizeof(xtr_hash_t));
	//hash = xmalloc_and_zero(sizeof(xtr_hash_t));

	// Allocate memory for head and collision arrays
	hash->head_size = hash_size;
	if ((hash->head = malloc(hash->head_size * sizeof(xtr_hash_cell_t))) == NULL)
	{
		perror("xtr_hash_new: hash->head: malloc");
		exit(-1);
	}
	hash->collision_size = XTR_HASH_COLLISION_ARRAY_SIZE(hash->head_size);
	if ((hash->collision = malloc(hash->collision_size * sizeof(xtr_hash_cell_t))) == NULL)
	{
		perror("xtr_hash_new: hash->collision: malloc");
		exit(-1);
	}

	// Allocate memory for the data storage pool
	hash->data_size = data_size;
	if ((hash->data_pool = malloc(data_size * (hash->head_size + hash->collision_size))) == NULL)
	{
		perror("xtr_hash_new: hash->data_pool: malloc");
		exit(-1);
	}
	memset(hash->data_pool, 0, data_size * (hash->head_size + hash->collision_size));
	//hash->data_pool = xmalloc_and_zero(data_size * (hash->head_size + hash->collision_size));

	// Set statistics to zero
	xtr_hash_stats_reset(hash);

	// Process creation flags
	hash->flags = flags;
	if (hash->flags & XTR_HASH_LOCK)
	{
		if (pthread_rwlock_init (&hash->lock, NULL) != 0)
		{
			perror("pthread_rwlock_init");
			exit(-1);
		}
	}

	// Initialize the head array
        for (i = 0; i < hash->head_size; i ++)
        {
                hash->head[i].data = hash->data_pool + (data_size * i);
		hash->head[i].next = XTR_KEY_NOT_HASHED; // Mark this cell free
        }

	// Initialize the collision list array
        for (i = 0; i < hash->collision_size; i ++)
        {
                hash->collision[i].data = hash->data_pool + (data_size * (hash->head_size + i));
                hash->collision[i].next = &(hash->collision[i]) + 1; // Points to the next free collision cell
        }
        hash->collision[hash->collision_size-1].next = XTR_HASH_FULL;
        hash->next_free_collision_cell = &(hash->collision[0]);

        return hash;
}


/**
 * xtr_hash_free
 * 
 * Frees the memory allocated for the given 'hash'.
 * 
 * @param hash The hash to free.
 */
void xtr_hash_free(xtr_hash_t *hash)
{
	if (hash != NULL)
	{
		if (hash->data_pool != NULL)
		{
			free(hash->data_pool);
		}
		if (hash->collision != NULL)
		{
			free(hash->collision);
		}
		if (hash->head != NULL)
		{
			free(hash->head);
		}
		free(hash);
	}
}

/**
 * xtr_hash_add
 *
 * Inserts the specified 'key' into the given 'hash', storing a copy of 'data' along with the 'key'.
 *
 * @param hash Pointer to the hash where the key is inserted.
 * @param key Key to be inserted.
 * @param data If not NULL, the buffer pointed by 'data' is stored along with the 'key'.
 *
 * @return 1 if key was successfully added; 0 if hash was full.
 */
int xtr_hash_add (xtr_hash_t *hash, uintptr_t key, void *data)
{
	// Apply the hash function to find corresponding cell in the head array
	xtr_hash_cell_t *cell = XTR_HASH_GET_CELL_FOR_KEY(hash, key);
	xtr_hash_cell_t *cell_to_add = cell;

	if (hash->flags & XTR_HASH_LOCK)
	{
		pthread_rwlock_wrlock(&hash->lock);
	}

#if defined(DEBUG)
	xtr_hash_stats_update(hash, num_add);
#endif
	
	if (XTR_KEY_HASHED(cell))
	{
		// There was a previous key stored, get a free collision cell to fill
		cell_to_add = hash->next_free_collision_cell;
		if (cell_to_add == XTR_HASH_FULL) return 0;
		hash->next_free_collision_cell = cell_to_add->next;
		// Mark the hash cell with 1+ collisions
		XTR_KEY_HASHED_WITH_COLLISION(cell, cell_to_add);
#if defined(DEBUG)
		xtr_hash_stats_update(hash, num_collisions);
#endif
	}
	else
	{
		// This is the first key stored, fill the cell in the head array without collisions
		XTR_KEY_HASHED_WITHOUT_COLLISION(cell);
	}

	// Store the key and copy the data to the data storage
	cell_to_add->key = key;
	if (data != NULL)
	{
		memcpy(cell_to_add->data, data, hash->data_size);
	}

	if (hash->flags & XTR_HASH_LOCK)
	{
		pthread_rwlock_unlock(&hash->lock);
	}

	return 1;
}


/**
 * xtr_hash_search
 * 
 * Searches the head array for the specified 'key'. If the hashed key collides, 
 * the search continues through the collision list. 
 *
 * @param hash Pointer to the hash structure.
 * @param key The key to look for.
 * @param[out] previous_out Pointer to the previous cell where the key is found. 
 *                          If 'key' was found in the head array, this is set to NULL.
 * @param[out] cell_out The cell containing the key we are looking for (either in the head array or the collision list).
 *
 * @return 1 if the key was found; 0 otherwise.
 */
static int xtr_hash_search (xtr_hash_t *hash, uintptr_t key, xtr_hash_cell_t **previous_out, xtr_hash_cell_t **cell_out)
{
	xtr_hash_cell_t *previous = NULL, *cell = XTR_HASH_GET_CELL_FOR_KEY(hash, key);

	if (XTR_KEY_HASHED(cell))
	{
		// Iterate over the collision list while the hashed key is not an exact match
		do 
		{
			if (cell->key == key)
			{
				*previous_out = previous;
				*cell_out = cell;
				return 1;
			}	
			previous = cell;
			cell = cell->next;
		}
		while (XTR_KEY_HAS_COLLISION(previous));
	}

	*previous_out = *cell_out = NULL;
	return 0;
}


/** 
 * xtr_hash_query
 * 
 * Finds the specified 'key' in 'hash' and copies back the data in the structure pointed by 'data'.
 *
 * @param hash Pointer to the hash structure.
 * @param key The key to look for.
 * @param data[out] If not NULL, stored data for the given 'key' is copied back to this buffer.
 *
 * @return 1 if the key was found; 0 otherwise.
 */
int xtr_hash_query (xtr_hash_t *hash, uintptr_t key, void *data)
{
	xtr_hash_cell_t *previous = NULL, *cell = NULL;

        if (hash->flags & XTR_HASH_LOCK)
        {
                pthread_rwlock_rdlock(&hash->lock);
        }

#if defined(DEBUG)
	xtr_hash_stats_update(hash, num_query);
#endif

	// Search for the specified key 
	if (xtr_hash_search (hash, key, &previous, &cell))
	{
		// Copy data back to the user
		if (data != NULL)
		{
			memcpy(data, cell->data, hash->data_size);
		}
		return 1;
	}

        if (hash->flags & XTR_HASH_LOCK)
        {
                pthread_rwlock_unlock(&hash->lock);
        }

	return 0;
}


/**
 * xtr_hash_fetch
 *
 * Removes the specified 'key' in 'hash' and copies back the data in the structure pointed by 'data'.
 *
 * @param hash Pointer to the hash structure.
 * @param key The key to remove from the hash.
 * @param data[out] If not NULL, stored data for the given 'key' is copied back to this buffer.
 *
 * @return 1 if the key was found; 0 otherwise.
 */
int xtr_hash_fetch (xtr_hash_t * hash, uintptr_t key, void *data)
{
	int found = 0;
	xtr_hash_cell_t *previous = NULL, *cell = NULL;

        if (hash->flags & XTR_HASH_LOCK)
        {
                pthread_rwlock_wrlock(&hash->lock);
        }

#if defined(DEBUG)
	xtr_hash_stats_update(hash, num_fetch);
#endif

        // Search for the specified key 
	if (xtr_hash_search (hash, key, &previous, &cell))
	{
		found = 1;

		// Copy data back to the user
		if (data != NULL)
		{
			memcpy(data, cell->data, hash->data_size);
		}

		// Remove the key 
		if (previous == NULL)
		{
			// Remove it from the head array
			free_head_cell(hash, cell);
		}
		else
		{
			// Remove it from the collision list
			free_collision_cell(hash, previous, cell);
		}
	}

        if (hash->flags & XTR_HASH_LOCK)
        {
                pthread_rwlock_unlock(&hash->lock);
        }

	return found;
}


/**
 * free_head_cell
 * 
 * Frees the given 'cell' in the head array.
 *
 * @param hash Pointer to the hash structure.
 * @param cell Pointer to the cell in the head array to free.
 */
static void free_head_cell(xtr_hash_t *hash, xtr_hash_cell_t *cell)
{
	if (XTR_KEY_HAS_COLLISION(cell))
	{
                // But there are collisions, promote 1st collision to head array.
		promote_collision(hash, cell->next, cell);
	}
	else
	{
                // And there are no collisions, just mark the head array as empty
		cell->next = XTR_KEY_NOT_HASHED;
	}
}


/**
 * promote_collision
 * 
 * The data stored in collision cell 'src' is moved to the head cell 'dest'. 
 * The collision cell 'src' is then free'd.
 *
 * BEFORE: 
 *                           +--------------------------
 *                      Head |  |  | X |  |  |  |  | ...
 *                           +--------------------------
 *                                  ||
 *         data +-------------------+| next
 *              |                    v
 *              |                  +---------
 *              |                  | O | ...  Collision
 *              |                  +---------
 *              |                    | data
 * data_pool    v                    v     
 *         ↘ +------------------------------------------+
 *           | h1 | h2 | ...       | c1 | c2 | ...      | 
 *           +------------------------------------------+
 * 
 * AFTER:
 *                            +--------------------------
 *                       Head |  |  | O |  |  |  |  | ...
 *                            +--------------------------
 *                                   | 
 *                                   | +--- next_free_collision_cell
 *                             +-----+ |
 *                        data |       v
 *                             |    +---------
 *                             |    |   | ...  Collision
 *                             |    +---------
 *                ______________\_____| data
 *               |               \_____
 *  data_pool    v                    v      
 *          ↘ +------------------------------------------+
 *            |    | h2 | ...       | h1 | c2 | ...      | 
 *            +------------------------------------------+
 * 
 * @param hash Pointer to the hash structure.
 * @param src Pointer to the cell in the collision list to move to the head array.
 * @param dest Pointer to the cell in the head array where 'src' will be moved to.
 */
static void promote_collision(xtr_hash_t *hash, xtr_hash_cell_t *src, xtr_hash_cell_t *dest) 
{
        void *free_data = dest->data;

	// Copy the key and move the data to the new location in the hash data pool
	dest->key  = src->key;

        // Swap data pointers to avoid a more costly memcpy (see diagram)
#if 0
	memcpy(src->data, dest->data, hash->data_size);
#else
        dest->data = src->data; 
        src->data  = free_data;
#endif

	// Free the collision cell
	free_collision_cell(hash, dest, src);
}


/**
 * free_collision_cell
 * 
 * Frees the given 'cell' in the collision array.
 * 
 * @param hash Pointer to the hash structure.
 * @param previous Pointer to the previous collision in the list of the cell to free.
 * @param cell Pointer to the cell in the collision list to free.
 */
static void free_collision_cell(xtr_hash_t *hash, xtr_hash_cell_t *previous, xtr_hash_cell_t *cell)
{
	// Make previous collision point to the next
	BYPASS_CELL(previous, cell);

	// Return the collision cell to the free list
	cell->next = hash->next_free_collision_cell;
	hash->next_free_collision_cell = cell;
}


/** 
 * xtr_hash_dump
 *
 * Dumps the contents of the hash table.
 * 
 * @param hash Pointer to the hash to dump.
 * @param pretty_print_func_ptr Callback pointer to a function that knows how to dump the data stored in this hash.  
 */
void xtr_hash_dump(xtr_hash_t *hash, void *pretty_print_func_ptr)
{
	int i = 0;
	void (*pretty_print)(void *) = pretty_print_func_ptr;

	// Iterate over all hash cells
	for (i = 0; i < hash->head_size; i ++)
	{
		xtr_hash_cell_t *cell = XTR_HASH_GET_CELL_FOR_INDEX(hash, i);

		if (XTR_KEY_HASHED(cell))
		{
			// Dump the key hashed in the current cell
			int num_collisions = 0;

			fprintf(stderr, "xtr_hash_dump: Index #%d: key=%lu collisions?=%s ", i, cell->key, XTR_KEY_HAS_COLLISION(cell) ? "yes" : "no" );
			if (pretty_print != NULL) pretty_print(cell->data);
			fprintf(stderr, "\n");

			// Iterate over all collisions of this hashed key
			while (XTR_KEY_HAS_COLLISION(cell))
			{
				cell = cell->next;
				num_collisions ++;
				fprintf(stderr, "xtr_hash_dump:       ↳ collision #%d: key=%lu ", num_collisions, cell->key);
				if (pretty_print != NULL) pretty_print(cell->data);
				fprintf(stderr, "\n");
			} 
		}
	}
}


/**
 * xtr_hash_stats_reset
 *
 * @param hash Pointer to the hash to print statistics for.
 */
void xtr_hash_stats_reset(xtr_hash_t *hash)
{
	hash->stats.num_add        = 0;
	hash->stats.num_collisions = 0;
	hash->stats.num_query      = 0;
	hash->stats.num_fetch      = 0;
	hash->stats.leftovers      = 0;
}


/**
 * xtr_hash_stats_dump
 *
 * Prints usage statistics for the given 'hash'
 *
 * @param hash Pointer to the hash to print statistics for.
 */
void xtr_hash_stats_dump (xtr_hash_t *hash)
{
	int i = 0;
	int count_free_in_collision = 0;
	int leftovers = 0;
	
	leftovers = 0;

	// Iterate over all hash cells
	for (i = 0; i < hash->head_size; i ++)
	{
		xtr_hash_cell_t *cell = XTR_HASH_GET_CELL_FOR_INDEX(hash, i);

		if (XTR_KEY_HASHED(cell))
		{
			// Count non-free cells in head array
			leftovers ++;
		}
	}
	
	// Count free collision cells
	xtr_hash_cell_t *collision = hash->next_free_collision_cell;
	while (collision != XTR_HASH_FULL)
	{
		count_free_in_collision ++;
		collision = collision->next;
	}

	// Add used collision cells to the count
	leftovers += (hash->collision_size - count_free_in_collision);
	xtr_hash_stats_update(hash, leftovers);

	fprintf(stderr, "xtr_hash_stats: Adds=%d\n",       hash->stats.num_add);
	fprintf(stderr, "xtr_hash_stats: Queries=%d\n",    hash->stats.num_query);
	fprintf(stderr, "xtr_hash_stats: Fetches=%d\n",    hash->stats.num_fetch);
	fprintf(stderr, "xtr_hash_stats: Collisions=%d (%.2lf%%)\n", hash->stats.num_collisions, 
		(hash->stats.num_add == 0 ? 0 : ((double)hash->stats.num_collisions / (double)hash->stats.num_add) * 100));
	fprintf(stderr, "xtr_hash_stats: Leftovers=%d (%.2lf%%)\n", hash->stats.leftovers, 
	        ((double)hash->stats.leftovers / (double)(hash->head_size + hash->collision_size)) * 100 );
}


