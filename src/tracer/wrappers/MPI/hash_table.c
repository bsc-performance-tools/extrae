#include "hash_table.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef UNIT_TEST
# include "xalloc.h"
#else
# define xfree            free
# define xrealloc         realloc
# define xmalloc          malloc
# define xmalloc_and_zero malloc
#endif /* UNIT_TEST */

//#define DEBUG


/*
 * xtr_hash_function
 * 
 * Calculates the hash function for the given key
 */
#define xtr_hash_function(hash, key) (key % hash->num_buckets)


/*
 * allocate_new_pool
 * 
 * Allocates a new pool of elements for the given hash.
 * @param hash Pointer to the hash structure.
 */
static void allocate_new_pool(xtr_hash *hash)
{
    int i = 0;
    int p = hash->num_pools ++;

    // Allocate memory for the new pool of elements
    hash->pool_array = xrealloc(hash->pool_array, sizeof(xtr_hash_pool) * hash->num_pools);

    // Allocate memory for the item pool
    hash->pool_array[p].item_pool = xmalloc_and_zero(hash->pool_size * sizeof(xtr_hash_item));

    // Allocate memory for the data storage pool
    hash->pool_array[p].data_pool = xmalloc_and_zero(hash->pool_size * hash->data_size);

    // Initialize the item pool    
    for (i = 0; i < hash->pool_size; i++)
    {
        // Link the item to the data pool
        hash->pool_array[p].item_pool[i].data = hash->pool_array[p].data_pool + (i * hash->data_size);

        // Link the current item to the next free item
        if (i < hash->pool_size - 1) {
            hash->pool_array[p].item_pool[i].next = &(hash->pool_array[p].item_pool[i + 1]);
    	}
        else {
            hash->pool_array[p].item_pool[i].next = NULL;
	    }
    }

    // Link the new pool to the free list 
    hash->pool_first_free = hash->pool_array[p].item_pool;
}


/*
 * xtr_hash_new
 * 
 * Creates a new hash table with the specified size, data size and flags.
 * @param hash_size Size of the hash table.
 * @param data_size Size of the data to be stored along with the keys.
 * @param flags Creation flags.
 * 
 * @return Pointer to the newly created hash table.
 */
xtr_hash *xtr_hash_new(xtr_hash_size hash_size, int data_size, int flags)
{
    int i = 0;

    // Allocate memory for the new hash
    xtr_hash *hash = xmalloc_and_zero(sizeof(xtr_hash));

    // Allocate memory the hash index
    hash->num_buckets = hash_size;
    hash->buckets = xmalloc_and_zero(hash_size * sizeof(xtr_hash_item *));
    for (i = 0; i < (int)hash_size; i++)
    {
        hash->buckets[i] = NULL;
    }

    // Allocate memory for the first pool of elements
    hash->pool_size = hash_size;
    hash->data_size = data_size;
    hash->num_pools = 0;
    allocate_new_pool(hash);

    // Set flags
    hash->flags = flags;
    if (hash->flags & XTR_HASH_LOCK)
    {
        if (pthread_rwlock_init(&hash->lock, NULL) != 0)
        {
            perror("pthread_rwlock_init");
            exit(-1);
        }
    }

    // Set statistics to zero
    xtr_hash_stats_reset(hash);

    return hash;
}


/*
 * xtr_hash_free
 * 
 * Frees the memory allocated for the given hash table.
 * @param hash Pointer to the hash table to be freed.
 */
void xtr_hash_free(xtr_hash *hash)
{
    int i = 0;

    for (i = 0; i < hash->num_pools; i++)
    {
      xfree(hash->pool_array[i].data_pool);
      xfree(hash->pool_array[i].item_pool);
    }
    xfree(hash->pool_array);
    xfree(hash->buckets);
    xfree(hash);
}


/**
 * xtr_hash_add
 *
 * Adds a new (key, data) pair to the hash table.
 * If the key already exists, and flag XTR_HASH_ALLOW_DUPLICATES is not set, the old key is replaced and the old data is copied back to the user in 'data_out'. 
 * If the hash is full, and flag XTR_HASH_AUTO_EXPAND is set, the hash is automatically expanded to hold more elements.
 * 
 * @param hash Pointer to the hash structure.
 * @param key The key to add to the hash.
 * @param data_in If not NULL, data to be stored in the hash.
 * @param data_out [out] If not NULL, and the insertion replaces a duplicate key, receives a copy of the replaced key data.
 * 
 * @return 1 if key was successfully added; >1 if the addition results in a duplicate key replaced, in which case data_out is updated; 0 if hash was full. 
 */
int xtr_hash_add(xtr_hash *hash, UINT64 key, void *data_in, void *data_out)
{
    int added = 0;
    int index = -1;

    if (hash->flags & XTR_HASH_LOCK)
    {
        pthread_rwlock_wrlock(&hash->lock);
    }

    // Calculate the hash index
    index = xtr_hash_function(hash, key);

#if defined(DEBUG)
    if (hash->buckets[index] != NULL)
    {
        xtr_hash_stats_update(hash, num_collisions);
    }
#endif

    // Look in the hash index through the collision list for the matching key
    if ( !(hash->flags & XTR_HASH_ALLOW_DUPLICATES) )
    {
        xtr_hash_item *it = hash->buckets[index];
        xtr_hash_item *prev = NULL;
        while ((it != NULL) && !added)
        {
            // Look for a duplicate in the collision list
            if (it->key == key)
            {
#if defined(DEBUG)
                xtr_hash_stats_update(hash, num_duplicates);
#endif
                // Copy back the old duplicate's data
                if (data_out != NULL)
                {
                    memcpy(data_out, it->data, hash->data_size);
                }
                // Save the new duplicate's data
                if (data_in != NULL)
                {
                    memcpy(it->data, data_in, hash->data_size);
                }
                // Promote the duplicate as the most recent element in the collision list
                if (prev != NULL)
                {
                    prev->next = it->next;
                    it->next = hash->buckets[index];
                    hash->buckets[index] = it;
                }
                added = 2;
            }
            prev = it;
            it = it->next;
        }
    }

    // If there was no duplicate, and the hash is full, allocate a new pool    
    if (!added && (hash->pool_first_free == NULL) && (hash->flags & XTR_HASH_AUTO_EXPAND))
    {
        allocate_new_pool(hash);
    }

    // Insert the item if duplicates are not allowed, or there was no collision
    if (!added && (hash->pool_first_free != NULL))
    {
        // Pick the first free item
        xtr_hash_item *it = hash->pool_first_free;
        hash->pool_first_free = hash->pool_first_free->next;

        // Fill the item
        it->key = key;
        it->next = hash->buckets[index];
        if (data_in != NULL)
        {
            memcpy(it->data, data_in, hash->data_size);
        }

        // Store in the hash index
        hash->buckets[index] = it;

        added = 1;
    }

#if defined(DEBUG)
    if (added)
    {
        xtr_hash_stats_update(hash, num_adds);
    }
    else
    {
        xtr_hash_stats_update(hash, num_overflows);
    }
#endif

    if (hash->flags & XTR_HASH_LOCK)
    {
        pthread_rwlock_unlock(&hash->lock);
    }

    return added;
}


/**
 * xtr_hash_remove
 *
 * Removes the specified 'key' from the hash table and copies back the data in the
 * structure pointed by 'data'.
 * 
 * @param hash Pointer to the hash structure.
 * @param key The key to remove.
 * @param data_out [out] If not NULL, stored data for the given 'key' is copied back to this buffer.
 * 
 * @return 1 if the key was found; 0 otherwise.
 */
int xtr_hash_remove(xtr_hash *hash, UINT64 key, void *data_out)
{
    int index = -1;
    int found = 0;
    xtr_hash_item *it = NULL;
    xtr_hash_item *prev = NULL;

    if (hash->flags & XTR_HASH_LOCK)
    {
        pthread_rwlock_rdlock(&hash->lock);
    }

    // Calculate the hash index
    index = xtr_hash_function(hash, key);

    // Look in the hash index through the collision list for the matching key
    it = hash->buckets[index];
    while ((it != NULL) && !found)
    {
        if (it->key == key)
        {
            // Copy data back to the user
            if (data_out != NULL)
            {
                memcpy(data_out, it->data, hash->data_size);
            }
            // Update the collision list
            if (prev == NULL)
            {
                // Erase first element
                hash->buckets[index] = it->next;
            }
            else
            {
                // Erase middle element
                prev->next = it->next;
            }
            // Queue the released element into the free pool
            it->next = hash->pool_first_free;
            hash->pool_first_free = it;

            found = 1;
        }
        prev = it;
        it = it->next;
    }

#if defined(DEBUG)
    if (found)
    {
        xtr_hash_stats_update(hash, num_deletes);
        xtr_hash_stats_update(hash, num_hits);
    }
    else
    {
        xtr_hash_stats_update(hash, num_misses);
    }
#endif

    if (hash->flags & XTR_HASH_LOCK)
    {
        pthread_rwlock_unlock(&hash->lock);
    }

    return found;
}


/**
 * xtr_hash_search
 *
 * Finds the specified 'key' in the hash table, and copies back the data in the structure
 * pointed by 'data_out'.
 *
 * @param hash Pointer to the hash structure.
 * @param key The key to search for.
 * @param data_out [out] If not NULL, stored data for the given 'key' is copied back to this buffer.
 *
 * @return 1 if the key was found; 0 otherwise.
 */
int xtr_hash_search(xtr_hash *hash, UINT64 key, void *data_out)
{
    xtr_hash_item *it = NULL;
    int index = -1;
    int found = 0;

    if (hash->flags & XTR_HASH_LOCK)
    {
        pthread_rwlock_rdlock(&hash->lock);
    }

    // Calculate the hash index
    index = xtr_hash_function(hash, key);

    // Look in the hash index through the collision list for the matching key
    it = hash->buckets[index];
    while ((it != NULL) && !found)
    {
        if (it->key == key)
        {
            // Copy data back to the user
            if (data_out != NULL)
            {
                memcpy(data_out, it->data, hash->data_size);
            }
            found = 1;
        }
        it = it->next;
    }

#if defined(DEBUG)
    if (found)
    {
        xtr_hash_stats_update(hash, num_hits);
    }
    else
    {
        xtr_hash_stats_update(hash, num_misses);
    }
#endif

    if (hash->flags & XTR_HASH_LOCK)
    {
        pthread_rwlock_unlock(&hash->lock);
    }

    return found;
}


/**
 * xtr_hash_dump
 *
 * Dumps the contents of the hash table.
 *
 * @param hash Pointer to the hash structure.
 * @param pretty_print_func_ptr Callback pointer to a function that knows how to
 * dump the data stored in this hash.
 */
void xtr_hash_dump(xtr_hash *hash, void *pretty_print_func_ptr)
{
    int i = 0;
    void (*pretty_print)(FILE *, void *) = pretty_print_func_ptr;

    // Iterate over all hash buckets
    for (i = 0; i < hash->num_buckets; i++)
    {
        xtr_hash_item *it = NULL;

        if ((it = hash->buckets[i]) != NULL)
        {
            int num_collisions = 0;

            fprintf(stderr, "xtr_hash_dump: Bucket #%d: ", i);
            // Iterate over the collision list
            do
            {
                fprintf(stderr, "<key: %ld data: ", it->key);
                pretty_print(stderr, it->data);
                fprintf(stderr, ">\n");
                num_collisions++;
            } while ((it = it->next) != NULL);
            fprintf(stderr, " [collisions = %d]\n", num_collisions);
        }
    }
}


/**
 * xtr_hash_stats_reset
 *
 * Resets usage statistics for the given 'hash'.
 * 
 * @param hash Pointer to the hash structure.
 */
void xtr_hash_stats_reset(xtr_hash *hash)
{
    hash->stats.num_adds = 0;
    hash->stats.num_overflows = 0;
    hash->stats.num_hits = 0;
    hash->stats.num_misses = 0;
    hash->stats.num_deletes = 0;
    hash->stats.num_collisions = 0;
    hash->stats.num_duplicates = 0;
}


/**
 * xtr_hash_stats_dump
 *
 * Prints usage statistics for the given 'hash'.
 *
 * @param hash Pointer to the hash structure.
 */
void xtr_hash_stats_dump(xtr_hash *hash)
{
    int num_free_items = 0;
    int num_leftovers = 0;
    int max_items = hash->pool_size * hash->num_pools;

    // Count all remaining free elements
    xtr_hash_item *it = hash->pool_first_free;
    while (it != NULL)
    {
        num_free_items++;
        it = it->next;
    }

    // Substract from the maximum pool size to calculate leftovers
    num_leftovers = max_items - num_free_items;

    fprintf(stderr, "xtr_hash_stats: Adds=%d\n", hash->stats.num_adds);
    fprintf(stderr, "xtr_hash_stats: Overflows=%d\n", hash->stats.num_overflows);
    fprintf(stderr, "xtr_hash_stats: Hits=%d\n", hash->stats.num_hits);
    fprintf(stderr, "xtr_hash_stats: Misses=%d\n", hash->stats.num_misses);
    fprintf(stderr, "xtr_hash_stats: Deletes=%d\n", hash->stats.num_deletes);
    fprintf(stderr, "xtr_hash_stats: Collisions=%d (%.2lf%%)\n", hash->stats.num_collisions,
                    (hash->stats.num_adds == 0 ? 0 : ((double)hash->stats.num_collisions / (double)hash->stats.num_adds) * 100));
    fprintf(stderr, "xtr_hash_stats: Duplicates=%d\n", hash->stats.num_duplicates);
    fprintf(stderr, "xtr_hash_stats: Leftovers=%d (%.2lf%%)\n", num_leftovers,
                    ((double)num_leftovers / max_items) * 100);
}

#if defined(UNIT_TEST)

static void pretty_int(FILE *fd, void *data) 
{
    int *x = data;

    fprintf(fd, "%d", *x);
}

int main(int argc, char **argv) 
{
    xtr_hash *hash = xtr_hash_new(4, sizeof(int), XTR_HASH_NONE | XTR_HASH_AUTO_EXPAND);

    int data1 = 111;
    int data2 = 222;
    int data3 = 333;
    int data4 = 444;
    int data5 = 555;
    int data6 = 666;
    int data7 = 777;
    int data8 = 888;

    xtr_hash_add(hash, 1, &data1, NULL);
    xtr_hash_add(hash, 2, &data2, NULL);
    xtr_hash_add(hash, 3, &data3, NULL);
    xtr_hash_add(hash, 4, &data4, NULL);
    xtr_hash_add(hash, 5, &data5, NULL);
    xtr_hash_dump(hash, pretty_int);
    xtr_hash_remove(hash, 1, NULL);
    xtr_hash_remove(hash, 2, NULL);
    xtr_hash_remove(hash, 3, NULL);
    xtr_hash_remove(hash, 4, NULL);
    xtr_hash_remove(hash, 5, NULL);
    xtr_hash_dump(hash, pretty_int);
    xtr_hash_add(hash, 1, &data1, NULL);
    xtr_hash_add(hash, 2, &data2, NULL);
    xtr_hash_add(hash, 3, &data3, NULL);
    xtr_hash_add(hash, 4, &data4, NULL);
    xtr_hash_add(hash, 5, &data5, NULL);
    xtr_hash_add(hash, 6, &data6, NULL);
    xtr_hash_add(hash, 7, &data7, NULL);
    xtr_hash_add(hash, 8, &data8, NULL);
    xtr_hash_dump(hash, pretty_int);
  
    fprintf(stderr, "Num pools: %d\n", hash->num_pools);
 
    /*
    int out;
    int ret = xtr_hash_add(hash, 6, &data6, &out);
    if (ret > 1) fprintf(stderr, "there was a replacement: %d\n", out);
    */

    /*
    int remove1;
    if ( hash_remove(h, 5, &remove1) ) printf("remove %d\n", remove1);

    int search3;
    if (xtr_hash_search(hash, 6, &search3)) printf("search %d\n", search3);
    if (xtr_hash_search(hash, 2, &search3)) printf("search %d\n", search3);
    if (xtr_hash_search(hash, 1, &search3)) printf("search %d\n", search3);
    if (xtr_hash_search(hash, 3, &search3)) printf("search %d\n", search3);
    */

    xtr_hash_stats_dump(hash);
}

#endif /* UNIT_TEST */

