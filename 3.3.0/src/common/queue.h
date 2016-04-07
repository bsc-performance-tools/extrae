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

#ifndef _QUEUE_H
#define _QUEUE_H

/* Define macros for the queue manipulation functions. */

/*
 * The "next" field points to the front of the list. This is the item
 * that will be returned on a "dequeue" operation. The "prev" field
 * points to the tail of the list. The "enqueue" operation links an
 * item at the tail of the list using the "prev" pointer.
 *
 * All queues are assumed to have a head node. An empty queue must be
 * initialized so that its "next" and "prev" fields point to the head node.
 */

/* 
 * Initialize the head node for a queue. 
 */
#define INIT_QUEUE(queue) \
((queue)->next = queue, (queue)->prev = queue)

/* 
 * Indicates if the queue is null or not. 
 */
#define IS_NULL(queue) \
((queue)->next == queue)

/* 
 * remove item from the queue 
 */
#define REMOVE_ITEM(item) \
((item)->next->prev = (item)->prev, (item)->prev->next = (item)->next)

/*
 * enqueue item at the head of the queue
 */
#define ENQUEUE_HEAD_ITEM(queue,item) \
((item)->next = (queue)->next, (item)->prev = (queue), (item)->next->prev = (item), (queue)->next = (item))

/*
 */
#define RESET_QUEUE(free_item_ptr,queue,type) \
{ \
    type *ptmp; \
    \
    while (!IS_NULL(queue)) { \
         ptmp = (queue)->next; \
         REMOVE_ITEM(ptmp);   \
         FREE_ITEM(free_item_ptr,ptmp); \
    } \
}

/*
 * returns the head of the queue
 */
#define GET_HEAD_ITEM(queue)  ( ((queue)->next == (queue)) ? NULL : (queue)->next)

/*
 * returns the next item
 */
#define GET_NEXT_ITEM(queue, item)  ( ((item)->next == (queue)) ? NULL : (item)->next)


/* 
 * enqueue item at the end of the queue 
 */
#define ENQUEUE_ITEM(queue,item) \
((item)->next = (queue), (item)->prev = (queue)->prev, (queue)->prev->next = (item), (queue)->prev = (item))

/* 
 * insert item item1 after item2 
 */
#define INSERT_ITEM_AFTER(item2,item1) \
((item1)->next = (item2)->next, (item1)->prev = (item2), (item2)->next->prev = (item1), (item2)->next = (item1))

/* 
 * insert item item1 before item2 
 */
#define INSERT_ITEM_BEFORE(item2,item1) \
((item1)->next = (item2), (item1)->prev = (item2)->prev, (item2)->prev->next = (item1), (item2)->prev = (item1))

/* 
 * dequeue a item from head of queue 
 */
#define DEQUEUE_ITEM(queue,item) \
((item)=(queue)->next, (item)==(queue) ? ((item)=NULL) : ((queue)->next=(item)->next, (item)->next->prev=(queue), (item)))

/*
 * Dequeue a item from head of queue, but do not check for an empty queue.
 * If the queue was empty, then item will be set equal to queue, so the
 * program should check this after the macro is executed.
 */
#define DEQUEUE_ITEM_NOCHECK(queue,item) \
((item)=(queue)->next, (queue)->next=(item)->next, (item)->next->prev=(queue))

/*
 * Insert an item into a queue sorted into increasing order of the FIELD values 
 */
#define INSERT_ITEM_INCREASING(queue,ptr,type,field) \
{ \
    type *ptmp; \
    \
    for (ptmp = (queue)->prev; ptmp != (queue); ptmp = ptmp->prev) { \
	if (ptmp->field <= (ptr)->field) \
	    break; \
    } \
    INSERT_ITEM_AFTER(ptmp, (ptr)); \
}

/* 
 * Insert an item into a queue sorted into decreasing order of the FIELD values 
 */
#define INSERT_ITEM_DECREASING(queue,ptr,type,field) \
{ \
    type *ptmp; \
    \
    for (ptmp = (queue)->prev; ptmp != (queue); ptmp = ptmp->prev) { \
	if (ptmp->field >= (ptr)->field) \
	    break; \
    } \
    INSERT_ITEM_AFTER(ptmp, (ptr)); \
}

/* 
 * these versions of the macros are the same except that the next and prev
 * fields are specified in the macro. This allows a structure to be on
 * different queues at the same time.
 */

/* 
 * Initialize the head node for a queue. 
 */
#define INIT_QUEUE_NP(queue,next,prev) \
((queue)->next = queue, (queue)->prev = queue)

/* 
 * remove item from the queue 
 */
#define REMOVE_ITEM_NP(item,next,prev) \
((item)->next->prev = (item)->prev, (item)->prev->next = (item)->next)

/* 
 * enqueue item at the end of the queue 
 */
#define ENQUEUE_ITEM_NP(queue,item,next,prev) \
((item)->next = (queue), (item)->prev = (queue)->prev, (queue)->prev->next = (item), (queue)->prev = (item))

/* 
 * insert item item1 after item2 
 */
#define INSERT_ITEM_AFTER_NP(item2,item1,next,prev) \
((item1)->next = (item2)->next, (item1)->prev = (item2), (item2)->next->prev = (item1), (item2)->next = (item1))

/* 
 * insert item item1 before item2 
 */
#define INSERT_ITEM_BEFORE_NP(item2,item1,next,prev) \
((item1)->next = (item2), (item1)->prev = (item2)->prev, (item2)->prev->next = (item1), (item2)->prev = (item1))

/*
 * dequeue a item from head of queue 
 */
#define DEQUEUE_ITEM_NP(queue,item,next,prev) \
((item)=(queue)->next, (item)==(queue) ? ((item)=NULL) : ((queue)->next=(item)->next, (item)->next->prev=(queue), (item)))

/*
 * Dequeue a item from head of queue, but do not check for an empty queue.
 * If the queue was empty, then item will be set equal to queue, so the
 * program should check this after the macro is executed.
 */
#define DEQUEUE_ITEM_NOCHECK_NP(queue,item,next,prev) \
((item)=(queue)->next, (queue)->next=(item)->next, (item)->next->prev=(queue))

/*
 * Insert an item into a queue sorted into increasing order of the FIELD values 
 */
#define INSERT_ITEM_INCREASING_NP(queue,ptr,type,field,next,prev) \
{ \
    type *ptmp; \
    \
    for (ptmp = (queue)->prev; ptmp != (queue); ptmp = ptmp->prev) { \
        if (ptmp->field <= (ptr)->field) \
            break; \
    } \
    INSERT_ITEM_AFTER_NP(ptmp, (ptr)); \
}

/*
 * Insert an item into a queue sorted into decreasing order of the FIELD values 
 */
#define INSERT_ITEM_DECREASING_NP(queue,ptr,type,field,next,prev) \
{ \
    type *ptmp; \
    \
    for (ptmp = (queue)->prev; ptmp != (queue); ptmp = ptmp->prev) { \
        if (ptmp->field >= (ptr)->field) \
            break; \
    } \
    INSERT_ITEM_AFTER_NP(ptmp, (ptr)); \
}


/* 
 * number of structs to allocate in one call to malloc when out of free items 
 */

#ifndef NITEMS
#define NITEMS 30
#endif

#ifndef TRACE_MALLOC

/* 
 * Define a macro for allocating a new structure. This is trickier than
 * the corresponding function definition since we do not refer to any
 * of the fields by name. (We could cast each item pointer to another type
 * of pointer with known field names, but we can also do it without
 * defining another structure.)
 */
#define ALLOC_NEW_ITEM(free_item_ptr, item_size, new_ptr, name) \
{ \
    int _ii; \
\
    /* Reduce calls to malloc and make more efficient use of space \
     * by allocating several structs at once. \
     */ \
    if ((free_item_ptr) == NULL) { \
        new_ptr = (void *) malloc((NITEMS) * (item_size)); \
        if ((new_ptr) == NULL) { \
            fprintf(stderr, "%s: out of memory\n", name); \
            exit(1); \
        } \
 \
        /* Link all the free structs using the first field in the struct. */ \
        /* This assumes that the structure is large enough to hold a pointer. */ \
        free_item_ptr = (new_ptr); \
        for (_ii = 0; _ii < (NITEMS) - 1; _ii++) { \
            *((long **)(free_item_ptr)) = (long *) ((char *) (new_ptr) + ((_ii + 1) * (item_size))); \
            free_item_ptr = (void *) *(long **)(free_item_ptr); \
        } \
        *((long **)(free_item_ptr)) = NULL; \
        free_item_ptr = (new_ptr); \
    } \
 \
    /* Remove a free item from the front of the list. */ \
    new_ptr = free_item_ptr; \
    free_item_ptr = (void *) *(long **)(free_item_ptr); \
}

/* 
 * Define a macro for allocating a new structure, initialized to zero.
 * Zero the item only when it is about to be used, not when allocating
 * the array. This saves real memory when large items are allocated
 * and not used, and saves time since the item must be zeroed anyway
 * just before it is returned (since it may have been used and then freed).
 */
#define ALLOC_NEW_ZITEM(free_item_ptr, item_size, new_ptr, name) \
{ \
    int _ii; \
\
    /* Reduce calls to malloc and make more efficient use of space \
     * by allocating several structs at once. \
     */ \
    if ((free_item_ptr) == NULL) { \
        new_ptr = (void *) malloc((NITEMS) * (item_size)); \
        if ((new_ptr) == NULL) { \
            fprintf(stderr, "%s: out of memory\n", name); \
            exit(1); \
        } \
 \
        /* Link all the free structs using the first field in the struct. */ \
        /* This assumes that the structure is large enough to hold a pointer. */ \
        free_item_ptr = (new_ptr); \
        for (_ii = 0; _ii < (NITEMS) - 1; _ii++) { \
            *((long **)(free_item_ptr)) = (long *) ((char *) (new_ptr) + ((_ii + 1) * (item_size))); \
            free_item_ptr = (void *) *(long **)(free_item_ptr); \
        } \
        *((long **)(free_item_ptr)) = NULL; \
        free_item_ptr = (new_ptr); \
    } \
 \
    /* Remove a free item from the front of the list. */ \
    new_ptr = free_item_ptr; \
    free_item_ptr = (void *) *(long **)(free_item_ptr); \
    memset(new_ptr, 0, item_size); \
}

/* define a macro for free-ing a new structure */
#define FREE_ITEM(free_item_ptr, item) \
{ \
    /* Put a free item on the front of the list. */ \
    *((long **) (item)) = (long *) free_item_ptr; \
    free_item_ptr = item; \
}

#else /* TRACE_MALLOC */

/* Versions that print a trace of their uses */
#define ALLOC_NEW_ITEM(free_item_ptr, item_size, new_ptr, name) \
{ \
    int _ii; \
\
    /* Reduce calls to malloc and make more efficient use of space \
     * by allocating several structs at once. \
     */ \
    if ((free_item_ptr) == NULL) { \
        new_ptr = (void *) malloc((NITEMS) * (item_size)); \
        if ((new_ptr) == NULL) { \
            fprintf(stderr, "%s: out of memory\n", name); \
            exit(1); \
        } \
 \
        /* Link all the free structs using the first field in the struct. */ \
        /* This assumes that the structure is large enough to hold a pointer. */ \
        free_item_ptr = (new_ptr); \
        for (_ii = 0; _ii < (NITEMS) - 1; _ii++) { \
            *((long **)(free_item_ptr)) = (long *) ((char *) (new_ptr) + ((_ii + 1) * (item_size))); \
            free_item_ptr = (void *) *(long **)(free_item_ptr); \
        } \
        *((long **)(free_item_ptr)) = NULL; \
        free_item_ptr = (new_ptr); \
    } \
 \
    /* Remove a free item from the front of the list. */ \
    new_ptr = free_item_ptr; \
    free_item_ptr = (void *) *(long **)(free_item_ptr); \
    fprintf(stderr, "%s new 0x%x 0x%x\n", (name), (void *) (new_ptr), (item_size)); \
}

#define ALLOC_NEW_ZITEM(free_item_ptr, item_size, new_ptr, name) \
{ \
    int _ii; \
\
    /* Reduce calls to malloc and make more efficient use of space \
     * by allocating several structs at once. \
     */ \
    if ((free_item_ptr) == NULL) { \
        new_ptr = (void *) malloc((NITEMS) * (item_size)); \
        if ((new_ptr) == NULL) { \
            fprintf(stderr, "%s: out of memory\n", name); \
            exit(1); \
        } \
 \
        /* Link all the free structs using the first field in the struct. */ \
        /* This assumes that the structure is large enough to hold a pointer. */ \
        free_item_ptr = (new_ptr); \
        for (_ii = 0; _ii < (NITEMS) - 1; _ii++) { \
            *((long **)(free_item_ptr)) = (long *) ((char *) (new_ptr) + ((_ii + 1) * (item_size))); \
            free_item_ptr = (void *) *(long **)(free_item_ptr); \
        } \
        *((long **)(free_item_ptr)) = NULL; \
        free_item_ptr = (new_ptr); \
    } \
 \
    /* Remove a free item from the front of the list. */ \
    new_ptr = free_item_ptr; \
    free_item_ptr = (void *) *(long **)(free_item_ptr); \
    memset(new_ptr, 0, item_size); \
    fprintf(stderr, "%s NEWZ 0x%x 0x%x\n", (name), (void *) (new_ptr), (item_size)); \
}

#define FREE_ITEM(free_item_ptr, item_ptr) \
{ \
    fprintf(stderr, "free item 0x%x\n", (void *) (item_ptr)); \
    /* Put a free item on the front of the list. */ \
    *((long **) (item_ptr)) = (long *) free_item_ptr; \
    free_item_ptr = item_ptr; \
}

#endif /* TRACE_MALLOC */

/* Count the number of items on the free list */
#define NUM_ITEMS_FREE(free_item_ptr, num_items) \
{ \
    long *ptr; \
\
    ptr = (long *) free_item_ptr; \
    for (num_items = 0; ptr; ptr = (long *) *ptr) \
	num_items++; \
}

#endif /* _QUEUE_H */
