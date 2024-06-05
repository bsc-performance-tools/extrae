
#include <stdlib.h>
#include <string.h>

#include "gpu_event_info.h"
#include "xalloc.h"

#define CHECK_CU_ERROR(err, cufunc)                             \
	if (err != CUDA_SUCCESS)                                      \
		{                                                           \
			printf ("Error %d for CUDA Driver API function '%s'.\n",  \
							err,  #cufunc);                                   \
			exit(-1);                                                 \
		}


/**
 * @brief Initializes an event list setting the first and last elements to NULL
 *
 * @param list A pointer to the `gpu_event_list_t` structure that represents the event list to initialize.
 * @param autoexpand Boolean to allow the allocation of more elemnets if the list is empty.
 * @param chunk_size size of the allocated memory chunk
 * 
 */
void gpuEventList_init(gpu_event_list_t *list, int autoexpand, size_t chunk_size)
{
	list->head = NULL;
	list->tail = NULL;
	list->autoexpand = autoexpand;
	list->chunk_size = chunk_size;
}

/**
 * @brief Allocates and initializes new event elements for the event list.
 *
 * This function allocates `size` number of new `gpu_event_t` elements, initializes them,
 * and adds them to the provided `gpu_event_list_t` list. Each `gpu_event_t` is initialized
 * by creating a CUDA event with default flags.
 *
 * @param list A pointer to the `gpu_event_list_t` structure where the new elements will be added.
 * @param size The number of new event elements to allocate and add to the list.
 *
 * @pre The `list` pointer must not be `NULL`.
 */
void gpuEventList_allocate_chunk(gpu_event_list_t *list, size_t size)
{
	size_t i;
	int err;
	gpu_event_t *event_info;
	for (i = 0; i < size; i++)
	{
		event_info = xmalloc_and_zero(sizeof(gpu_event_t));

		err = cudaEventCreateWithFlags (&(event_info->ts_event), CU_EVENT_DEFAULT);
		CHECK_CU_ERROR(err, cudaEventCreateWithFlags);

		gpuEventList_add(list, event_info);
	}
}

/**
 * @brief Adds a new event element to the event list.
 *
 * This function appends a new `gpu_event_t` element to the end of the provided `gpu_event_list_t` list.
 * It updates the `tail` pointer to the new element, and if the list was previously empty, it also sets
 * the `head` pointer to the new element.
 *
 * @param list A pointer to the `gpu_event_list_t` structure representing the event list.
 * @param element A pointer to the `gpu_event_t` element to add to the list.
 *
 * @pre The `list` and `element` pointers must not be `NULL`.
 */
void gpuEventList_add(gpu_event_list_t *list, gpu_event_t *element)
{
	element->next = NULL;

	if (list->tail != NULL) {
		list->tail->next = element;
	} else {
		list->head = element;
	}
	list->tail = element;
}

/**
 * @brief Removes and returns the head element from the event list.
 *
 * This function removes the head `gpu_event_t` element from the provided `gpu_event_list_t` list
 * and returns a pointer to it. If the list becomes empty after the operation, the `tail` pointer
 * is set to `NULL`. The removed element's `next` pointer is set to `NULL` before returning.
 *
 * @param list A pointer to the `gpu_event_list_t` structure representing the event list.
 * @return A pointer to the removed `gpu_event_t` element, or `NULL` if the list was empty.
 *
 * @note The caller is responsible for handling the memory of the returned element if required.
 * @pre The `list` pointer must not be `NULL`.
 */
gpu_event_t *gpuEventList_pop(gpu_event_list_t *list)
{
	if(list->autoexpand && gpuEventList_isempty(list))
	{
		gpuEventList_allocate_chunk(list, list->chunk_size);
	}

	gpu_event_t *element = list->head;
	if(element != NULL) {
		list->head = element->next;
		if(list->tail == element) list->tail = NULL;
		element->next = NULL;
	}
	return element;
}

/**
 * @brief Returns the tail element of the event list without removing it.
 *
 * This function returns a pointer to the tail `gpu_event_t` element of the provided `gpu_event_list_t` list.
 * The tail element is the last element in the list. If the list is empty, the function returns `NULL`.
 *
 * @param list A pointer to the `gpu_event_list_t` structure representing the event list.
 * @return A pointer to the tail `gpu_event_t` element, or `NULL` if the list is empty.
 *
 * @note The function does not modify the list; it only provides a reference to the tail element.
 * @pre The `list` pointer must not be `NULL`.
 */
gpu_event_t *gpuEventList_peek_tail(gpu_event_list_t *list)
{
	return list->tail;
}

/**
 * @brief Checks if the event list is empty.
 *
 * @param list A pointer to the `EventInfoList` structure representing the event list.
 * @return 1 if the list is empty, 0 otherwise.
 */
int gpuEventList_isempty(gpu_event_list_t *list)
{
	return list->head == NULL;
}

/**
 * @brief Frees all elements in the event list and destroys associated CUDA events.
 *
 * This function iterates through the `gpu_event_list_t` list and frees all `gpu_event_t` elements.
 * For each element, the associated CUDA event is destroyed using `cudaEventDestroy`.
 *
 * @param list A pointer to the `gpu_event_list_t` structure representing the event list to free.
 */
void gpuEventList_free(gpu_event_list_t *list)
{
	if(list == NULL) return;

	gpu_event_t *current = list->head;
	gpu_event_t *next;
	while (current)
	{
		next = current->next;
		cudaEventDestroy(current->ts_event);
		xfree(current);
		current = next;
	}

	list->head = NULL;
	list->tail = NULL;
}
