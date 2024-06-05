#pragma once

# include <cuda_runtime_api.h>
# include <cupti.h>
# include <cupti_events.h>
#include <stddef.h>

typedef enum {
	EXTRAE_CUDA_NEW_TIME,
	EXTRAE_CUDA_PREVIOUS_TIME
}Extrae_CUDA_Time_Type;

typedef struct gpu_event_t gpu_event_t;

typedef struct gpu_event_t
{
	cudaEvent_t ts_event;            /**< CUDA event timestamp. */
	unsigned event;                  /**< Event identifier. */
	unsigned long long value;        /**< Event value. */
	unsigned tag;                    /**< Event tag. */
	unsigned size;                   /**< Event size. */
	Extrae_CUDA_Time_Type timetype;  /**< CUDA timing type. */
	gpu_event_t *next;               /**< Pointer to the next gpu_event_t in the list. */
} gpu_event_t;

typedef struct
{
	gpu_event_t *head;
	gpu_event_t *tail;
	int autoexpand;
	size_t chunk_size;
}gpu_event_list_t;

void gpuEventList_init(gpu_event_list_t *list, int autoexpand, size_t chunk_size);
void gpuEventList_allocate_chunk(gpu_event_list_t *list, size_t size);
void gpuEventList_add(gpu_event_list_t *list, gpu_event_t *element);
gpu_event_t *gpuEventList_pop(gpu_event_list_t *list);
gpu_event_t *gpuEventList_peek_tail(gpu_event_list_t *list);
int gpuEventList_isempty(gpu_event_list_t *list);
void gpuEventList_free(gpu_event_list_t *list);
