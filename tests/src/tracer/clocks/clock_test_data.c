#include "common.h"
#include "trace_buffers.h"

/*
 * The existing clock tests run without the Extrae backend.
 * This helper provides the thread record normally supplied by the backend.
 * For example, Clock_getCurrentTime(0) uses it to save thread 0's timestamp.
 */
Extrae_thread_trace_data_t *get_thread_trace_data (unsigned thread_id)
{
	static Extrae_thread_trace_data_t data;
	return thread_id == 0 ? &data : NULL;
}
