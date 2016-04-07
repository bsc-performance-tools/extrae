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

#ifndef OMPT_API_H_INCLUDED
#define OMPT_API_H_INCLUDED

/* callback management */
OMPT_API int ompt_set_callback( /* register a callback for an event                      */
  ompt_event_t event,           /* the event of interest                                 */
  ompt_callback_t callback      /* function pointer for the callback                     */
);

OMPT_API int ompt_get_callback( /* return the current callback for an event (if any)     */
  ompt_event_t event,           /* the event of interest                                 */
  ompt_callback_t *callback     /* pointer to receive the return value                   */
);

/* state inquiry */
OMPT_API int ompt_enumerate_state( /* extract the set of states supported                */
  ompt_state_t current_state,   /* current state in the enumeration                      */
  ompt_state_t *next_state,     /* next state in the enumeration                         */
  const char **next_state_name  /* string description of next state                      */
);

/* lock type inquiry */
OMPT_API int ompt_enumerate_lock_type( /* extract the set of lock types supported        */
  ompt_lock_type_t current_type,/* current lock type in the enumeration                  */
  ompt_lock_type_t *next_type,  /* next lock type in the enumeration                     */
  const char **next_type_name   /* string description of next lock type                  */
);

/* thread inquiry */
OMPT_API ompt_thread_id_t ompt_get_thread_id( /* identify the current thread             */
  void
);

OMPT_API ompt_state_t ompt_get_state( /* get the state for a thread                      */
  ompt_wait_id_t *wait_id        /* for wait states: identify what awaited               */
);

/* parallel region inquiry */
OMPT_API ompt_parallel_id_t ompt_get_parallel_id( /* identify a parallel region          */
  int ancestor_level             /*  how many levels removed from the current region     */
);

OMPT_API int ompt_get_parallel_team_size( /* query # threads in a parallel region        */
  int ancestor_level             /*  how many levels removed from the current region     */
);

/* task inquiry */
OMPT_API ompt_bool ompt_get_task_info(
  int ancestor_level,            /* how many levels removed from the current task        */
  ompt_task_type_t *type,        /* return the type of the task                          */
  ompt_task_id_t *task_id,       /* return the ID of the task                            */
  
  ompt_frame_t **task_frame,     /* return the task_frame of the task                    */
  ompt_parallel_id_t *par_id     /* return the ID of the parallel region                 */
);

/* target device inquiry */
OMPT_API int ompt_target_get_device_id(         /* return active target device ID        */
  void
);

OMPT_API int ompt_get_num_devices(void);

OMPT_API int ompt_target_get_device_info(
  int device_id,
  const char **type,
  ompt_target_device_t **device,
  ompt_function_lookup_t *lookup,
  const char *documentation
);

OMPT_TARG_API ompt_target_time_t ompt_target_get_time( /* return current time on device  */
  ompt_target_device_t *device   /* target device handle                                 */
);

OMPT_TARG_API double ompt_target_translate_time(
  ompt_target_device_t *device,  /* target device handle                                 */
  ompt_target_time_t time
);

/* target tracing control */
OMPT_TARG_API int ompt_target_set_trace_ompt(
  ompt_target_device_t *device,  /* target device handle                                 */
  ompt_bool enable,              /* enable or disable                                    */
  ompt_record_type_t rtype       /* a record type                                        */
);

OMPT_TARG_API int ompt_target_set_trace_native(
  ompt_target_device_t *device,  /* target device handle                                 */
  ompt_bool enable,              /* enable or disable                                    */
  uint32_t  flags                /* event classes to monitor                             */
);

OMPT_TARG_API int ompt_target_start_trace (
  ompt_target_device_t *device,  /* target device handle                                 */
  ompt_target_buffer_request_callback_t request,  /* fn pointer to request trace buffer  */
  ompt_target_buffer_complete_callback_t complete /* fn pointer to return trace buffer   */
);

OMPT_TARG_API int ompt_target_pause_trace(
  ompt_target_device_t *device,  /* target device handle                                 */
  ompt_bool begin_pause
);

OMPT_TARG_API int ompt_target_stop_trace(
  ompt_target_device_t *device   /* target device handle                                 */
);

/* target trace record processing */
OMPT_TARG_API int ompt_target_advance_buffer_cursor(
  ompt_target_buffer_t *buffer,        /* handle for target trace buffer                 */
  ompt_target_buffer_cursor_t current, /* cursor identifying position in buffer          */
  ompt_target_buffer_cursor_t *next    /* pointer to new cursor for next position        */
);

OMPT_TARG_API ompt_record_type_t ompt_target_buffer_get_record_type(
  ompt_target_buffer_t *buffer,        /* handle for target trace buffer                 */
  ompt_target_buffer_cursor_t current  /* cursor identifying position in buffer          */
);

OMPT_TARG_API ompt_record_ompt_t *ompt_target_buffer_get_record_ompt(
  ompt_target_buffer_t *buffer,        /* handle for target trace buffer                 */
  ompt_target_buffer_cursor_t current  /* cursor identifying position in buffer          */
);

OMPT_TARG_API ompt_record_correlation_t *ompt_target_buffer_get_record_correlation(
  ompt_target_buffer_t *buffer,        /* handle for target trace buffer                 */
  ompt_target_buffer_cursor_t current  /* cursor identifying position in buffer          */
);
   
OMPT_TARG_API void *ompt_target_buffer_get_record_native(
  ompt_target_buffer_t *buffer,        /* handle for target trace buffer                 */
  ompt_target_buffer_cursor_t current  /* cursor identifying position in buffer          */
);
  
OMPT_TARG_API ompt_record_native_abstract_t *
ompt_target_buffer_get_record_native_abstract(
  void *native_record                  /* pointer to native trace record                 */
);

#endif /* OMPT_API_H_INCLUDED */
