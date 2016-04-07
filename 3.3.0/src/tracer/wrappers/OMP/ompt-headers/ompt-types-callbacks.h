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

#ifndef OMPT_TYPES_CALLBACKS_H_INCLUDED
#define OMPT_TYPES_CALLBACKS_H_INCLUDED

/* initialization */
typedef void (*ompt_interface_fn_t)(
  void
);

typedef ompt_interface_fn_t (*ompt_function_lookup_t)(
  const char *entry_point           /* entry point to look up       */
);

/* threads */	
typedef void (*ompt_thread_callback_t) ( /* for thread              */	   
  ompt_thread_id_t thread_id        /* ID of thread                 */
);

typedef void (*ompt_thread_type_callback_t) ( /* for thread         */
  ompt_thread_type_t thread_type,   /* type of thread               */	   
  ompt_thread_id_t thread_id        /* ID of thread                 */
);
	
typedef void (*ompt_wait_callback_t) ( /* for wait                  */
  ompt_wait_id_t wait_id            /* wait ID                      */
);

typedef void (*ompt_atomic_callback_t) ( /* for wait                */
  ompt_wait_id_t wait_id,           /* wait ID                      */
  uint32_t atomic_type              /* type of atomic operation     */
);

typedef void (*ompt_sync_callback_t) ( 			   
  ompt_parallel_id_t parallel_id,   /* ID of parallel region        */
  ompt_task_id_t  task_id,          /* ID of task                   */
  const void *codeptr_ra            /* return address of api call   */
);

/* parallel & workshares */									    
typedef void (*ompt_workshare_begin_callback_t) ( /* for workshares   */			   
  ompt_parallel_id_t parallel_id,   /* ID of parallel region        */
  ompt_task_id_t  task_id,          /* ID of task                   */
  const void *codeptr_ra            /* return address of api call   */
);								    
							   	    
typedef void (*ompt_parallel_begin_callback_t) ( /* for new parallel  */
  ompt_task_id_t parent_task_id,    /* ID of parent task            */
  const ompt_frame_t *parent_frame, /* frame data of parent task    */
  ompt_parallel_id_t parallel_id,   /* ID of parallel region        */
  uint32_t requested_team_size,     /* requested number of threads  */
  const void *codeptr_ofn,          /* pointer to outlined function */
  ompt_invoker_t invoker            /* who invokes master task?     */
);

typedef void (*ompt_parallel_callback_t) ( /* for inside parallel   */			   
  ompt_parallel_id_t parallel_id,   /* ID of parallel region        */
  ompt_task_id_t  task_id           /* ID of task                   */
);

typedef void (*ompt_parallel_end_callback_t) (
  ompt_parallel_id_t parallel_id,   /* ID of parallel region        */
  ompt_task_id_t task_id,           /* ID of task                   */
  ompt_invoker_t invoker            /* who invokes master task?     */
);
			   		
/* tasks */						    						    
typedef void (*ompt_task_callback_t) ( /* for tasks                 */	   
  ompt_task_id_t  task_id           /* ID of task                   */
);

typedef void (*ompt_task_begin_callback_t) ( /* for new tasks       */
  ompt_task_id_t parent_task_id,    /* ID of parent task            */
  const ompt_frame_t *parent_frame, /* frame data for parent task   */
  ompt_task_id_t new_task_id,       /* ID of created task           */
  const void *codeptr_ofn           /* pointer to outlined function */
);
  
typedef void (*ompt_target_task_begin_callback_t) (
  ompt_task_id_t parent_task_id,    /* ID of parent task            */
  const ompt_frame_t *parent_frame, /* frame data for parent task   */
  ompt_task_id_t host_task_id,      /* ID of target task on host    */
  int device_id,                    /* ID of the device             */
  const void *target_task_code,     /* ptr to target code           */
  ompt_target_task_type_t task_type /* the type of the target task  */
);

typedef void (*ompt_target_task_end_callback_t) ( 	   
  ompt_task_id_t host_task_id       /* ID of target task on host    */
);

/* lock initialization */
typedef void (*ompt_lock_callback_t) (
  ompt_wait_id_t wait_id,           /* wait ID                      */
  ompt_lock_hint_t lock_hint,       /* OMP lock hint                */
  ompt_lock_type_t lock_type        /* lock type assigned           */ 
);

/* target device */
typedef void (*ompt_target_data_begin_callback_t) ( 
  ompt_task_id_t task_id,           /* ID of encountering task      */
  int device_id,                    /* ID of the device             */
  const void *codeptr_ra            /* return address of api call   */
);

typedef struct ompt_target_map_entry_s {
  void *host_addr;                  /* host  address of the data    */
  void *device_addr;                /* device address of the data   */ 
  size_t bytes;                     /* number of bytes mapped       */
  uint32_t mapping_flags;           /* sync/async, to/from          */
} ompt_target_map_entry_t;

typedef void (*ompt_target_data_map_begin_callback_t) (
  ompt_task_id_t task_id,           /* ID of encountering task      */
  int device_id,                    /* ID of the device             */
  const ompt_target_map_entry_t *items, /* items to be mapped       */
  uint32_t nitems,                  /* # of items to be mapped      */
  ompt_target_activity_id_t  map_id /* ID for map event             */
);

typedef void (*ompt_target_data_map_end_callback_t) (
  int device_id,                    /* ID of the device             */
  ompt_target_activity_id_t  map_id /* ID for map event             */
);

/* task dependences */
typedef void (*ompt_task_dependences_callback_t) (                                   
  ompt_task_id_t task_id,            /* ID of task with dependences */
  const ompt_task_dependence_t *deps,/* vector of task dependences  */
  int ndeps                          /* number of dependences       */
);

typedef void (*ompt_task_pair_callback_t) (
  ompt_task_id_t first_task_id,
  ompt_task_id_t second_task_id
);
        
/* program */						   
typedef void (*ompt_control_callback_t) ( /* for control            */	   
  uint64_t command,                 /* command of control call      */
  uint64_t modifier                 /* modifier of control call     */
);
  
typedef void (*ompt_callback_t)(    /* for shutdown                 */
  void
); 

/* target trace buffer management routines */
typedef void (*ompt_target_buffer_request_callback_t) (
  ompt_target_buffer_t** buffer,    /* pointer to host memory to store target records */
  size_t *bytes                     /* buffer size in bytes */
);
  
typedef void (*ompt_target_buffer_complete_callback_t) (
  int device_id,                     /* target device                                 */
  const ompt_target_buffer_t *buffer,/* pointer to buffer with target event records   */
  size_t bytes,                      /* number of valid bytes in the buffer           */
  ompt_target_buffer_cursor_t begin, /* position of first record                      */
  ompt_target_buffer_cursor_t end    /* position after last record                    */ 
);

#endif /* OMPT_TYPES_CALLBACKS_H_INCLUDED */
