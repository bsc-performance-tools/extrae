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

#include "common.h"

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif

#include "misc_wrapper.h"
#include "wrapper.h"
#include "threadid.h"
#include "threadinfo.h"

#include "ompt-helper.h"
#include "ompt-target.h"

/* List storing information about all the OMPT devices */
static extrae_device_info_t *List_of_Devices = NULL;

/* Global pointer to the ompt_get_num_devices API function */
static int (*ompt_get_num_devices_fn)(void) = NULL;

/**
 * Extrae_get_device_info
 *
 * Returns a struct containing information for the specified device (lookup, context, Extrae's thread identifier and latency correction).
 * 
 * @param device_id The device identifier
 * @return A extrae_device_info_t struct that contains the information for the given device; or NULL of the device is not found
 */
static extrae_device_info_t * Extrae_get_device_info ( int device_id )
{
  int i;

  for (i = 0; i < ompt_get_num_devices_fn(); i ++) 
  {
    if (List_of_Devices[i].ompt_device_id == device_id)
    {
      return &List_of_Devices[i];
    }
  } 
  return NULL;
}

/** 
 * Extrae_ompt_target_buffer_request
 * 
 * Callback invoked from OMPT to request a new buffer for storing ompt_record_ompt_t events from the devices 
 *
 * @param buffer A pointer to allocated host memory to store target records 
 * @param bytes The allocated buffer size in bytes
 */
void Extrae_ompt_target_buffer_request( ompt_target_buffer_t **buffer, size_t *bytes )
{
  int buf_size = sizeof(ompt_record_ompt_t) * EXTRAE_OMPT_TARGET_BUFFER_SIZE;

  ompt_target_buffer_t *buf = malloc( buf_size );

  if (buf != NULL)
  {
    *buffer = buf;
    *bytes  = buf_size;
  }
  else
  {
    *buffer = NULL;
    *bytes  = 0;
  }
}

/**
 * Extrae_ompt_target_buffer_complete
 *
 * Callback invoked from OMPT when the previously requested buffer through Extrae_ompt_target_buffer_request is full 
 * 
 * @param device_id The target device
 * @param buffer Pointer to buffer with target event records
 * @param bytes Number of valid bytes in the buffer
 * @param begin Position of first record
 * @param end Position after last record
 */
void Extrae_ompt_target_buffer_complete( int device_id,
                                         const ompt_target_buffer_t *buffer,
                                         size_t bytes,
                                         ompt_target_buffer_cursor_t begin,
                                         ompt_target_buffer_cursor_t end )
{
  /* Declare function pointers to the OMPT target API functions to manipulate the buffer of ompt_record_ompt_t events */
  ompt_record_ompt_t (*ompt_target_buffer_get_record_ompt_fn) (const ompt_target_buffer_t *, 
                                                               ompt_target_buffer_cursor_t) = NULL;

  int (*ompt_target_advance_buffer_cursor_fn) (const ompt_target_buffer_t *, 
                                               ompt_target_buffer_cursor_t, 
                                               ompt_target_buffer_cursor_t *) = NULL;

  double (*ompt_target_translate_time_fn) (ompt_target_device_t *, 
                                           ompt_target_time_t) = NULL;

  /* Retrieve the Extrae's logical thread id, the lookup and the context for the given device */
  extrae_device_info_t  *device_info      = Extrae_get_device_info( device_id );

  int                    extrae_thread_id = device_info->extrae_thread_id;
  ompt_function_lookup_t device_lookup    = device_info->lookup;
  ompt_target_device_t  *device_ptr       = device_info->device_ptr;

  /* Retrieve the pointers to the OMPT target functions to manipulate the buffer through the device lookup */
  ompt_target_buffer_get_record_ompt_fn = 
    (ompt_record_ompt_t(*)( const ompt_target_buffer_t *, 
                            ompt_target_buffer_cursor_t )) device_lookup("ompt_target_buffer_get_record_ompt");

  ompt_target_advance_buffer_cursor_fn = 
    (int(*)( const ompt_target_buffer_t *, 
             ompt_target_buffer_cursor_t, 
             ompt_target_buffer_cursor_t * )) device_lookup("ompt_target_advance_buffer_cursor");

  /* Also retrieve the function to query the device time translated to the host clock */
  ompt_target_translate_time_fn = 
    (double(*)( ompt_target_device_t *, 
                ompt_target_time_t )) device_lookup("ompt_target_translate_time");

  /* Check the buffer has data */
  if (bytes > 0)
  {
    ompt_target_buffer_cursor_t current, next;

    /* Iterate through all the ompt_record_ompt_t events stored in the buffer */
    current = begin;
    do
    {
      int task_entering, task_exiting;
      const void *task_function_address = NULL;
      unsigned long long time = 0;

      /* Fetch the record to process */
      ompt_record_ompt_t current_record = ompt_target_buffer_get_record_ompt_fn(buffer, current);

      /* Check the event type of the record */
      switch(current_record.type)
      {

        /* TASK BEGIN */
        case ompt_event_task_begin:
        {
          /* Get the id and @ from the entering task */
          task_entering = current_record.record.new_task.new_task_id;
          task_function_address = current_record.record.new_task.codeptr_ofn;

          /* Store the pair of task (id, @) in a table of tasks that are running */
          Extrae_OMPT_register_ompt_task_id_tf (task_entering, task_function_address, FALSE);
        } break;
	
        /* TASK SWITCH */
        case ompt_event_task_switch:
        {
          /* Get the id both from the exiting and entering tasks (the @ is not available here, 
             we have to retrieve it from the table stored at the TASK BEGIN event) */
          task_exiting  = current_record.record.task_switch.first_task_id;
          task_entering = current_record.record.task_switch.second_task_id;

          /* Correct the time of this event */
          time = (long long)ompt_target_translate_time_fn(device_ptr, current_record.time) + device_info->latency;

          /* Mark in the trace the routine associated to the task that is exiting */
          if (task_exiting > 0)
          {
            /* Retrieve the @ of the routine from the task id (stored in the table at the TASK BEGIN event) */
            if ((task_function_address = Extrae_OMPT_get_tf_task_id(task_exiting, NULL, NULL)))
            {
              THREAD_TRACE_MISCEVENT( extrae_thread_id, time, TASKFUNC_EV, 0, EMPTY);
          
              /* Mark that this task is no longer running (why? see TASK END) */
              Extrae_OMPT_tf_task_id_set_running (task_exiting, FALSE);
            }
          }

          /* Mark in the trace the routine associated to the task that is entering */
          if (task_entering > 0)
          {
            /* Retrieve the @ of the routine from the task id (stored in the table at the TASK BEGIN event) */
            if ((task_function_address = Extrae_OMPT_get_tf_task_id(task_entering, NULL, NULL)))
            {       
              THREAD_TRACE_MISCEVENT( extrae_thread_id, time, TASKFUNC_EV, (UINT64) task_function_address, EMPTY);
          
              /* Mark that this task is now running (why? see TASK END) */
              Extrae_OMPT_tf_task_id_set_running (task_entering, TRUE);
            }
          }
        } break;
          
        /* TASK END */
        case ompt_event_task_end:
        {
          /* Get the id from the exiting task */ 
          task_exiting = current_record.record.task.task_id;

          /* Check whether the exiting task is marked as running or not. This is necessary because some runtimes notify 
           * the end of the task also through a previous TASK SWITCH where this task is exiting. If we have emitted the 
           * exit event at the TASK SWITCH, we don't have to emit another one here. 
           */
          if (Extrae_OMPT_tf_task_id_is_running(task_exiting))
          {
            THREAD_TRACE_MISCEVENT( extrae_thread_id, time, TASKFUNC_EV, 0, EMPTY);
          }
          /* Remove this task from the table of running tasks */
          Extrae_OMPT_unregister_ompt_task_id_tf (task_exiting);

        } break;

        default:
        {
          /* Other OMPT events are ignored at the moment */
        } break;
      }
			
      /* Fetch the next OMPT record */
      ompt_target_advance_buffer_cursor_fn( buffer, current, &next );
      current = next;

    } while (current != end); /* End of the loop that processes the ompt_record_ompt_t records */
  } 
}

/**
 * ompt_target_initialize
 *
 * Initializes the tracing of OMPT devices. This is directly called from ompt_initialize, 
 * so the tracing of the OMPT devices is started right after activating the tracing for the host.
 *
 * \param lookup The lookup pointer used to retrieve the OMPT API functions.
 */
void ompt_target_initialize(ompt_function_lookup_t lookup)
{
  int device_id = 0;

  /* Declare function pointers to several OMPT API functions */
  int (*ompt_target_get_device_info_fn) (int, 
                                         const 
                                         char **, 
                                         ompt_target_device_t **, 
                                         ompt_function_lookup_t *, 
                                         const char *) = NULL;

  ompt_target_time_t (*ompt_target_get_time_fn) (ompt_target_device_t *) = NULL;

  double (*ompt_target_translate_time_fn) (ompt_target_device_t *, 
                                           ompt_target_time_t) = NULL;

  /* Retrieve the ompt_target_get_device_info API function through the global lookup */
  ompt_target_get_device_info_fn = (int(*)(int, 
                                           const char **, 
                                           ompt_target_device_t **, 
                                           ompt_function_lookup_t *, 
                                           const char *)) lookup("ompt_target_get_device_info");

  /* Retrieve the ompt_get_num_devices API function through the global lookup */
  ompt_get_num_devices_fn = (int(*)( void )) lookup("ompt_get_num_devices");
  
  /* Iterate through all the devices. We assume the identifiers returned from ompt_get_num_devices() go from 0 to n-1 */
  for (device_id = 0; device_id < ompt_get_num_devices_fn(); device_id ++)
  {
    int new_extrae_thread_id = 0;

    unsigned long long current_host_time;
    ompt_target_time_t current_device_time_raw;
    double             current_device_time;
    long long          latency;

    const char            *device_name = NULL;
    ompt_target_device_t  *device_ptr  = NULL;
    ompt_function_lookup_t device_lookup;

    /* Retrieve information from this device */
    ompt_target_get_device_info_fn( device_id, 
                                    &device_name,
                                    &device_ptr,
                                    &device_lookup,
                                    NULL );

    /* Retrieve the ompt_target_get_time API function through the device lookup */ 
    ompt_target_get_time_fn = (ompt_target_time_t(*)(ompt_target_device_t *)) device_lookup("ompt_target_get_time");

    /* Retrieve the ompt_target_translate_time API function through the device lookup */
    ompt_target_translate_time_fn = (double(*)(ompt_target_device_t *, ompt_target_time_t)) device_lookup("ompt_target_translate_time");

    /* Read the current device time, then the host time, and store the delta to compute later the time synchronization */
    current_device_time_raw = ompt_target_get_time_fn( device_ptr );
    current_host_time       = TIME;
    current_device_time     = ompt_target_translate_time_fn (device_ptr, current_device_time_raw);
    latency                 = (long long)current_device_time - (long long)current_host_time;

    /* Increase by 1 the number of threads in Extrae */
    new_extrae_thread_id = Backend_getNumberOfThreads();
    Backend_ChangeNumberOfThreads (new_extrae_thread_id + 1);
    Extrae_set_thread_name (new_extrae_thread_id, (char *)device_name);

    /* Save a mapping to translate from the OMPT device id to the logical thread id in Extrae, 
       and also store all the information about the device that we'll be needing later */
    List_of_Devices = realloc( List_of_Devices, (device_id + 1) * sizeof(List_of_Devices) );

    List_of_Devices[device_id].ompt_device_id   = device_id; 
    List_of_Devices[device_id].lookup           = device_lookup;
    List_of_Devices[device_id].device_ptr       = device_ptr;
    List_of_Devices[device_id].extrae_thread_id = new_extrae_thread_id;
    List_of_Devices[device_id].latency          = latency;

    /* Configure which events will be traced calling ompt_target_set_trace_ompt (ompt_record_type_t activates all) */

    int (*ompt_target_set_trace_ompt_fn)(ompt_target_device_t *, 
                                         ompt_bool, 
                                         ompt_record_type_t) = NULL;

    ompt_target_set_trace_ompt_fn = (int(*)(ompt_target_device_t *, 
                                            ompt_bool, 
                                            ompt_record_type_t)) device_lookup("ompt_target_set_trace_ompt");

    ompt_target_set_trace_ompt_fn( device_ptr, 1, ompt_record_ompt );
		

    /* Activate the tracing on this device calling ompt_target_start_trace */

    int (*ompt_target_start_trace_fn)(ompt_target_device_t *, 
                                      ompt_target_buffer_request_callback_t, 
                                      ompt_target_buffer_complete_callback_t) = NULL;

    ompt_target_start_trace_fn = (int(*)(ompt_target_device_t *, 
                                         ompt_target_buffer_request_callback_t, 
                                         ompt_target_buffer_complete_callback_t)) device_lookup("ompt_target_start_trace");

    /* The parameters Extrae_ompt_target_buffer_request and Extrae_ompt_target_buffer_complete are the two callbacks that
     * OMPT will invoke to request the allocation of a new buffer to store ompt_record_ompt_t events, and to flush the 
     * buffer when it is full. 
     */
    ompt_target_start_trace_fn( device_ptr, Extrae_ompt_target_buffer_request, Extrae_ompt_target_buffer_complete );

  } /* End of the loop that iterates through all the devices */

}
