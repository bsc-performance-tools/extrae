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

#ifndef OMPT_RECORDS_H_INCLUDED
#define OMPT_RECORDS_H_INCLUDED

/* OMPT record type */
typedef enum {
  ompt_record_ompt,
  ompt_record_correlation,
  ompt_record_native,
  ompt_record_invalid
} ompt_record_type_t; 

typedef enum {
  ompt_record_native_class_info = 0,
  ompt_record_native_class_event = 1
} ompt_record_native_class_t;

/* native record abstract */
typedef struct ompt_record_native_abstract_s {
  ompt_record_native_class_t rclass;
  const char *type;
  ompt_target_time_t start_time;
  ompt_target_time_t end_time;
  ompt_target_activity_id_t dev_task_id;
  uint64_t hwid;
} ompt_record_native_abstract_t;

/* correlation */
typedef struct ompt_record_correlation_s {
  ompt_task_id_t host_task_id;         
  ompt_target_activity_id_t dev_task_id;
} ompt_record_correlation_t;

/* record types */
typedef struct ompt_record_thread_s {
  ompt_thread_id_t thread_id;      /* ID of thread                 */
} ompt_record_thread_t;

typedef struct ompt_record_thread_type_s {
  ompt_thread_type_t thread_type;  /* type of thread               */
  ompt_thread_id_t thread_id;      /* ID of thread                 */
} ompt_record_thread_type_t;

typedef struct ompt_record_wait_id_s {
  ompt_wait_id_t wait_id;          /* wait ID                      */
} ompt_record_wait_id_t;

typedef struct ompt_record_parallel_s {
  ompt_parallel_id_t parallel_id;  /* ID of parallel region        */
  ompt_task_id_t task_id;          /* ID of task                   */
} ompt_record_parallel_t;

typedef struct ompt_record_workshare_begin_s {
  ompt_parallel_id_t parallel_id;  /* ID of parallel region         */
  ompt_task_id_t task_id;          /* ID of task                    */
  void *codeptr_ra;                /* runtime call return address   */
} ompt_record_workshare_begin_t;

typedef struct ompt_record_parallel_begin_s {
  ompt_task_id_t parent_task_id;   /* ID of parent task             */
  ompt_frame_t *parent_frame;      /* frame data of parent task     */
  ompt_parallel_id_t parallel_id;  /* ID of parallel region         */
  uint32_t requested_team_size;    /* requested number of threads   */
  void *codeptr_ofn;               /* pointer to outlined function  */
} ompt_record_parallel_begin_t;

typedef struct ompt_record_task_s {
  ompt_task_id_t task_id; /* ID of task */
} ompt_record_task_t;

typedef struct ompt_record_task_pair_s {
  ompt_task_id_t first_task_id; 
  ompt_task_id_t second_task_id;  
} ompt_record_task_pair_t;

typedef struct ompt_record_task_begin_s {
  ompt_task_id_t parent_task_id;   /* ID of parent task             */
  ompt_frame_t *parent_task_frame; /* frame data of parent task     */
  ompt_task_id_t new_task_id;      /* ID of created task            */
  void *codeptr_ofn;               /* pointer to outlined function  */
} ompt_record_task_begin_t;

typedef struct ompt_record_target_task_begin_s {
  ompt_task_id_t parent_task_id;    /* ID of parent task            */
  ompt_frame_t *parent_task_frame;  /* frame data for parent task   */
  ompt_task_id_t target_task_id;    /* ID of target task            */
  int device_id;                    /* ID of the device             */
  void *target_task_code;           /* pointer to target task code  */
} ompt_record_target_task_begin_t;

typedef struct ompt_record_target_data_begin_s {
  ompt_task_id_t task_id;           /* ID of encountering task      */
  int device_id;                    /* ID of the device             */
  void *codeptr_ra;                 /* return address of api call   */
} ompt_record_target_data_begin_t;
 
typedef struct ompt_record_data_map_begin_s {
  ompt_task_id_t task_id;           /* ID of encountering task      */
  int device_id;                    /* ID of the device             */
  void *host_addr;                  /* host  address of the data    */
  void *device_addr;                /* device address of the data   */ 
  size_t bytes;                     /* number of bytes mapped       */
  uint32_t mapping_flags;           /* sync/async, to/from          */
  void *target_map_code;            /* ptr to target map code       */
} ompt_record_data_map_begin_t;

typedef struct ompt_record_data_map_done_s {
  ompt_task_id_t task_id;           /* ID of current task           */
  int device_id;                    /* ID of the device             */
  void *host_addr;                  /* host  address of the data    */
  void *device_addr;                /* device address of the data   */ 
  size_t bytes;                     /* number of bytes mapped       */
  uint32_t mapping_flags;           /* sync/async, to/from          */
} ompt_record_data_map_done_t;

typedef struct ompt_record_task_dependences_s {
  ompt_task_id_t task_id;           /* ID of task with dependences  */
  ompt_task_dependence_t *deps;     /* vector of task dependences   */
  int ndeps;                        /* number of dependences        */
} ompt_record_task_dependences_t;

/* OMPT record */
typedef struct ompt_record_ompt_s {
  ompt_event_t type;                /* event type                       */
  ompt_target_time_t time;          /* time record created              */
  ompt_thread_id_t thread_id;       /* thread ID for this record        */
  ompt_target_activity_id_t dev_task_id;  /* link to host context       */
  union
  {
    ompt_record_thread_t thread;                 /* for thread          */
    ompt_record_thread_type_t type;              /* for thread type     */
    ompt_record_wait_id_t waitid;                /* for wait            */
    ompt_record_parallel_t parallel;             /* for inside parallel */
    ompt_record_workshare_begin_t new_workshare; /* for workshares      */
    ompt_record_parallel_begin_t new_parallel;   /* for new parallel    */
    ompt_record_task_t task;                     /* for tasks           */
    ompt_record_task_pair_t task_switch;         /* for task switch     */
    ompt_record_task_begin_t new_task;           /* for new tasks       */
  } record;
} ompt_record_ompt_t;

#endif /* OMP_RECORDS_H_INCLUDED */
