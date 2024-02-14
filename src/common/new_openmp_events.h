#pragma once

#define NEW_OMP_BASE_EV              60000

// OpenMP event types
enum
{
  NEW_OMP_CALL_EV = NEW_OMP_BASE_EV,  // Marks the runtime call/callback executed 

  NEW_OMP_NESTED_EV,                  // Marks nested parallelism level in nested regions 
  NEW_OMP_PARALLEL_EV,                // Marks the fork operation (master thread) and the whole parallel phase (all threads)
  NEW_OMP_WSH_EV,                     // Worksharings constructs
  NEW_OMP_SYNC_EV,                    // Synchronization constructs 
  NEW_OMP_LOCK_EV,                    // Status of locks in ATOMIC & CRITICAL synchronizations
  NEW_OMP_LOCK_NAME_EV,               // Address of named locks 
  NEW_OMP_ORDERED_EV,                 // Phases of ORDERED & DOACROSS synchronizations
  NEW_OMP_TASKGROUP_EV,               // Phases of a TASKGROUP synchronization
  NEW_OMP_TASKING_EV,                 // TASK and TASKLOOP constructs

  NEW_OMP_ADDRESS_EV,                 // Address of outlined function without state change
  NEW_OMP_OUTLINED_ADDRESS_EV,        // Address of outlined function with state change. TODO: split into NEW_OMP_ADDRESS_EV and NEW_OMP_OUTLINED_EV
  NEW_OMP_OUTLINED_LINE_EV,           // Emitted in merger for file/line of NEW_OMP_OUTLINED_ADDRESS_EV

  NEW_OMP_TASK_INST_ID_EV,            // Task ID    (instantiation)
  NEW_OMP_TASK_INST_ADDRESS_EV,       // Task @ADDR (instantiation) TODO: review
  NEW_OMP_TASK_INST_LINE_EV,          // Emitted in merger for file/line of OMP_TASK_INST_ADDRESS_EV
  NEW_OMP_TASK_EXEC_ID_EV,            // Task ID    (execution)
  NEW_OMP_TASK_EXEC_ADDRESS_EV,       // Task @ADDR (execution). TODO: split into NEW_OMP_ADDRESS_EV and NEW_OMP_TASK_EXEC_EV
  NEW_OMP_TASK_EXEC_LINE_EV           // Emitted in merger for file/line of OMP_TASK_EXEC_ADDRESS_EV
};

#define NEW_OMP_OUTLINED_NAME_EV      NEW_OMP_OUTLINED_ADDRESS_EV  // Swap in merger of NEW_OMP_OUTLINED_ADDRESS_EV for function name translation
#define NEW_OMP_TASK_INST_NAME_EV     NEW_OMP_TASK_INST_ADDRESS_EV // Swap in merger of OMP_TASK_INST_ADDRESS_EV for function name translation
#define NEW_OMP_TASK_EXEC_NAME_EV     NEW_OMP_TASK_EXEC_ADDRESS_EV // Swap in merger of OMP_TASK_EXEC_ADDRESS_EV for function name translation


// Values for NEW_OMP_CALL_EV
enum
{
  /* GOMP */
  GOMP_ATOMIC_START_VAL = 1,
  GOMP_ATOMIC_END_VAL,
  GOMP_BARRIER_VAL,
  GOMP_CRITICAL_START_VAL,
  GOMP_CRITICAL_END_VAL,
  GOMP_CRITICAL_NAME_START_VAL,
  GOMP_CRITICAL_NAME_END_VAL,
  GOMP_LOOP_STATIC_START_VAL,
  GOMP_LOOP_DYNAMIC_START_VAL,
  GOMP_LOOP_GUIDED_START_VAL,
  GOMP_LOOP_RUNTIME_START_VAL,
  GOMP_LOOP_STATIC_NEXT_VAL,
  GOMP_LOOP_DYNAMIC_NEXT_VAL,
  GOMP_LOOP_GUIDED_NEXT_VAL,
  GOMP_LOOP_RUNTIME_NEXT_VAL,
  GOMP_LOOP_ORDERED_STATIC_START_VAL,
  GOMP_LOOP_ORDERED_DYNAMIC_START_VAL,
  GOMP_LOOP_ORDERED_GUIDED_START_VAL,
  GOMP_LOOP_ORDERED_RUNTIME_START_VAL,
  GOMP_LOOP_ORDERED_STATIC_NEXT_VAL,
  GOMP_LOOP_ORDERED_DYNAMIC_NEXT_VAL,
  GOMP_LOOP_ORDERED_GUIDED_NEXT_VAL,
  GOMP_LOOP_ORDERED_RUNTIME_NEXT_VAL,
  GOMP_PARALLEL_LOOP_STATIC_START_VAL,
  GOMP_PARALLEL_LOOP_DYNAMIC_START_VAL,
  GOMP_PARALLEL_LOOP_GUIDED_START_VAL,
  GOMP_PARALLEL_LOOP_RUNTIME_START_VAL,
  GOMP_LOOP_END_VAL,
  GOMP_LOOP_END_NOWAIT_VAL,
  GOMP_ORDERED_START_VAL,
  GOMP_ORDERED_END_VAL,
  GOMP_PARALLEL_START_VAL,
  GOMP_PARALLEL_END_VAL,
  GOMP_PARALLEL_SECTIONS_START_VAL,
  GOMP_PARALLEL_SECTIONS_VAL,
  GOMP_SECTIONS_START_VAL,
  GOMP_SECTIONS_NEXT_VAL,
  GOMP_SECTIONS_END_VAL,
  GOMP_SECTIONS_END_NOWAIT_VAL,
  GOMP_SINGLE_START_VAL,
  GOMP_TASKWAIT_VAL,
  GOMP_PARALLEL_VAL,
  GOMP_PARALLEL_LOOP_STATIC_VAL,
  GOMP_PARALLEL_LOOP_DYNAMIC_VAL,
  GOMP_PARALLEL_LOOP_GUIDED_VAL,
  GOMP_PARALLEL_LOOP_RUNTIME_VAL,
  GOMP_TASKGROUP_START_VAL,
  GOMP_TASKGROUP_END_VAL,
  GOMP_TASK_VAL,
  GOMP_TASKLOOP_VAL,
  GOMP_LOOP_DOACROSS_STATIC_START_VAL,
  GOMP_LOOP_DOACROSS_DYNAMIC_START_VAL,
  GOMP_LOOP_DOACROSS_GUIDED_START_VAL,
  GOMP_LOOP_DOACROSS_RUNTIME_START_VAL,
  GOMP_DOACROSS_POST_VAL,
  GOMP_DOACROSS_WAIT_VAL,
  GOMP_PARALLEL_LOOP_NONMONOTONIC_DYNAMIC_VAL,
  GOMP_LOOP_NONMONOTONIC_DYNAMIC_START_VAL,
  GOMP_LOOP_NONMONOTONIC_DYNAMIC_NEXT_VAL,
  GOMP_PARALLEL_LOOP_NONMONOTONIC_GUIDED_VAL,
  GOMP_LOOP_NONMONOTONIC_GUIDED_START_VAL,
  GOMP_LOOP_NONMONOTONIC_GUIDED_NEXT_VAL,
  GOMP_PARALLEL_LOOP_NONMONOTONIC_RUNTIME_VAL,
  GOMP_PARALLEL_LOOP_MAYBE_NONMONOTONIC_RUNTIME_VAL,
  GOMP_LOOP_NONMONOTONIC_RUNTIME_START_VAL,
  GOMP_LOOP_MAYBE_NONMONOTONIC_RUNTIME_START_VAL,
  GOMP_LOOP_NONMONOTONIC_RUNTIME_NEXT_VAL,
  GOMP_LOOP_MAYBE_NONMONOTONIC_RUNTIME_NEXT_VAL,
  GOMP_TEAMS_REG_VAL,
  GOMP_SET_LOCK_VAL,
  GOMP_UNSET_LOCK_VAL,

  MAX_OMP_CALLS
};


// Values for all other OpenMP events 
enum
{
  /* Exit values */

  // NEW_OMP_PARALLEL_EV
    NEW_OMP_PARALLEL_END_VAL = 0,
    NEW_OMP_FORK_END_VAL = 0,

  // NEW_OMP_LOCK_EV
    NEW_OMP_LOCK_RELEASED_VAL = 0,      // Mutex is unlocked

  // NEW_OMP_ORDERED_EV 
    NEW_OMP_ORDERED_POST_READY_VAL = 0, // Signal already sent

  // NEW_OMP_TASKGROUP_EV 
    NEW_OMP_TASKGROUP_END_VAL = 0,      // Taskgroup finished

  // NEW_OMP_WORKSHARING_EV
    NEW_OMP_WSH_END_VAL = 0,            // Worksharing region finished
    NEW_OMP_WSH_NEXT_CHUNK_END_VAL = 0, // Work request completed

  /* All other constructs can't share values to allow a single StackedVal view */

  // NEW_OMP_PARALLEL_EV
    NEW_OMP_PARALLEL_REGION_FORK_VAL = 1,   // Forking #pragma omp parallel (master thread)
    NEW_OMP_PARALLEL_REGION_VAL,            // Inside #pragma omp parallel (all threads)
    NEW_OMP_PARALLEL_LOOP_FORK_VAL,         // Forking #pragma omp parallel for (master thread)
    NEW_OMP_PARALLEL_LOOP_VAL,              // Inside #pragma omp parallel for (all threads)
    NEW_OMP_PARALLEL_SECTIONS_FORK_VAL,     // Forking #pragma omp parallel sections (master thread)
    NEW_OMP_PARALLEL_SECTIONS_VAL,          // Inside #pragma omp parallel sections (all threads)
    NEW_OMP_TEAMS_FORK_VAL,                 // Forking #pragma omp teams (master thread)
    NEW_OMP_TEAMS_VAL,                      // Inside #pragma omp teams (all threads)

  // NEW_OMP_WSH_EV
    NEW_OMP_WSH_NEXT_CHUNK_VAL,         // Work request
    NEW_OMP_WSH_DO_VAL,                 // DO/FOR loop (unspecified scheduling)
    NEW_OMP_WSH_DO_STATIC_VAL,          // DO/FOR loop (static scheduling)
    NEW_OMP_WSH_DO_DYNAMIC_VAL,         // DO/FOR loop (dynamic scheduling)
    NEW_OMP_WSH_DO_GUIDED_VAL,          // DO/FOR loop (guided scheduling)
    NEW_OMP_WSH_DO_RUNTIME_VAL,         // DO/FOR loop (runtime scheduling)
    NEW_OMP_WSH_DO_ORDERED_STATIC_VAL,  // DO/FOR ordered loop (static scheduling)
    NEW_OMP_WSH_DO_ORDERED_DYNAMIC_VAL, // DO/FOR ordered loop (dynamic scheduling)
    NEW_OMP_WSH_DO_ORDERED_GUIDED_VAL,  // DO/FOR ordered loop (guided scheduling)
    NEW_OMP_WSH_DO_ORDERED_RUNTIME_VAL, // DO/FOR ordered loop (runtime scheduling)
    NEW_OMP_WSH_DOACROSS_STATIC_VAL,    // DOACROSS loop (static scheduling)
    NEW_OMP_WSH_DOACROSS_DYNAMIC_VAL,   // DOACROSS loop (dynamic scheduling)
    NEW_OMP_WSH_DOACROSS_GUIDED_VAL,    // DOACROSS loop (guided scheduling)
    NEW_OMP_WSH_DOACROSS_RUNTIME_VAL,   // DOACROSS loop (runtime scheduling)
    NEW_OMP_WSH_SECTION_VAL,            // SECTION
    NEW_OMP_WSH_SINGLE_VAL,             // SINGLE
    NEW_OMP_WSH_MASTER_VAL,             // MASTER

  // NEW_OMP_SYNC_EV
    NEW_OMP_BARRIER_VAL,                // #pragma omp barrier
    NEW_OMP_JOIN_WAIT_VAL,              // Implicit join at the end of parallels (wait)
    NEW_OMP_JOIN_NOWAIT_VAL,            // Implicit join at the end of parallels (nowait)
    NEW_OMP_LOCK_ATOMIC_VAL,            // #pragma omp atomic
    NEW_OMP_LOCK_CRITICAL_VAL,          // #pragma omp critical (unnamed)
    NEW_OMP_LOCK_CRITICAL_NAMED_VAL,    // #pragma omp critical (named)
    NEW_OMP_ORDERED_VAL,                // #pragma omp ordered
    NEW_OMP_TASKGROUP_VAL,              // #pragma omp taskgroup
    NEW_OMP_TASKWAIT_VAL,               // #pragma omp taskwait
    NEW_OMP_POST_VAL,                   // DOACROSS post
    NEW_OMP_WAIT_VAL,                   // DOACROSS wait

  // NEW_OMP_LOCK_EV
    NEW_OMP_LOCK_REQUEST_VAL,           // Requesting mutex lock
    NEW_OMP_LOCK_TAKEN_VAL,             // Mutex is locked
    NEW_OMP_LOCK_RELEASE_REQUEST_VAL,   // Requesting mutex unlock

  // NEW_OMP_ORDERED_EV
    NEW_OMP_ORDERED_WAIT_START_VAL,     // Wait for required data
    NEW_OMP_ORDERED_WAIT_OVER_VAL,      // Enter ordered region when data is ready and wait is over
    NEW_OMP_ORDERED_POST_START_VAL,     // Send signal that required data is ready

  // NEW_OMP_TASKGROUP_EV
    NEW_OMP_TASKGROUP_OPENING_VAL,      // Opening taskgroup
    NEW_OMP_TASKGROUP_ENTERING_VAL,     // Enter taskgroup region
    NEW_OMP_TASKGROUP_WAITING_VAL,      // Wait child completion

  // NEW_OMP_TASKING_EV
    NEW_OMP_TASK_INST_VAL,              // Task instantiation
    NEW_OMP_TASK_EXEC_VAL,              // Task execution
    NEW_OMP_TASKLOOP_INST_VAL,          // Taskloop instantiation
    NEW_OMP_TASKLOOP_EXEC_VAL           // Taskloop execution
};

