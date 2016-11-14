.. _cha:InstrumentedRoutines:

Instrumented routines
=====================


.. _sec:MPIinstrumentedroutines:

MPI
---

These are the instrumented MPI routines in the |TRACE| package:

* MPI_Init
* MPI_Init_thread [#MPISUPPORT]_
* MPI_Finalize
* MPI_Bsend
* MPI_Ssend
* MPI_Rsend
* MPI_Send
* MPI_Bsend_init
* MPI_Ssend_init
* MPI_Rsend_init
* MPI_Send_init
* MPI_Ibsend
* MPI_Issend
* MPI_Irsend
* MPI_Isend
* MPI_Recv
* MPI_Irecv
* MPI_Recv_init
* MPI_Reduce
* MPI_Ireduce
* MPI_Reduce_scatter
* MPI_Ireduce_scatter
* MPI_Allreduce
* MPI_Iallreduce
* MPI_Barrier
* MPI_Ibarrier
* MPI_Cancel
* MPI_Test
* MPI_Wait
* MPI_Waitall
* MPI_Waitany
* MPI_Waitsome
* MPI_Bcast
* MPI_Ibcast
* MPI_Alltoall
* MPI_Ialltoall
* MPI_Alltoallv
* MPI_Ialltoallv
* MPI_Allgather
* MPI_Iallgather
* MPI_Allgatherv
* MPI_Iallgatherv
* MPI_Gather
* MPI_Igather
* MPI_Gatherv
* MPI_Igatherv
* MPI_Scatter
* MPI_Iscatter
* MPI_Scatterv
* MPI_Iscatterv
* MPI_Comm_rank
* MPI_Comm_size
* MPI_Comm_create
* MPI_Comm_free
* MPI_Comm_dup
* MPI_Comm_split
* MPI_Comm_spawn
* MPI_Comm_spawn_multiple
* MPI_Cart_create
* MPI_Cart_sub
* MPI_Start
* MPI_Startall
* MPI_Request_free
* MPI_Scan
* MPI_Iscan
* MPI_Sendrecv
* MPI_Sendrecv_replace
* MPI_File_open [#MPIIOSUPPORT]_
* MPI_File_close [#MPIIOSUPPORT]_
* MPI_File_read [#MPIIOSUPPORT]_
* MPI_File_read_all [#MPIIOSUPPORT]_
* MPI_File_write [#MPIIOSUPPORT]_
* MPI_File_write_all [#MPIIOSUPPORT]_
* MPI_File_read_at [#MPIIOSUPPORT]_
* MPI_File_read_at_all [#MPIIOSUPPORT]_
* MPI_File_write_at [#MPIIOSUPPORT]_
* MPI_File_write_at_all [#MPIIOSUPPORT]_
* MPI_Get [#MPIRMASUPPORT]_
* MPI_Put [#MPIRMASUPPORT]_
* MPI_Win_complete [#MPIRMASUPPORT]_
* MPI_Win_create [#MPIRMASUPPORT]_
* MPI_Win_fence [#MPIRMASUPPORT]_
* MPI_Win_free [#MPIRMASUPPORT]_
* MPI_Win_post [#MPIRMASUPPORT]_
* MPI_Win_start [#MPIRMASUPPORT]_
* MPI_Win_wait [#MPIRMASUPPORT]_

* MPI_Probe
* MPI_Iprobe
* MPI_Testall
* MPI_Testany
* MPI_Testsome
* MPI_Request_get_status
* MPI_Intercomm_create
* MPI_Intercomm_merge



.. _sec:OpenMPruntimesinstrumented:

OpenMP
------


.. _subsec:openmpruntimesintel:

Intel compilers - icc, iCC, ifort
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The instrumentation of the Intel OpenMP runtime for versions 8.1 to 10.1 is only
available using the |TRACE| package based on DynInst library.

These are the instrument routines of the Intel OpenMP runtime functions using
DynInst:

* __kmpc_fork_call
* __kmpc_barrier
* __kmpc_invoke_task_func
* __kmpc_set_lock [#OMPLOCKS]_
* __kmpc_unset_lock [#OMPLOCKS]_

The instrumentation of the Intel OpenMP runtime for version 11.0 to 12.0 is
available using the |TRACE| package based on the :envvar:`LD_PRELOAD` and also
the DynInst mechanisms. The instrumented routines include:

* __kmpc_fork_call
* __kmpc_barrier
* __kmpc_dispatch_init_4
* __kmpc_dispatch_init_8
* __kmpc_dispatch_next_4
* __kmpc_dispatch_next_8
* __kmpc_dispatch_fini_4
* __kmpc_dispatch_fini_8
* __kmpc_single
* __kmpc_end_single
* __kmpc_critical [#OMPLOCKS]_
* __kmpc_end_critical [#OMPLOCKS]_
* omp_set_lock [#OMPLOCKS]_
* omp_unset_lock [#OMPLOCKS]_
* __kmpc_omp_task_alloc
* __kmpc_omp_task_begin_if0
* __kmpc_omp_task_complete_if0
* __kmpc_omp_taskwait


.. _subsec:openmpruntimesibm:

IBM compilers - xlc, xlC, xlf
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|TRACE| supports IBM OpenMP runtime 1.6.

These are the instrumented routines of the IBM OpenMP runtime:

* _xlsmpParallelDoSetup_TPO
* _xlsmpParRegionSetup_TPO
* _xlsmpWSDoSetup_TPO
* _xlsmpBarrier_TPO
* _xlsmpSingleSetup_TPO
* _xlsmpWSSectSetup_TPO
* _xlsmpRelDefaultSLock [#OMPLOCKS]_
* _xlsmpGetDefaultSLock [#OMPLOCKS]_
* _xlsmpGetSLock [#OMPLOCKS]_
* _xlsmpRelSLock [#OMPLOCKS]_


.. _subsec:openmpruntimesgnu:

GNU compilers - gcc, g++, gfortran
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|TRACE| supports GNU OpenMP runtime 4.2 and 4.9.

These are the instrumented routines of the GNU OpenMP runtime:

* GOMP_parallel_start
* GOMP_parallel_sections_start
* GOMP_parallel_end
* GOMP_sections_start
* GOMP_sections_next
* GOMP_sections_end
* GOMP_sections_end_nowait
* GOMP_loop_end
* GOMP_loop_end_nowait
* GOMP_loop_static_start
* GOMP_loop_dynamic_start
* GOMP_loop_guided_start
* GOMP_loop_runtime_start
* GOMP_loop_ordered_static_start
* GOMP_loop_ordered_dynamic_start
* GOMP_loop_ordered_guided_start
* GOMP_loop_ordered_runtime_start
* GOMP_loop_static_next
* GOMP_loop_dynamic_next
* GOMP_loop_guided_next
* GOMP_loop_runtime_next
* GOMP_parallel_loop_static_start
* GOMP_parallel_loop_dynamic_start
* GOMP_parallel_loop_guided_start
* GOMP_parallel_loop_runtime_start
* GOMP_barrier
* GOMP_critical_start [#OMPLOCKS]_
* GOMP_critical_end [#OMPLOCKS]_
* GOMP_critical_name_start [#OMPLOCKS]_
* GOMP_critical_name_end [#OMPLOCKS]_
* GOMP_atomic_start [#OMPLOCKS]_
* GOMP_atomic_end [#OMPLOCKS]_
* GOMP_task
* GOMP_taskwait

* GOMP_parallel
* GOMP_taskgroup_start
* GOMP_taskgroup_end


.. sec:pthreadinstrumentedroutines:

pthread
-------

These are the instrumented routines of the pthread runtime:

* pthread_create
* pthread_detach
* pthread_join
* pthread_exit
* pthread_barrier_wait
* pthread_mutex_lock
* pthread_mutex_trylock
* pthread_mutex_timedlock
* pthread_mutex_unlock

.. pthread_cond_* routines seem to be not instrumentable. the application hangs
  when instrumenting them
  * pthread_cond_signal
  * pthread_cond_broadcast
  * pthread_cond_wait
  * pthread_cond_timedwait

* pthread_rwlock_rdlock
* pthread_rwlock_tryrdlock
* pthread_rwlock_timedrdlock
* pthread_rwlock_wrlock
* pthread_rwlock_trywrlock
* pthread_rwlock_timedwrlock
* pthread_rwlock_unlock


.. sec:CUDAinstrumentedroutines:

CUDA
----

These are the instrumented CUDA routines in the |TRACE| package:

* cudaLaunch
* cudaConfigureCall
* cudaThreadSynchronize
* cudaStreamCreate
* cudaStreamSynchronize
* cudaMemcpy
* cudaMemcpyAsync
* cudaDeviceReset
* cudaDeviceSynchronize
* cudaThreadExit

The CUDA accelerators do not have memory for the tracing buffers, so the tracing
buffer resides in the host side.

Typically, the CUDA tracing buffer is flushed at ``cudaThreadSynchronize``,
``cudaStreamSynchronize`` and ``cudaMemcpy`` calls, so it is possible that the
tracing buffer for the device gets filled if no calls to this routines are
executed.


.. sec:OPENCLinstrumentedroutines:

OpenCL
------

These are the instrumented OpenCL routines in the |TRACE| package:

* clBuildProgram
* clCompileProgram
* clCreateBuffer
* clCreateCommandQueue
* clCreateContext
* clCreateContextFromType
* clCreateKernel
* clCreateKernelsInProgram
* clCreateProgramWithBinary
* clCreateProgramWithBuiltInKernels
* clCreateProgramWithSource
* clCreateSubBuffer
* clEnqueueBarrierWithWaitList
* clEnqueueBarrier
* clEnqueueCopyBuffer
* clEnqueueCopyBufferRect
* clEnqueueFillBuffer
* clEnqueueMarkerWithWaitList
* clEnqueueMarker
* clEnqueueMapBuffer
* clEnqueueMigrateMemObjects
* clEnqueueNativeKernel
* clEnqueueNDRangeKernel
* clEnqueueReadBuffer
* clEnqueueReadBufferRect
* clEnqueueTask
* clEnqueueUnmapMemObject
* clEnqueueWriteBuffer
* clEnqueueWriteBufferRect
* clFinish
* clFlush
* clLinkProgram
* clSetKernelArg
* clWaitForEvents
* clRetainCommandQueue
* clReleaseCommandQueue
* clRetainContext
* clReleaseContext
* clRetainDevice
* clReleaseDevice
* clRetainEvent
* clReleaseEvent
* clRetainKernel
* clReleaseKernel
* clRetainMemObject
* clReleaseMemObject
* clRetainProgram
* clReleaseProgram

The OpenCL accelerators have small amounts of memory, so the tracing buffer
resides in the host side.

Typically, the accelerator tracing buffer is flushed at each ``cl_Finish``
call, so it is possible that the tracing buffer for the accelerator gets filled
if no calls to this routine are executed.

However if the operated OpenCL command queue is tagged as not Out-of-Order, then
flushes will also happen at ``clEnqueueReadBuffer``, ``clEnqueueReadBufferRect``
and ``clEnqueueMapBuffer`` if their corresponding blocking parameter is set to
true.



.. rubric:: Footnotes

.. [#MPISUPPORT] The MPI library must support this routine

.. [#MPIIOSUPPORT] The MPI library must support MPI/IO routines

.. [#MPIRMASUPPORT] The MPI library must support 1-sided (or RMA -remote memory address-)
  routines

.. [#OMPLOCKS] The instrumentation of OpenMP locks can be enabled/disabled
