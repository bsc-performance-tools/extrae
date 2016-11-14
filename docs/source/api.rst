.. _cha:api:

|TRACE| API
===========

There are two levels of the API in the |TRACE| instrumentation package. Basic
API refers to the basic functionality provided and includes emitting events,
source code tracking, changing instrumentation mode and so. Extended API is an
*experimental* addition to provide several of the basic API within single and
powerful calls using specific data structures.


.. _sec:basicapi:

Basic API
---------

The following routines are defined in :file:`${EXTRAE_HOME}/include/extrae.h`
These routines are intended to be called by C/C++ programs. The instrumentation
package also provides bindings for Fortran applications. The Fortran API
bindings have the same name as the C API but honoring the Fortran compiler
function name mangling scheme. To use the API in Fortran applications you must
use the module provided in :file:`${EXTRAE_HOME}/include/extrae_module.f` by
using the language clause ``use``. This module which provides the appropriate
function and constant declarations for |TRACE|.

.. _func:extrae_get_version:

* .. function:: void Extrae_get_version (unsigned \*major, unsigned \*minor, unsigned \*revision)

  Returns the version of the underlying |TRACE| package. Although an application
  may be compiled to a specific |TRACE| library, by using the appropriate shared
  library commands, the application may use a different |TRACE| library.

.. _func:extrae_init:

* .. function:: void Extrae_init (void)

  Initializes the tracing library.

  This routine is called automatically in different circumstances, which
  include:

  * 
    Call to MPI_Init when the appropriate instrumentation library is linked or
    preload with the application.
  * 
    Usage of the DynInst launcher.
  * 
    If either the ``libseqtrace.so``, ``libomptrace.so`` or ``libpttrace.so``
    are linked dynamically or preloaded with the application.

  No major problems should occur if the library is initialized twice, only a
  warning appears in the terminal output noticing the intent of double
  initialization.

.. _func:extrae_init_type_t:

* .. function:: extrae_init_type_t Extrae_is_initialized (void)

  This routine tells whether the instrumentation has been initialized, and if
  so, also which mechanism was the first to initialize it (regular API or MPI
  initialization).

.. _func:extrae_fini:

* .. function:: void Extrae_fini (void)

  Finalizes the tracing library and dumps the intermediate tracing buffers onto
  disk.

.. _func:extrae_event:

* .. function:: void Extrae_event (extrae_type_t type, extrae_value_t value)

  Adds a single timestamped event into the tracefile.

  Some common uses of events are:

  * Identify loop iterations (or any code block):

    Given a loop, the user can set a unique type for the loop and a value
    related to the iterator value of the loop. For example:

    .. code-block:: c

      for (i = 1; i <= MAX_ITERS; i++)
      {
        Extrae_event (1000, i);
        [original loop code]
        Extrae_event (1000, 0);
      }

    The last added call to ``Extrae_event`` marks the end of the loop setting the
    event value to 0, which facilitates the analysis with Paraver.

  * Identify user routines:
  
    Choosing a constant type (6000019 in this example) and different values for
    different routines (set to 0 to mark a "leave" event).

    .. code-block:: c

      void routine1 (void)
      {
        Extrae_event (6000019, 1);
        [routine 1 code]
        Extrae_event (6000019, 0);
      }

      void routine2 (void)
      {
        Extrae_event (6000019, 2);
        [routine 2 code]
        Extrae_event (6000019, 0);
      }

  * Identify any point in the application using a unique combination of type and value.

.. _func:extrae_nevent:

* .. function:: void Extrae_nevent (unsigned count, extrae_type_t \*types, extrae_value_t \*values)

  Allows the user to place *count* events with the same timestamp at the given
  position.

.. _func:extrae_counters:

* .. function:: void Extrae_counters (void)

  Emits the value of the active hardware counters set. See chapter
  :ref:`cha:xml` for further information.

.. _func:extrae_eventandcounters:

* .. function:: void Extrae_eventandcounters (extrae_type_t event, extrae_value_t value)

  This routine lets the user add an event and obtain the performance counters
  with one call and a single timestamp.

.. _func:extrae_neventandcounters:

* .. function:: void Extrae_neventandcounters (unsigned count, extrae_type_t \*types, extrae_value_t \*values)

  This routine lets the user add several events and obtain the performance
  counters with one call and a single timestamp.

.. _func:extrae_define_event_type:

* .. function:: void Extrae_define_event_type (extrae_type_t \*type, char \*description, unsigned \*nvalues, extrae_value_t \*values, char \*\*description_values)

  This routine adds to the Paraver Configuration File human readable information
  regarding type ``type`` and its values ``values``. If no values need to be
  decribed set ``nvalues`` to 0 and also set ``values`` and
  ``description_values`` to NULL.

.. _func:extrae_shutdown:

* .. function:: void Extrae_shutdown (void)

  Turns off the instrumentation.

.. _func:extrae_restart:

* .. function:: void Extrae_restart (void)

  Turns on the instrumentation.

.. _func:extrae_previous_hwc_set:

* .. function:: void Extrae_previous_hwc_set (void)

  Makes the previous hardware counter set defined in the XML file to be the
  active set (see section :ref:`sec:XMLSectionMPI` for further information).

.. _func:extrae_next_hwc_set:

* .. function:: void Extrae_next_hwc_set (void)

  Makes the following hardware counter set defined in the XML file to be the
  active set (see section :ref:`sec:XMLSectionMPI` for further information).

.. _func:extrae_set_tracing_tasks:

* .. function:: void Extrae_set_tracing_tasks (int from, int to)

  Allows the user to choose from which tasks (not *threads*!) store information
  in the tracefile.

.. _func:extrae_set_options:

* .. function:: void Extrae_set_options (int options)

  Permits configuring several tracing options at runtime. The ``options``
  parameter has to be a bitwise or combination of the following options,
  depending on the user's needs:

  * EXTRAE_CALLER_OPTION

    Dumps caller information at each entry or exit point of the MPI routines.
    Caller levels need to be configured at XML (see chapter :ref:`cha:XML`).

  * EXTRAE_HWC_OPTION

    Activates hardware counter gathering.

  * EXTRAE_MPI_OPTION

    Activates tracing of MPI calls.

  * EXTRAE_MPI_HWC_OPTION

    Activates hardware counter gathering in MPI routines.

  * EXTRAE_OMP_OPTION

    Activates tracing of OpenMP runtime or outlined routines.
  
  * EXTRAE_OMP_HWC_OPTION

    Activates hardware counter gathering in OpenMP runtime or outlined routines.
  
  * EXTRAE_UF_HWC_OPTION
  
    Activates hardware counter gathering in the user functions.

.. _func:extrae_network_counters:

* .. function:: void Extrae_network_counters (void)

  Emits the value of the network counters if the system has this capability.
  *(Only available for systems with Myrinet GM/MX networks)*.

.. _func:extrae_network_routes:

* .. function:: void Extrae_network_routes (int task)

  Emits the network routes for an specific ``task``. *(Only available for
  systems with Myrinet GM/MX networks*.

.. _func:extrae_user_function:

* .. function:: unsigned long long Extrae_user_function (unsigned enter)

  Emits an event into the tracefile which references the source code (data
  includes: source line number, file name and function name). If ``enter`` is 0
  it marks an end (*i.e.,* leaving the function), otherwise it marks the
  beginning of the routine. The user must be careful to place the call of this
  routine in places where the code is always executed, being careful not to
  place them inside ``if`` and ``return`` statements. The function returns the
  address of the reference.

  .. code-block:: c

    void routine1 (void)
    {
      Extrae_user_function (1);
      [routine 1 code]
      Extrae_user_function (0);
    }

    void routine2 (void)
    {
      Extrae_user_function (1);
      [routine 2 code]
      Extrae_user_function (0);
    }

  In order to gather performance counters during the execution of these calls,
  the ``user-functions`` tag and its ``counters`` have to be both enabled int
  section :ref:`sec:XMLSectionUF`.

  .. warning::
  
    Note that you need to compile your application binary with debugging
    information (typically the ``-g`` compiler flag) in order to translate the
    captured addresses into valuable information such as function name, file
    name and line number.

.. _func:extrae_flush:

* .. function:: void Extrae_flush (void)

   Forces the calling thread to write the events stored in the tracing buffers
   to disk.


.. _sec:extendedapi:

Extended API
------------

.. warning::

  This API is in experimental stage and it is only available in C. Use it at
  your own risk!

The extended API makes use of two special structures located in
:file:`${PREFIX}/include/extrae_types.h`. The structures are
``extrae_UserCommunication`` and ``extrae_CombinedEvents``. The former is
intended to encode an event that will be converted into a |PARAVER|
communication when its partner equivalent event has found. The latter is used to
generate events containing multiple kinds of information at the same time.

.. code-block:: c

  struct extrae_UserCommunication
  {
    extrae_user_communication_types_t type;
    extrae_comm_tag_t tag;
    unsigned size; /* size_t? */
    extrae_comm_partner_t partner;
    extrae_comm_id_t id;
  };

The structure ``extrae_UserCommunication`` contains the following fields:

* :option:`type`
  Available options are:

  * ``EXTRAE_USER_SEND``, if this event represents a send point.
  * ``EXTRAE_USER_RECV``, if this event represents a receive point.

* :option:`tag`
  The tag information in the communication record.
* :option:`size`
  The size information in the communication record.
* :option:`partner`
  The partner of this communication (receive if this is a send or send if this
  is a receive). Partners (ranging from 0 to N-1) are considered across tasks
  whereas all threads share a single communication queue.
* :option:`id`
  An identifier that is used to match communications between partners.


.. code-block:: c

  struct extrae_CombinedEvents
  {
    /* These are used as boolean values */
    int HardwareCounters;
    int Callers;
    int UserFunction;
    /* These are intended for N events */
    unsigned nEvents;
    extrae_type_t *Types;
    extrae_value_t *Values;
    /* These are intended for user communication records */
    unsigned nCommunications;
    extrae_user_communication_t *Communications;
  };

The structure ``extrae_CombinedEvents`` contains the following fields:

* :option:`HardwareCounters`
  Set to non-zero if this event has to gather hardware performance counters.
* :option:`Callers`
  Set to non-zero if this event has to emit callstack information.
* :option:`UserFunction`
  Available options are:

  * ``EXTRAE_USER_FUNCTION_NONE``, if this event should not provide information
    about user routines.
  * ``EXTRAE_USER_FUNCTION_ENTER``, if this event represents the starting point
    of a user routine.
  * ``EXTRAE_USER_FUNCTION_LEAVE``, if this event represents the ending point of
    a user routine.

* :option:`nEvents`
  Set the number of events given in the ``Types`` and ``Values`` fields.
* :option:`Types`
  A pointer containing ``nEvents`` type that will be stored in the trace.
* :option:`Values`
  A pointer containing ``nEvents`` values that will be stored in the trace.
* :option:`nCommunications`
  Set the number of communications given in the ``Communications`` field.
* :option:`Communications`
  A pointer to ``extrae_UserCommunication`` structures containing
  ``nCommunications`` elements that represent the involved communications.

The extended API contains the following routines:

* .. function:: void Extrae_init_UserCommunication (struct extrae_UserCommunication \*)

  Use this routine to initialize an ``extrae_UserCommunication`` structure.

* .. function:: void Extrae_init_CombinedEvents (struct extrae_CombinedEvents \*)

  Use this routine to initialize an ``extrae_CombinedEvents`` structure.

* .. function:: void Extrae_emit_CombinedEvents (struct extrae_CombinedEvents \*)

  Use this routine to emit to the tracefile the events set in the
  ``extrae_CombinedEvents`` given.

* .. function:: void Extrae_resume_virtual_thread (unsigned vthread)

  This routine changes the thread identifier so as to be ``vthread`` in the
  final tracefile. *Improper use of this routine may result in corrupt
  tracefiles.*

* .. function:: void Extrae_suspend_virtual_thread (void)

  This routine recovers the original thread identifier (given by routines like
  ``pthread_self`` or ``omp_get_thread_num``, for instance).

* .. function:: void Extrae_register_codelocation_type (extrae_type_t t1, extrae_type_t t2, const char\* s1, const char \*s2)

  Registers type ``t2`` to reference user source code location by using its
  address. During the merge phase the ``mpi2prv`` command will assign type
  ``t1`` to the event type that references the user function and to the event
  ``t2`` to the event that references the file name and line location. The
  strings ``s1`` and ``s2`` refers, respectively, to the description of ``t1``
  and ``t2``

* .. function:: void Extrae_register_function_address (void \*ptr, const char \*funcname, const char \*modname, unsigned line)

  By default, the ``mpi2prv`` process uses the binary debugging information to
  translate program addresses into information that contains function name, the
  module name and line. The ``Extrae_register_function_address`` allows
  providing such information by hand during the execution of the instrumented
  application. This function must provide the function name (``funcname``),
  module name (``modname``) and line number for a given address.

* .. function:: void Extrae_register_stacked_type (extrae_type_t type)

  Registers which event types are required to be managed in a stack way whenever
  ``void Extrae_resume_virtual_thread`` or ``void
  Extrae_suspend_virtual_thread`` are called.

* .. function:: void Extrae_set_threadid_function (unsigned (\*threadid_function)(void))

  Defines the routine that will be used as a thread identifier inside the
  tracing facility.

* .. function:: void Extrae_set_numthreads_function (unsigned (\*numthreads_function)(void))

  Defines the routine that will count all the executing threads inside the
  tracing facility.

* .. function:: void Extrae_set_taskid_function (unsigned (\*taskid_function)(void))

  Defines the routine that will be used as a task identifier inside the tracing
  facility.

* .. function:: void Extrae_set_numtasks_function (unsigned (\*numtasks_function)(void))

  Defines the routine that will count all the executing tasks inside the tracing
  facility.

* .. function:: void Extrae_set_barrier_tasks_function (void (\*barriertasks_function)(void))

  Establishes the barrier routine among tasks. It is needed for synchronization
  purposes.


.. _sec:JavaBindings:

Java bindings
-------------

If Java is enabled at configure time, a basic instrumentation library for serial
application based on JNI bindings to |TRACE| will be installed. The current
bindings are within the package ``es.bsc.cepbatools.extrae`` and the following
bindings are provided:

* .. function:: void Init ();

  Initializes the instrumentation package.

* .. function:: void Fini ();

  Finalizes the instrumentation package.

* .. function:: void Event (int type, long value);

  Emits one event into the trace-file with the given pair type-value.

* .. function:: void Eventandcounters (int type, long value);

  Emits one event into the trace-file with the given pair type-value as well as
  read the performance counters.

* .. function:: void nEvent (int types[], long values[]);

  Emits a set of pair type-value at the same timestamp. Note that both arrays
  must be the same length to proceed correctly, otherwise the call ignores the
  call.

* .. function:: void nEventandcounters (int types[], long values[]);

  Emits a set of pair type-value at the same timestamp as well as read the
  performance counters. Note that both arrays must be the same length to proceed
  correctly, otherwise the call ignores the call.

* .. function:: void defineEventType (int type, String description, long[] values, String[] descriptionValues);

  Adds a description for a given event type (through ``type`` and
  ``description`` parameters). If the array ``values`` is non-null,
  then the array ``descriptionValues`` should be the an array of the same
  length and each entry should be a string describing each of the values given
  in ``values``.

* .. function:: void SetOptions (int options);

  This API call changes the behavior of the instrumentation package but none of
  the options currently apply to the Java instrumentation.

* .. function:: void Shutdown();

  Disables the instrumentation until the next call to ``Restart()``.

* .. function:: void Restart();

  Resumes the instrumentation from the previous ``Shutdown()`` call.


.. _subsec:AdvancedJavaBindings:

Advanced Java Bindings
^^^^^^^^^^^^^^^^^^^^^^

Since |TRACE| does not have features to automatically discover the thread
identifier of the threads that run within the virtual machine, there are some
calls that allows to do this manually.

These calls are, however, intended for expert users and should be avoided
whenever possible because their behavior may be highly modified, or even
removed, in future releases.

* .. function:: SetTaskID (int id);

  Tells |TRACE| that this process should be considered as task with identifier
  ``id``. Use this call before invoking ``Init()``.

* .. function:: SetNumTasks (int num);

  Instructs |TRACE| to allocate the structures for ``num`` processes. Use this
  call before invoking ``Init()``.

* .. function:: SetThreadID (int id);

  Instructs |TRACE| that this thread should be considered as thread with
  identifier ``id``.

* .. function:: SetNumThreads (int num);

  Tells |TRACE| that there are ``num`` threads active within this process. Use
  this call before invoking ``Init()``.

* .. function:: Comm (boolean send, int tag, int size, int partner, long id);

  Allows generating communications between two processes. The call emits one of
  the two-point communication part, so it is necessary to invoke it from both
  the sender and the receiver part. The ``send`` parameter determines whether
  this call will act as send or receive message. The ``tag`` and ``size``
  parameters are used to match the communication and their parameters can be
  displayed in |TRACE|. The ``partner`` refers to the communication partner and
  it is identified by its TaskID. The ``id`` is meant for matching purposes but
  cannot be recovered during the analysis with |PARAVER|.


.. _sec:ExtraeCmdLine:

Command-line version
--------------------

|TRACE| incorporates a mechanism to generate trace-files from the command-line
in a very naïve way in order to instrument executions driven by shell-scripted
applications.

The command-line binary is installed in ``${EXTRAE_HOME}/bin/extrae-cmd`` and
supports the following commands:

* :option:`init` <TASKID> <THREADS>

  This command initializes the tracing on the node that executed the command.
  The initialization command receives two parameters (TASKID, THREADS). The
  TASKID parameter gives an task identifier to the following forthcoming events.
  The THREADS parameter indicates how many threads should the task contain.

* :option:`emit` <THREAD-SLOT> <TYPE> <VALUE>

  This command emits an event with the pair TYPE, VALUE into the the thread
  THREAD at the timestamp when the command is invoked.

* :option:`fini`

  This command finalizes the instrumentation using the command-line version.
  Note that this finalization does not automatically call the merge process
  (``mpi2prv``).


.. warning::

  To use these commands, **do not** export neither :envvar:`EXTRAE_ON` nor
  :envvar:`EXTRAE_CONFIG_FILE`, otherwise the behavior of these commands is
  undefined.

The initialization can be executed only once per node, so if you want to
represent multiple tasks you need different tasks.
