:orphan:

.. _extrae_event(3):
.. _extrae_counters(3):
.. _extrae_eventandcounters(3):
.. _extrae_shutdown(3):
.. _extrae_restart(3):
.. _extrae_set_tracing_tasks(3):
.. _extrae_set_options(3):


extrae_event
============

SYNOPSIS
--------

C/C++ API:
^^^^^^^^^^

#include <extrae_user_events.h>

void Extrae_event (unsigned int type, unsigned int value);

void Extrae_counters (void);

void Extrae_eventandcounters (unsigned int type, unsigned int value);

void Extrae_shutdown (void);

void Extrae_restart (void);

void Extrae_set_tracing_tasks (unsigned int from, unsigned int to);

void Extrae_set_options (int options);

Extrae_previous_hwc_set ();

Extrae_next_hwc_set ();


Fortran API:
^^^^^^^^^^^^

extrae_event (INTEGER type, INTEGER value)

extrae_counters ()

extrae_eventandcounters (INTEGER type, INTEGER value)

extrae_shutdown ()

extrae_restart ()

extrae_set_tracing_tasks (INTEGER from, INTEGER to)

extrae_set_options (INTEGER options)

extrae_previous_hwc_set ()

extrae_next_hwc_set ()


DESCRIPTION
-----------

|TRACE| instruments MPI routines by default, but it also allows manual
instrumentation of the user code using the provided API.

**Extrae_event**
  Add a single timestamped event into the tracefile. The event has two
  arguments: type and value.

  Some common use of events are:

  * Identify loop iterations (or any code block): Given a loop, the user can set
    a unique type for the loop and a value related to the iterator value of the
    loop. For example:

  .. code-block:: c
  
    for (i = 0; i <= MAX_LOOP; i++)
    {
      Extrae_event (1000, i);
      
      [loop code]
    }
    Extrae_event (1000, 0);

  The last added call to Extrae_event marks the end of the loop, setting the
  event value to 0, which facilitates the analysis with |PARAVER|.

  * Identify user routines: Choosing a constant type (60000019 is a common
    choice in other tracing tools) and different values for different routines
    (set to 0 to mark a "leave" event). For example:

  .. code-block:: c

    void
    routine1 (void)
    {
      Extrae_event (6000019, 1);
      [routine 1 code]
      Extrae_event (6000019, 0);
    }

    void
    routine2 (void)
    {
      Extrae_event (6000019, 2);
      [routine 2 code]
      Extrae_event (6000019, 0);
    }

  * Identify any point in the application using a unique combination of type and
    value.

**Extrae_counters**
  Obtain information of the processor performance counters. The counters are
  those pointed by EXTRAE_COUNTERS (see :manpage:`extrae(1)`).

**Extrae_eventandcounters**
  Add an event and obtain the performance counters with a single call. All the
  information will have exactly the same timestamp.

**Extrae_shutdown**
  Stop the tracing, can be restarted invoking **Extrae_restart**.

**Extrae_set_tracing_tasks**
  Change which tasks emit information to the intermediate tracefiles. All tasks
  are traced by default.

**Extrae_set_options**
  Changes the internal behaviour of the tracing facility. Use the bitwise OR
  operator (| in C, IOR in Fortran) to specify multiple options. Avalable
  options are:

  EXTRAE_DISABLE_ALL_OPTIONS
    Disable EVERYTHING but the user events inserted manually on the source code.

  EXTRAE_CALLER_OPTION
    Every MPI call will emit information of the routine that invoked that
    invoked them.

  EXTRAE_HWC_OPTION
    Emit information about the hardware counters.

  EXTRAE_MPI_HWC_OPTION
    Emit hardware counters information on every call to MPI.

  EXTRAE_MPI_OPTION
    Emit information of MPI calls.

  EXTRAE_ENABLE_ALL_OPTIONS
    Enable EVERYTHING (default value).

**Extrae_next_hwc**, **Extrae_previous_hwc_set** 
  Change the hardware counter set if multiple sets are specified in the XML
  configuration file.


REPORTING BUGS
--------------

If you find any bug in the documentation or in the software, pelase send a
descriptive mail to: **tools@bsc.es**


SEE ALSO
--------

:manpage:`mpi2prv(1)`, :manpage:`extrae(1)`
