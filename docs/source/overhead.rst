.. _cha:Overhead:

Overhead
========

|TRACE| includes a set of tests to evaluate the overhead imposed to the
application by different components.

These tests are installed in :file:`${EXTRAE_HOME}/share/tests/overhead` and can
be run by executing the :file:`run_overhead_tests.sh` script within this
directory.

Note that this script compiles and executes the generated binaries on the same
system, so this script will require some tuning to run in a system that uses a
batch-queuing system and/or needs cross-compiling.

Currently there are the following tests the evaluate the necessary time to
perform certain operations:

* **posix_clock**
  grab the current time using the posix clock. Even the simpler emitted event
  requires gathering a timestamp.

* **extrae_event**
  emit one event (without performance counters) into the tracing buffer using
  the :ref:`Extrae_event <func:Extrae_event>` API call.

* **extrae_nevent4**
  emit four events (without performance counters) into the tracing buffer using
  the :ref:`Extrae_nevent4 <func:Extrae_nevent>` API call.

* **extrae_eventandcounters**
  emit one event (and reading 4 peformance counters) into the tracing buffer
  through the :ref:`Extrae_eventandcounters <func:Extrae_eventandcounters>` call.

* **papi_read1**
  capture the value of one performance counter through PAPI.

* **papi_read4**
  capture the value of four performance counters through PAPI.

* **extrae_user_function**
  involves traversing the processor call-stack while searching the frame that
  points to the current routine (as the :ref:`Extrae_user_function
  <func:Extrae_user_function>` API call).

* **extrae_get_caller1**
  traverses one level of the processor call-stack.

* **extrae_get_caller6**
  traverses six levels of the processor call-stack.

* **extrae_trace_callers**
  collects three frames from the processor call-stack.

* **extrae_event_Java**
  measures the time required to emit one event (without performance counters)
  from Java through the JNI connector.

* **extrae_nevent4_Java**
  measures the time needed to emit four events (without performance counters)
  from Java through the JNI connector.

:numref:`fig:overheads` depicts the overhead of |TRACE| |release| in the
following systems:

* System based on ``Intel Xeon E5649`` (*Nehalem*) processors.
  |TRACE| was compiled with support for libunwind 1.1 and PAPI 5.0.1.

* System based on ``Intel Xeon E5-2670`` (*SandyBridge*) processors.
  |TRACE| was compiled with support for libunwind 1.1, PAPI 5.4.1 and IBM's
  Java7.

* System based on ``Intel Xeon E5-2680`` (*Haswell*) processors.
  |TRACE| was compiled with support for libunwind 1.1 and PAPI 5.4.1 and
  OpenJDK's Java 1.8.

* System based on ``IBM Power8``.
  |TRACE| was compiled with support for libunwind (downloaded from GIT) and PAPI
  5.4.1.

* System based on ``Cortex-A15`` (*Samsung Exynos 5*).
  |TRACE| was compiled with support for libunwind (downloaded from GIT) and PAPI
  5.4.1.

The reader may notice that the ARM processor requires more time to execute the
tests than the rest, even for the simpler cases (``posix_clock`` and
``extrae_event``). The Power8-based system takes a similar amount of time than
Intel-based systems except for the call-stack traversal. Within Intel-based
systems, the *SandyBridge* processor reduced the time significantly from the
*Nehalem* processor but the *Haswell* does not show a great reduction from
*SandyBridge*.

.. _fig:overheads:

.. figure:: overheads/overheads.eps
  :align: center

  Overhead result in a variety of systems for |TRACE| |release|
