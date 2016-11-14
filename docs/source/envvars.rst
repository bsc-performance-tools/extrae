.. _cha:envvars:

Environment variables
=====================

Althouth |TRACE| is configured through an XML file (which is pointed by
:envvar:`EXTRAE_CONFIG_FILE`), it also supports minimal configuration via
environment variables for those systems that do not have the library responsible
for parsing the XML files (*i.e.*, libxml2).

This appendix presents the environment variables |TRACE| package uses if
:envvar:`EXTRAE_CONFIG_FILE` is not set and their description. For those
environment variables that refer to XML :option:`enabled` attributes (*i.e.*,
that can be set to "yes" or "no") are considered to be enabled if their value is
defined to 1.

.. envvar:: EXTRAE_BUFFER_SIZE

Sets the number of records that the instrumentation buffer can hold before
flushing them.

.. envvar:: EXTRAE_COUNTERS

See section :ref:`subsec:ProcessorPerformanceCounters`. Just one set can be
defined. Counters (in PAPI) groups (in PMAPI) are given separated by commas.

.. envvar:: EXTRAE_CONTROL_FILE

The instrumentation will be enabled only when the pointed file exists.

.. envvar:: EXTRAE_CONTROL_GLOPS

Starts the instrumentation when the specified number of global collectives have
been executed.

.. envvar:: EXTRAE_CONTROL_TIME

Checks the file pointed by :envvar:`EXTRAE_CONTROL_FILE` at this period.

.. envvar:: EXTRAE_DIR

Specifies where temporal files will be created during instrumentation.

.. envvar:: EXTRAE_DISABLE_MPI

Disables MPI instrumentation.

.. envvar:: EXTRAE_DISABLE_OMP

Disables OpenMP instrumentation.

.. envvar:: EXTRAE_DISABLE_PTHREAD

Disables pthread instrumentation.

.. envvar:: EXTRAE_FILE_SIZE

Sets the maximum size (in Mbytes) for the intermediate trace file.

.. envvar:: EXTRAE_FUNCTIONS

List of routines to be instrumented, as described in section
:ref:`sec:XMLSectionUF`, using the GNU C compiler
:option:`-fininstrument-funtions` option, or the IBM XL compiler
:option:`-qdebug=function_trace` option at compile and link time.

.. envvar:: EXTRAE_FUNCTIONS_COUNTERS_ON

Specifies if the performance counters should be collected when a user function
event is emitted.

.. envvar:: EXTRAE_FINAL_DIR

Specifies where files will be stored when the application ends.

.. envvar:: EXTRAE_GATHER_MPITS

Gathers intermediate trace files into a single directory. Only available when
instrumenting MPI applications.

.. envvar:: EXTRAE_HOME

Points where |TRACE| is installed.

.. envvar:: EXTRAE_INITIAL_MODE

Chooses whether the instrumentation runs in :option:`detail` or in
:option:`bursts` mode.

.. envvar:: EXTRAE_BURST_THRESHOLD

Specifies the threshold time to filter running bursts.

.. envvar:: EXTRAE_MINIMUM_TIME

Specifies the minimum amount of instrumentation time.

.. envvar:: EXTRAE_MPI_CALLER

Chooses which MPI calling routines should be dumped to the tracefile.

.. envvar:: EXTRAE_MPI_COUNTERS_ON

Set to 1 if MPI must report performance counter values.

.. envvar:: EXTRAE_MPI_STATISTICS

Set to 1 if basic MPI statistics must be collected in burst mode. Only available
in systems with Myrinet GM/MX networks.

.. envvar:: EXTRAE_NETWORK_COUNTERS

Set to 1 to dump network performance counters values.

.. envvar:: EXTRAE_PTHREAD_COUNTERS_ON

Set to 1 if pthread must report performance counters values.

.. envvar:: EXTRAE_OMP_COUNTERS_ON

Set to 1 if OpenMP must report performance counters values.

.. envvar:: EXTRAE_PTHREAD_LOCKS

Set to 1 if pthread locks have to be instrumented.

.. envvar:: EXTRAE_OMP_LOCKS

Set to 1 if OpenMP locks have to be instrumented.

.. envvar:: EXTRAE_ON

Enables instrumentation.

.. envvar:: EXTRAE_PROGRAM_NAME

Specifies the prefix of the resulting intermediate trace files.

.. envvar:: EXTRAE_SAMPLING_CALLER

Determines the callstack segment stored through time-sampling capabilities.

.. envvar:: EXTRAE_SAMPLING_CLOCKTYPE

Determines domain for sampling clock. Options are: ``DEFAULT``, ``REAL``,
``VIRTUAL`` and ``PROF``.

.. envvar:: EXTRAE_SAMPLING_PERIOD

Enables time-sampling capabilities with the indicated period.

.. envvar:: EXTRAE_SAMPLING_VARIABILITY

Adds some variability to the sampling period.

.. envvar:: EXTRAE_RUSAGE

Instrumentation emits resource usage at flush points if set to 1.

.. envvar:: EXTRAE_SKIP_AUTO_LIBRARY_INITIALIZE

Do not init instrumentation automatically in the main symbol.

.. envvar:: EXTRAE_TRACE_TYPE

Chooses whether the resulting tracefiles are intended for |PARAVER| or
|DIMEMAS|.
