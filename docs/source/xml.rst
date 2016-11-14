.. _cha:XML:

|TRACE| XML configuration file
==============================

|TRACE| is configured through a XML file that is set through the
:envvar:`EXTRAE_CONFIG_FILE` environment variable. The included examples provide
several XML files to serve as a basis for the end user. For instance, the MPI
examples provide four XML configuration files:

* :file:`extrae.xml` Exemplifies all the options available to set up in the
  configuration file. We will discuss below all the sections and options
  available. It is also available on this document on appendix
  :ref:`cha:wholeXML`.
* :file:`extrae_explained.xml` The same as the above with some comments on each
  section.
* :file:`summarized_trace_basic.xml` A small example for gathering information
  of MPI and OpenMP information with some performace counters and calling
  information at each MPI call.
* :file:`detailed_trace_basic.xml` A small example for gathering a summarized
  information of MPI and OpenMP parallel paradigms.
* :file:`extrae_bursts_1ms.xml` An XML configuration example to setup the
  bursts tracing mode. This XML file will only capture the regions in between
  MPI calls that last more than the given threshold (1ms in this example).

Please note that most of the nodes present in the XML file have an
:option:`enabled` attribute that allows turning on and off some parts of the
instrumentation mechanism. For example, ``<mpi enabled="yes">`` means MPI
instrumentation is enabled and process all the contained XML subnodes, if any;
whether ``<mpi enabled="no">`` means to skip gathering MPI information and do
not process XML subnodes.

Each section points which environment variables could be used if the tracing
package lacks XML support. See appendix :ref:`cha:EnvVars` for the entire list.

Sometimes the XML tags are used for time selection (duration, for instance). In
such tags, the following postfixes can be used: ``n`` or ``ns`` for nanoseconds,
``u`` or ``us`` for microseconds, ``m`` or ``ms`` for milliseconds, ``s`` for
seconds, ``M`` for minutes, ``H`` for hours and ``D`` for days.


.. _sec:XMLSectionTraceConfiguration:

XML Section: Trace configuration
--------------------------------

The basic trace behavior is determined in the first part of the XML and
*contains* all of the remaining options. It looks like:

.. highlight:: xml

.. literalinclude:: xml/config.xml

The ``<?xml version='1.0'?>`` is mandatory for all XML files. Don't touch this.
The available tunable options are under the ``<trace>`` node:

* :option:`enabled` Set to ``yes`` if you want to generate tracefiles.
* :option:`home` Set to where the instrumentation package is installed. Usually
  it points to the same location that :envvar:`EXTRAE_HOME` environment
  variable.
* :option:`initial-mode` Available options

  * :option:`detail` Provides detailed information of the tracing.
  * :option:`bursts` Provides summarized information of the tracing. This mode
    removes most of the information present in the detailed traces (like OpenMP
    and MPI calls among others) and only produces information for computation
    bursts.

* :option:`type` Available options

  * :option:`paraver` The intermediate files are meant to generate |PARAVER|
    tracefiles.
  * :option:`dimemas` The intermediate files are meant to generate |DIMEMAS|
    tracefiles.

.. seealso::

  :envvar:`EXTRAE_ON`, :envvar:`EXTRAE_HOME`, :envvar:`EXTRAE_INITIAL_MODE` and
  :envvar:`EXTRAE_TRACE_TYPE` environment variables in appendix :ref:`cha:EnvVars`.


.. _sec:XMLSectionMPI:

XML Section: MPI
----------------

The MPI configuration part is nested in the config file (see section
:ref:`sec:XMLSectionTraceConfiguration`) and its nodes are the following:

.. highlight:: xml

.. literalinclude:: xml/mpi.xml

MPI calls can gather performance information at the begin and end of MPI calls.
To activate this behavior, just set to ``yes`` the attribute of the nested
``<counters>`` node.

.. seealso::

  :envvar:`EXTRAE_DISABLE_MPI` and :envvar:`EXTRAE_MPI_COUNTERS_ON`
  environment variables in appendix :ref:`cha:EnvVars`.


.. _sec:XMLSectionPThread:

XML Section: pthread
--------------------

The pthread configuration part is nested in the config file (see section
:ref:`sec:XMLSectionTraceConfiguration`) and its nodes are the following:

.. highlight:: xml

.. literalinclude:: xml/pthread.xml

The tracing package allows to gather information of some pthread routines. In
addition to that, the user can also enable gathering information of locks and
also gathering performance counters in all of these routines. This is achieved
by modifying the enabled attribute of the ``<locks>`` and ``<counters>``,
respectively.

.. seealso::

  :envvar:`EXTRAE_DISABLE_PTHREAD`, :envvar:`EXTRAE_PTHREAD_LOCKS` and :envvar:`
  EXTRAE_PTHREAD_COUNTERS_ON` environment variables in appendix
  :envvar:`cha:EnvVars`.


.. _sec:XMLSectionOpenMP:

XML Section: OpenMP
-------------------

The OpenMP configuration part is nested in the config file (see section
:ref:`sec:XMLSectionTraceConfiguration`) and its nodes are the following:

.. highlight:: xml

.. literalinclude:: xml/openmp.xml

The tracing package allows to gather information of some OpenMP runtimes and
outlined routines. In addition to that, the user can also enable gathering
information of locks and also gathering performance counters in all of these
routines. This is achieved by modifying the enabled attribute of the
``<locks>`` and ``<counters>``, respectively.

.. seealso:: 

  :envvar:`EXTRAE_DISABLE_OMP`, :envvar:`EXTRAE_OMP_LOCKS` and
  :envvar:`EXTRAE_OMP_COUNTERS_ON` environment variables in appendix
  :ref:`cha:EnvVars`.


.. _sec:XMLSectionCallers:

XML Section: Callers
--------------------

.. highlight:: xml

.. literalinclude:: xml/callers.xml

Callers are the routine addresses present in the process stack at any given
moment during the application run. Callers can be used to link the tracefile
with the source code of the application.

The instrumentation library can collect a partial view of those addresses during
the instrumentation. Such collected addresses are translated by the merging
process if the correspondent parameter is given and the application has been
compiled and linked with debug information.

There are three points where the instrumentation can gather this information:

* Entry of MPI calls
* Sampling points *(if sampling is available in the tracing package)*
* Dynamic memory calls (malloc, free, realloc)

The user can choose which addresses to save in the trace (starting from 1, which
is the closest point to the MPI call or sampling point) specifying several stack
levels by separating them by commas or using the hyphen symbol.

.. seealso::

  :envvar:`EXTRAE_MPI_CALLER` environment variable in appendix
  :ref:`cha:EnvVars`.


.. _sec:XMLSectionUF:

XML Section: User functions
---------------------------

.. highlight:: xml

.. literalinclude:: xml/userfunctions.xml

The file contains a list of functions to be instrumented by |TRACE|. There are
different alternatives to instrument application functions, and some
alternatives provides additional flexibility, as a result, the format of the
list varies depending of the instrumentation mechanism used:

* DynInst
  Supports instrumentation of  user functions, outer loops, loops and basic
  blocks.
  The given list contains the desired function names to be instrumented. After
  each function name, optionally you can define different basic blocks or loops
  inside the desired function always by providing different suffixes that are
  provided after the ``+`` character. For instance:

  * To instrument the entry and exit points of foo function just provide the
    function name (``foo``).
  * To instrument the entry and exit points of foo function plus the entry and
    exit points of its outer loop, suffix the function name with ``outerloops``
    (*i.e.,* ``foo+outerloops``).
  * To instrument the entry and exit points of foo function plus the entry and
    exit points of its N-th loop function you have to suffix it as ``loop_N``,
    for instance ``foo+loop_3``.
  * To instrument the entry and exit points of foo function plus the entry and
    exit points of its N-th basic block inside the function you have to use the
    suffix ``bb_N``, for instance ``foo+bb_5``. In this case, it is also
    possible to specifically ask for the entry or exit point of the basic block
    by additionally suffixing ``_s`` or ``_e``, respectively.

  Additionally, these options can be added by using comas, as in:
  ``foo+outerloops,loop_3,bb_3_e,bb_4_s,bb_5``.

  To discover the instrumentable loops and basic blocks of a certain function
  you can execute the command :command:`${EXTRAE_HOME}/bin/extrae -config
  extrae.xml -decodeBB`, where ``extrae.xml`` is an |TRACE| configuration file
  that provides a list on the user functions attribute that you want to get the
  information.

* GCC and ICC (through :option:`-finstrument-functions`) GNU and Intel compilers
  provide a compile and link flag named :option:`-finstrument-functions` that
  instruments the routines of a source code file that |TRACE| can use. To take
  advantage of this functionality the list of routines must point to a list with
  the format: ``<HEX_addr>#<F_NAME>``, where *<HEX_addr>* refers to the
  hexadecimal address of the function in the binary file (obtained through
  :command:`nm <binary>` and *<F_NAME>* is the name of the function to be
  instrumented. For instance to instrument the routine ``pi_kernel`` from the
  ``pi`` binary we execute :command:`nm` as follows:

  .. code-block:: sh
  
    $ nm -a pi | grep pi_kernel
    00000000004005ed T pi_kernel
  
  and add ``00000000004005ed#pi_kernel`` into the function list.

The :option:`exclude-automatic-functions` attribute is used only by the DynInst
instrumenter. By setting this attribute to ``yes`` the instrumenter will avoid
automatically instrumenting the routines that either call OpenMP outlined
routines (*i.e.,* routines with OpenMP pragmas) or call CUDA kernels.

Finally, in order to gather performance counters in these functions and also in
those instrumented using the ``extrae_user_function`` API call, the node
``counters`` has to be enabled.

.. warning::

  Note that you need to compile your application binary with debugging
  information (typically the :option:`-g` compiler flag) in order to translate
  the captured addresses into valuable information such as: function name, file
  name and line number.

.. seealso::

  :envvar:`EXTRAE_FUNCTIONS` environment variable in appendix
  :ref:`cha:EnvVars`.


.. _sec:XMLSectionPerformanceCounters:

XML Section: Performance counters
---------------------------------

The instrumentation library can be compiled with support for collecting
performance metrics of different components available on the system. These
components include:

* Processor performance counters. Such access is granted by PAPI [#PAPI]_ or
  PMAPI [#PMAPI]_.
* Network performance counters. *(Only available in systems with Myrinet GM/MX
  networks).*
* Operating system accounts.

Here is an example of the counters section in the XML configuration file:

.. highlight:: xml

.. literalinclude:: xml/counters.xml

.. seealso::

  :envvar:`EXTRAE_COUNTERS`, :envvar:`EXTRAE_NETWORK_COUNTERS` and
  :envvar:`EXTRAE_RUSAGE` environment variables in appendix :ref:`cha:EnvVars`.


.. _subsec:ProcessorPerformanceCounters:

Processor performance counters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Processor performance counters are configured in the ``<cpu>`` nodes. The user
can configure many sets in the ``<cpu>`` node using the ``<set>`` node, but
just one set will be used at any given time in a specific task. The ``<cpu>``
node supports the ``starting-set-distribution`` attribute with the following
accepted values:

* :option:`number` (*in range 1..N, where N is the number of configured sets*)
  All tasks will start using the set specified by number.
* :option:`block` Each task will start using the given sets distributed in
  blocks (*i.e.,* if two sets are defined and there are four running tasks:
  tasks 1 and 2 will use set 1, and tasks 3 and 4 will use set 2).
* :option:`cyclic` Each task will start using the given sets distributed
  cyclically (*i.e.,* if two sets are defined and there are four running tasks:
  tasks 1 and 3 will use, and tasks 2 and 4 will use set 2).
* :option:`thread-cyclic` Sets will be distributed cyclically between tasks and
  threads in a task.
* :option:`random` Each task will start using a random set, and also calls
  either to ``Extrae_next_hwc_set`` or ``Extrae_previous_hwc_set`` will change
  to a random set.

Each set contains a list of performance counters to be gathered at different
instrumentation points (see sections :ref:`sec:XMLSectionMPI`,
:ref:`sec:XMLSectionOpenMP` and :ref:`sec:XMLSectionUF`). If the tracing library
is compiled to support PAPI, performance counters must be given using the
canonical name (like PAPI_TOT_CYC and PAPI_L1_DCM), or the PAPI code in
hexadecimal format (like 8000003b and 80000000, respectively) [#SET]_ If the
tracing library is compiled to support PMAPI, only one group identifier can be
given per set [#GROUP]_ and can be either the group name (like pm_basic and
pm_hpmcount1) or the group number (like 6 and 22, respectively).

In the given example (which refers to PAPI support in the tracing library) two
sets are defined. First set will read PAPI_TOT_INS (total instructions),
PAPI_TOT_CYC (total cycles) and PAPI_L1_DCM (1st level cache misses).  Second
set is configured to obtain PAPI_TOT_INS (total instructions), PAPI_TOT_CYC
(total cycles) and PAPI_FP_INS (floating point instructions).

Additionally, if the underlying performance library supports sampling
mechanisms, each set can be configured to gather information (see section
:ref:`sec:XMLSectionCallers`) each time the specified counter reaches a specific
value. The counter that is used for sampling must be present in the set. In the
given example, the first set is enabled to gather sampling information every
100M cycles.

Furthermore, performance counters can be configured to report accounting on
different basis depending on the ``domain`` attribute specified on each set.
Available options are:

* :option:`kernel` Only counts events ocurred when the application is running in
  kernel mode.
* :option:`user` Only counts events ocurred when the application is running in
  user-space mode.
* :option:`all` Counts events independently of the application running mode.

In the given example, first set is configured to count all the events ocurred,
while the second one only counts those events ocurred when the application is
running in user-space mode.

Finally, the instrumentation can change the active set in a manual and an
automatic fashion. To change the active set manually see
:ref:`Extrae_previous_hwc_set <func:extrae_previous_hwc_set>` and
:ref:`Extrae_next_hwc_set <func:extrae_next_hwc_set>` API calls in section
:ref:`sec:BasicAPI`. To change automatically the active set two options are
allowed: based on time and based on application code. The former mechanism
requires adding the attribute ``changeat-time`` and specify the minimum time to
hold the set. The latter requires adding the attribute ``changeat-globalops``
with a value. The tracing library will automatically change the active set when
the application has executed as many MPI global operations as selected in that
attribute. When In any case, if either attribute is set to zero, then the set
will not me changed automatically.


.. _subsec:NetworkPerformanceCounters:

Network performance counters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Network performance counters are only available on systems with Myrinet GM/MX
networks and they are fixed depending on the firmware used. Other systems, like
BG/* may provide some network performance counters, but they are accessed
through the PAPI interface (see section :ref:`sec:XMLSectionPerformanceCounters`
and PAPI documentation).

If ``<network>`` is enabled the network performance counters appear at the end
of the application run, giving a summary for the whole run.


.. _subsec:OperatingSystemAccounting:

Operating system accounting
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Operating system accounting is obtained through the :manpage:`getrusage(2)`
system call when ``<resource-usage>`` is enabled. As network performance
counters, they appear at the end of the application run, giving a summary for
the whole run.


.. _sec:XMLSectionStorage:

XML Section: Storage management
-------------------------------

The instrumentation package can be instructed on what/where/how produce the
intermediate trace files. These are the available options:

.. highlight:: xml

.. literalinclude:: xml/storage.xml

Such options refer to:

* :option:`trace-prefix` Sets the intermediate trace file prefix. Its
  default value is ``TRACE``.
* :option:`size` Let the user restrict the maximum size (in megabytes)
  of each resulting intermediate trace file [#BUFSIZE]_.
* :option:`temporal-directory` Where the intermediate trace files will
  be stored during the execution of the application. By default they are stored
  in the current directory. If the directory does not exist, the instrumentation
  will try to make it.
* :option:`final-directory` Where the intermediate trace files will be
  stored once the execution has been finished. By default they are stored in the
  current directory. If the directory does not exist, the instrumentation will
  try to make it.

.. seealso::

  :envvar:`EXTRAE_PROGRAM_NAME`, :envvar:`EXTRAE_FILE_SIZE`,
  :envvar:`EXTRAE_DIR`, :envvar:`EXTRAE_FINAL_DIR` and
  :envvar:`EXTRAE_GATHER_MPITS` environment variables in appendix
  :ref:`cha:EnvVars`.


.. _sec:XMLSectionBuffer:

XML Section: Buffer management
------------------------------

Modify the buffer management entry to tune the tracing buffer behavior.

.. highlight:: xml

.. literalinclude:: xml/buffer.xml

By, default (even if the enabled attribute is ``no``) the tracing buffer is set
to 500k events. If ``<size>`` is enabled the tracing buffer will be set to the
number of events indicated by this node. If the circular option is enabled, the
buffer will be created as a circular buffer and the buffer will be dumped only
once with the last events generated by the tracing package.

.. seealso::

  :envvar:`EXTRAE_BUFFER_SIZE` environment variable in appendix :ref:`cha:EnvVars`.


.. _sec:XMLSectionTraceControl:

XML Section: Trace control
--------------------------

.. highlight:: xml

.. literalinclude:: xml/trace-control.xml

This section groups together a set of options to limit/reduce the final trace
size. There are three mechanisms which are based on file existence, global
operations executed and external remote control procedures.

Regarding the ``file``, the application starts with the tracing disabled, and it
is turned on when a control file is created. Use the property ``frequency`` to
choose at which frequency this check must be done. If not supplied, it will be
checked every 100 global operations on MPI_COMM_WORLD.

If the ``global-ops`` tag is enabled, the instrumentation package begins
disabled and starts the tracing when the given number of global operations on
MPI_COMM_WORLD has been executed.

The ``remote-control`` tag section allows to configure some external mechanisms
to automatically control the tracing. Currently, there is only one option which
is built on top of MRNet and it is based on clustering and spectral analysis to
generate a small yet representative trace.

These are the options in the ``mrnet`` tag:

* :option:`target` the approximate requested size for the final trace (in Mb).
* :option:`analysis` one between ``clustering`` and ``spectral``.
* :option:`start-after` number of seconds before the first analysis starts.

The ``clustering`` tag configures the clustering analysis parameters:

* :option:`max_tasks` maximum number of tasks to get samples from.
* :option:`max_points` maximum number of points to cluster.

The ``spectral`` tag section configures the spectral analysis parameters:

* :option:`min_seen` minimum times a given type of period has to be seen to trace a
  sample.
* :option:`max_periods` maximum number of representative periods to trace. 0 equals to
  unlimited.
* :option:`num_iters` number of iterations to trace for every representative period
  found.
* :option:`signals` performance signals used to analyze the application. If not
  specified, ``DurBurst`` is used by default.

.. seealso::

  :envvar:`EXTRAE_CONTROL_FILE`, :envvar:`EXTRAE_CONTROL_GLOPS`,
  :envvar:`EXTRAE_CONTROL_TIME` environment variables in appendix
  :ref:`cha:EnvVars`.


.. _sec:XMLSectionBursts:

XML Section: Bursts
-------------------

.. highlight:: xml

.. literalinclude:: xml/bursts.xml

If the user enables this option, the instrumentation library will just emit
information of computation bursts (*i.e.,* not does not trace MPI calls, OpenMP
runtime, and so on) when the current mode (through initial-mode in section
:ref:`sec:XMLSectionTraceConfiguration`) is set to ``bursts``. The library will
discard all those computation bursts that last less than the selected threshold.

In addition to that, when the tracing library is running in burst mode, it
computes some statistics of MPI activity. Such statistics can be dumped in the
tracefile by enabling ``mpi-statistics``.

.. seealso::

  :envvar:`EXTRAE_INITIAL_MODE`, :envvar:`EXTRAE_BURST_THRESHOLD` and
  :envvar:`EXTRAE_MPI_STATISTICS` environment variables in appendix
  :ref:`cha:EnvVars`.


.. _sec:XMLSectionOthers:

XML Section: Others
-------------------

.. highlight:: xml

.. literalinclude:: xml/others.xml

This section contains other configuration details that do not fit in the
previous sections. At the moment, there are three options to be configured.

* The ``minimum-time`` option indicates the instrumentation package the minimum
  instrumentation time. To enable it, set ``enabled`` to ``yes`` and set the
  minimum time within the ``minimum-time`` tag.
* The option labeled as ``finalize-on-signal`` instructs the instrumentation
  package to listen for different types of signals [#SIGNALS]_ and dump and
  finalize the execution whenever they occur. If a signal occurs but it is not
  configured, then the execution may finish without generating the trace-file.
  *Caveat:* Some MPI implementations use ``SIGUSR1`` and/or ``SIGUSR2``, so if
  you want to capture those signals check first that enabling them do not alter
  with the application execution.
* The ``flush-sampling-buffer-at-instrumentation-point`` lets the user decide
  whether the sampling buffer should be checked for flushing at instrumentation
  points. If this option is not enabled, then the buffer will only be dumped
  once at the end of the application execution.


.. _sec:XMLSectionSampling:

XML Section: Sampling
---------------------

.. highlight:: xml

.. literalinclude:: xml/sampling.xml

This section configures the time-based sampling capabilities. Every sample
contains processor performance counters (if enabled in section
:ref:`subsec:ProcessorPerformanceCounters` and either PAPI or PMAPI are referred
at configure time) and callstack information (if enabled in section
:ref:`sec:XMLSectionCallers` and proper dependencies are set at configure time).

This section contains two attributes besides ``enabled``. These are:

* :option:`type` determines which timer domain is used (see :manpage:`setitimer(2)` or
  :manpage:`setitimer(3p)` for further information on time domains). Available
  options are: ``real`` (which is also the ``default`` value, ``virtual`` and
  ``prof`` (which use the SIGALRM, SIGVTALRM and SIGPROF respectively).  The
  default timing accumulates real time, but only issues samples at master
  thread. To let all the threads to collect samples, the type must be
  ``virtual`` or ``prof``.
* :option:`period` specifies the sampling periodicity. In the example above, samples
  are gathered every 50ms.
* :option:`variability` specifies the variability to the sampling periodicity. Such
  variability is calculated through the ``random()`` system call and then is
  added to the periodicity. In the given example, the variability is set to
  10ms, thus the final sampling period ranges from 45 to 55ms.

.. seealso::

  :envvar:`EXTRAE_SAMPLING_PERIOD`, :envvar:`EXTRAE_SAMPLING_VARIABILITY`,
  :envvar:`EXTRAE_SAMPLING_CLOCKTYPE` and :envvar:`EXTRAE_SAMPLING_CALLER`
  environment variables in appendix :ref:`cha:EnvVars`.


.. _sec:XMLSectionCUDA:

XML Section: CUDA
-----------------

.. highlight:: xml

.. literalinclude:: xml/cuda.xml

This section indicates whether the CUDA calls should be instrumented or not. If
``enabled`` is set to yes, CUDA calls will be instrumented, otherwise they
will not be instrumented.


.. _sec:XMLSectionOPENCL:

XML Section: OpenCL
-------------------

.. highlight:: xml

.. literalinclude:: xml/opencl.xml

This section indicates whether the OpenCL calls should be instrumented or not.
If ``enabled`` is set to yes, Opencl calls will be instrumented, otherwise they
will not be instrumented.


.. _sec:XMLSectionIO:

XML Section: Input/Output
-------------------------

.. highlight:: xml

.. literalinclude:: xml/input-output.xml

This section indicates whether I/O calls (``read`` and ``write``) are meant to
be instrumented. If ``enabled`` is set to yes, the aforementioned calls will be
instrumented, otherwise they will not be instrumented.


.. _sec:XMLSectionDynamicMemory:

XML Section: Dynamic memory
---------------------------

.. highlight:: xml

.. literalinclude:: xml/dynamic-memory.xml

This section indicates whether dynamic memory calls (``malloc``, ``free``,
``realloc``) are meant to be instrumented. If ``enabled`` is set to yes, the
aforementioned calls will be instrumented, otherwise they will not be
instrumented.

This section allows deciding whether allocation and free-related memory calls
shall be instrumented.

Additionally, the configuration can also indicate whether allocation calls
should be instrumented if the requested memory size surpasses a given threshold
(32768 bytes, in the example).


.. _sec:XMLIntelPEBS:

XML Section: Memory references through Intel PEBS sampling
----------------------------------------------------------

.. highlight:: xml

.. literalinclude:: xml/intel-pebs.xml

This section tells |TRACE| to use the PEBS feature from recent Intel processors
[#PEBS]_ to sample memory references. These memory references capture the linear
address referenced, the component of the memory hierarchy that solved the
reference and the number of cycles to solve the reference.

In the example above, PEBS monitors one out of every million load instructions
and only grabs those that require at least 10 cycles to be solved.


.. _sec:XMLSectionMerge:

XML Section: Merge
------------------

.. highlight:: xml

.. literalinclude:: xml/merge.xml

If this section is enabled and the instrumentation package is configured to
support this, the merge process will be automatically invoked after the
application run. The merge process will use all the resources devoted to run the
application.

In the given example, the leaf of this node will be used as the tracefile name
(:file:`mpi_ping.prv``). Current available options for the merge process are
given as attribute of the ``<merge>`` node and they are:

* :option:`synchronization`: which can be set to ``default``, ``node``, ``task``,
  ``no``. This determines how task clocks will be synchronized (*default* is
  node).
* :option:`binary`: points to the binary that is being executed. It will be used to
  translate gathered addresses (MPI callers, sampling points and user functions)
  into source code references.
* :option:`tree-fan-out`: *only for MPI executions* sets the tree-based topology to
  run the merger in a parallel fashion.
* :option:`max-memory`: limits the intermediate merging process to run up to the
  specified limit (in MBytes).
* :option:`joint-states`: which can be set to ``yes``, ``no``. Determines if the
  resulting Paraver tracefile will split or join equal consecutive states
  (*default is ``yes``*).
* :option:`keep-mpits`: whether to keep the intermediate tracefiles after performing
  the merge.
* :option:`sort-addresses`: whether to sort all addresses that refer to the source
  code (enabled by default).
* :option:`overwrite`: set to ``yes`` if the new tracefile can overwrite an existing
  tracefile with the same name. If set to ``no``, then the tracefile will be
  given a new name using a consecutive id.

In Linux systems, the tracing package can take advantage of certain
functionalities from the system and can guess the binary name, and from it the
tracefile name. In such systems, you can use the following reduced XML section
replacing the earlier section.

.. highlight:: xml

.. literalinclude:: xml/merge-reduced.xml

.. seealso::

  For further references, see chapter :ref:`cha:Merging`.


.. _sec:EnvVars_in_XML:

Using environment variables within the XML file
-----------------------------------------------

XML tags and attributes can refer to environment variables that are defined in
the environment during the application run. If you want to refer to an
environment variable within the XML file, just enclose the name of the variable
using the dollar symbol (``$``), for example: {``$FOO$``}.

Note that the user has to put an specific value or a reference to an environment
variable which means that expanding environment variables in text is not allowed
as in a regular shell (i.e., the instrumentation package will not convert the
follwing text {``bar$FOO$bar``).



.. rubric:: Footnotes

.. [#PAPI] More information available on their website
  http://icl.cs.utk.edu/papi. |TRACE| requires at least PAPI 3.x.

.. [#PMAPI] PMAPI is only available for AIX operating system, and it is on the
  base operating system since AIX5.3. |TRACE| requires at least AIX 5.3.

.. [#SET] Some architectures do not allow grouping some performance counters in
  the same set.

.. [#GROUP] Each group contains several performance counters.

.. [#BUFSIZE] This check is done each time the buffer is flushed, so the
  resulting size of the intermediate trace file depends also on the number of
  elements contained in the tracing buffer (see :ref:`sec:XMLSectionBuffer`).

.. [#SIGNALS] See :manpage:`man signal(2)` and :manpage:`man signal(7)` for more
  details.

.. [#PEBS] Check for availability on your system by looking for pebs in
  :file:`/proc/cpuinfo`.
