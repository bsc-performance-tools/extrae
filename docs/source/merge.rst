.. _cha:Merging:

Merging process
===============

Once the application has finished, and if the automatic merge process is not
setup, the merge must be executed manually. Here we detail how to run the merge
process manually.

The inserted probes in the instrumented binary are responsible for gathering
performance metrics of each task/thread and for each of them several files are
created where the XML configuration file specified (see section
:ref:`sec:XMLSectionStorage`). Such files are:

* As many ``.mpit`` files as tasks and threads were running the target
  application. Each file contains information gathered by the specified
  task/thread in raw binary format.
* A single ``.mpits`` file that contain a list of related ``.mpit`` files.
* If the DynInst based instrumentation package was used, an addition ``.sym``
  file that contains some symbolic information gathered by the DynInst library.

In order to use |PARAVER|, those intermediate files (*i.e.*, ``.mpit`` files)
must be merged and translated into |PARAVER| trace file format. The same applies
if the user wants to use the |DIMEMAS| simulator. To proceed with any of these
translations, all the intermediate trace files must be merged into a single
trace file using one of the available mergers in the ``bin`` directory (see
:numref:`tab:MergerDescription`).

The target trace type is defined in the XML configuration file used at the
instrumentation step (see section :ref:`sec:XMLSectionTraceConfiguration`), and
it has to match with the used merger (``mpi2prv`` and ``mpimpi2prv`` for
|PARAVER| and ``mpi2dim`` and ``mpimpi2dim`` for |DIMEMAS|). However, it is
possible to force the format nevertheless the selection done in the XML file
using the parameters ``-paraver`` or ``-dimemas`` [#TIMING]_.

.. tabularcolumns:: |l|p{10cm}|

.. table:: Description of the available mergers in the |TRACE| package
  :name: tab:MergerDescription

  +-----------------------+---------------------------------------------+
  | :command:`mpi2prv`    | Sequential version of the |PARAVER| merger. |
  +-----------------------+---------------------------------------------+
  | :command:`mpi2dim`    | Sequential version of the |DIMEMAS| merger. |
  +-----------------------+---------------------------------------------+
  | :command:`mpimpi2prv` | Parallel version of the |PARAVER| merger.   |
  +-----------------------+---------------------------------------------+
  | :command:`mpimpi2dim` | Parallel version of the |DIMEMAS| merger.   |
  +-----------------------+---------------------------------------------+


.. _sec:ParaverMerger:

|PARAVER| Merger
----------------

As stated before, there are two |PARAVER| mergers: ``mpi2prv`` and
``mpimpi2prv``. The former is for use in a single processor mode, while the
latter is meant to be used with multiple processors using MPI (and cannot be run
using one MPI task).

|PARAVER| merger receives a set of intermediate trace files and generates three
files with the same name (which is set with the :option:`-o` option) but differ
in the extension. The |PARAVER| trace itself (.prv file) that contains
timestamped records that represent the information gathered during the execution
of the instrumented application. It also generates the |PARAVER| Configuration
File (.pcf file), which is responsible for translating values contained in the
|PARAVER| trace into a more human readable values. Finally, it also generates a
file containing the distribution of the application across the cluster
computation resources (.row file).


.. _subsec:SequentialParaverMerger:

Sequential |PARAVER| Merger
^^^^^^^^^^^^^^^^^^^^^^^^^^^

These are the available options for the sequential |PARAVER| merger:

.. option:: -d, -dump

  Dumps the information stored in the intermediate trace files.

.. option:: -dump-without-time

  The information dumped with :option:`-d` or :option:`-dump` does not show 
  the timestamp.

.. option:: -e <BINARY>

  Uses the given <BINARY> to translate addresses that are stored in the
  intermediate trace files into useful information (including function name,
  source file and line). The application has to be compiled with :option:`-g`
  flag so as to obtain valuable information.

  .. note::
    Since |TRACE| version 2.4.0 this flag is superseded in Linux systems where
    :file:`/proc/self/maps` is readable. The instrumentation part will annotate
    the binaries and shared libraries in use and will try to use them before
    using <BINARY>. This flag is still available in Linux systems as a default
    case just in case the binaries and libraries pointed by
    :file:`/proc/self/maps` are not available.

.. option:: -emit-library-events

  Emit additional events for the source code references that belong to a
  separate shared library that cannot be translated. Only add information with
  respect to the shared library name. This option is disabled by default.

.. option:: -evtnum <N>

  Partially processes (up to <N> events) the intermediate trace files to generate
  the |DIMEMAS| tracefile.

.. option:: -f <FILE.mpits>

  *(where <FILE.mpits> file is generated by the instrumentation)*

  The merger uses the given file (which contains a list of intermediate trace
  files of a single executions) instead of giving set of intermediate trace
  files.

  This option looks first for each file listed in the parameter file. Each
  contained file is searched in the absolute given path, if it does not exist,
  then it's searched in the current directory.

.. option:: -f-relative <FILE.mpits>

  *(where <FILE.mpits> file is generated by the instrumentation)*

  This options behaves like the :option:`-f` options but looks for the
  intermediate files in the current directory.

.. option:: -f-absolute <FILE.mpits>

  *(where <FILE.mpits> file is generated by the instrumentation)*

  This options behaves like the :option:`-f` options but uses the full path of
  every intermediate file so as to locate them.

.. option:: -h

  Provides minimal help about merger options.

.. option:: -keep-mpits, -no-keep-mpits

  Tells the merger to keep (or remove) the intermediate files after the trace
  generation.

.. option:: -maxmem <M>

  The last step of the merging process will be limited to use ``<M>`` megabytes
  of memory. By default, ``<M>`` is 512.

.. option:: -s <FILE.sym>

  *(where <FILE.sym> file is generated with the Dyninst instrumentator)*

  Passes information regarding instrumented symbols into the merger to aid the
  |PARAVER| analysis. If :option:`-f`, :option:`-f-relative` or
  :option:`-f-absolute` paramters are given, the merge process will try to
  automatically load the symbol file associated to that ``<FILE.mpits>`` file.

.. option:: no-syn

  If set, the merger will not attempt to synchronize the different tasks. This
  is useful when merging intermediate files obtained from a single node (and
  thus, share a single clock).

.. option:: -o <FILE.prv[.gz]>

  Choose the name of the target |PARAVER| tracefile, can be compressed with the
  libz library. If :option:`-o` is not
  given, the merging process will automatically name the tracefile using the
  application binary name, if possible.

.. option:: -remove-files

  The merging process removes the intermediate tracefiles when succesfully
  generating the |PARAVER| tracefile.

.. option:: -skip-sendrecv

  Do not match point to point communications issued by ``MPI_Sendrecv`` or
  ``MPI_Sendrecv_replace``.

.. option:: -sort-addresses

  Sort event values that reference source code locations so as the values are
  sorted by file name first and then line number (enabled by default).

.. option:: -split-states

  Do not join consecutive states that are the same into a single one.

.. option:: -syn

  If different nodes are used in the execution of a tracing run, there can exist
  some clock differences on all the nodes. This option makes :option:`mpi2prv`
  to recalculate all the timings based on the end of the ``MPI_Init`` call. This
  will usually lead to "synchronized" tasks, but it will depend on how the
  clocks advance in time.

.. option:: -syn-node

  If different nodes are used in the execution of a tracing run, there can exist
  some clock differences on all the nodes. This option makes :option:`mpi2prv`
  to recalculate all the timings based on the end of the ``MPI_Init`` call and
  the node where they ran. This will usually lead to better synchronized tasks
  than using :option:`-syn`, but, again, it will depend on how the clocks
  advance in time.

.. option:: -translate-addresses, -no-trace-overwrite

  Tells the merger to overwrite (or not) the final tracefile if it already
  exists. If the tracefile exists and :option:`-no-trace-overwrite` is given,
  the tracefile name will have an increasing numbering in addition to the name
  given by the user.

.. option:: -unique-caller-id

  Choose whether use a unique value identifier for different callers locations
  (MPI calling routines, user routines, OpenMP outlined routines and pthread
  routines).


.. _subsec:ParallelParaverMerger:

Parallel |PARAVER| Merger
^^^^^^^^^^^^^^^^^^^^^^^^^

These options are specific to the parallel version of the |PARAVER| merger:

.. option:: -block

  Intermediate trace files will be distributed in a block fashion instead of a
  cyclic fashion to the merger.

.. option:: -cyclic

  Intermediate trace files will be distributed in a cyclic fashion instead of a
  block fashion to the merger.

.. option:: -size

  The intermediate trace files will be sorted by size and then assigned to
  processors in a such manner that each processor receives approximately the
  same size.

.. option:: -consecutive-size

  Intermediate trace files will be distributed consecutively to processors but
  trying to distribute the overall size equally among processors.

.. option:: -use-disk-for-comms

  Use this option if your memory resources are limited. This option uses an
  alternative matching communication algorithm that saves memory but uses
  intensively the disk.

.. option:: -tree-fan-out <N>

  Use this option to instruct the merger to generate the tracefile using a
  tree-based topology. This should improve the performance when using a large
  number of processes at the merge step. Depending on the combination of
  processes and the width of the tree, the merger will need to run several
  stages to generate the final tracefile.

  The number of processes used in the merge process must be equal or greater
  than the <N> parameter. If it is not, the merger itself will automatically
  set the width of the tree to the number of processes used.


.. _sec:DimemasMerger:

|DIMEMAS| merger
----------------

As stated before, there are two |DIMEMAS| mergers: :command:`mpi2dim` and
:command:`mpimpi2dim`. The former is for use in a single processor mode while
the latter is meant to be used with multiple processors using MPI.

In contrast with |PARAVER| merger, |DIMEMAS| mergers generate a single output
file with the ``.dim`` extension that is suitable for the |DIMEMAS| simulator from
the given intermediate trace files.

These are the available options for both |DIMEMAS| mergers:

.. option:: -evtnum <N>

  Partially processes (up to <N> events) the intermediate trace files to
  generate the |DIMEMAS| tracefile.

.. option:: -f <FILE.mpits>

  *(where <FILE.mpits> file is generated by the instrumentation)*

  The merger uses the given file (which contains a list of intermediate trace
  files of a single executions) instead of giving set of intermediate trace
  files.

  This option takes only the file name of every intermediate file so as to
  locate them.

.. option:: -f-relative <FILE.mpits>

  *(where <FILE.mpits> file is generated by the instrumentation)*

  This options works exactly as the :option:`-f` option.

.. option:: -f-absolute <FILE.mpits>

  *(where <FILE.mpits> file is generated by the instrumentation)*

  This options behaves like the :option:`-f` option but uses the full path of
  every intermediate file so as to locate them.

.. option:: -h

  Provides minimal help about merger options.

.. option:: -maxmem <M>

  The last step of the merging process will be limited to use ``<M>`` megabytes
  of memory. By default, M is 512.

.. option:: -o <FILE.dim>

  Choose the name of the target |DIMEMAS| tracefile.


.. _sec:MergerEnvVars:

Environment variables
---------------------

There are some environment variables that are related to the mergers:


.. _subsec:ParaverMergerEnvVars:

Environment variables suitable to the |PARAVER| merger
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. envvar:: EXTRAE_LABELS

Lets the user add custom information to the generated |Paraver| Configuration
File (``.pcf``). Just set this variable to point to a file containing labels for
the unknown (user) events.

The format for the file is:

.. code-block:: none

  EVENT_TYPE
  0 [type1] [label1]
  0 [type2] [label2]
  ...
  0 [typeK] [labelK]

Where ``[typeN]`` is the event value and ``[labelN]`` is the description for the
event with value ``[typeN]``.

It is also possible to link both type and value of an event:

.. code-block:: none

  EVENT_TYPE
  0 [type] [label]
  VALUES
  [value1] [label1]
  [value2] [label2]
  ...
  [valueN] [labelN]

With this information, |PARAVER| can deal with both type and value when giving
textual information to the end user. If |PARAVER| does not find any information
for an event/type it will shown it in numerical form.


.. envvar:: MPI2PRV_TMP_DIR

Points to a directory where all intermediate temporary files will be stored.

These files will be removed as soon the application ends.


.. _subsec:DimemasMergerEnvVars:

Environment variables suitable to the |DIMEMAS| merger
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. envvar:: MPI2DIM_TMP_DIR

Points to a directory where all intermediate temporary files will be stored.

These files will be removed as soon the application ends.


.. rubric:: Footnotes

.. [#TIMING] The timing mechanism differ in |PARAVER|/|DIMEMAS| at the
  instrumentation level. If the output trace format does not correspond with that
  selected in the XML some timing inaccuracies may be present in the final
  tracefile. Such inaccuracies are known to be higher due to clock granularity if
  the XML is set to obtain |DIMEMAS| tracefiles but the resulting tracefile is
  forced to be in |PARAVER| format.
