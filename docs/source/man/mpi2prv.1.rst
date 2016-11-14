:orphan:

.. _mpi2prv(1):


mpi2prv
=======

mpi2dim - MPI merge facility for the Dimemas visualizer tool

mpimpi2prv - Parallel MPI merge facility for the Paraver visualizer tool

mpimpi2dim - Parallel MPI merge facility for the Dimemas visualizer tool

SYNOPSIS
--------

**mpi2prv** [**-h**] *ifile1* **...** [**--**] *ifileN* [**-o** *ofile*]

**mpi2dim** [**-h**] *ifile1* **...** [**--**] *ifileN* [**-o** *ofile*]

**mpirun ... mpi2prv** [**-h**] *ifile1* **...** [**--**] *ifileN* [**-o** *ofile*]

**mpirun ... mpi2dim** [**-h**] *ifile1* **...** [**--**] *ifileN* [**-o** *ofile*]


DESCRIPTION
-----------

**mpi2prv** and **mpi2dim** are two different utilities to generate traces from
the intermediate files written by |TRACE| instrumentation tool for the |PARAVER|
visualizer and the |DIMEMAS| simulator, respectively. The merge process usually
takes several mpit files and produces a reduced set of files depending of the
desired output. For |PARAVER| traces (**mpi2prv**)), the merge process will
generate .prv(.gz), .pcf and .row files, while for |DIMEMAS| traces
(**mpi2dim**), the merge process will generate .dim, .pcf, and .row files. If
the machine supports MPI applications the parallel versions of these mergers are
also available (**mpimpi2prv** and **mpimpi2dim**).


OPTIONS
-------

**--**
  the next trace file is considered a different parallel task.

**-block**
  MPIT files will be distributed in a block fashion instead of in a cyclic
  fashion to the parallel |PARAVER| merger.

  **Only valid for mpimpi2prv**.

**-cyclic**
  MPIT files will be distributed in a cyclic fashion instead of in a block
  fashion to the parallel |PARAVER| merger.

  **Only valid for mpimpi2prv**.

**-size**
  MPIT files will be distributed by size in the parallel |PARAVER| merger.

  **Only valid for mpimpi2prv**.

**-consecutive-size**
  MPIT files will be distributed by size but preserving the given order in the
  parallel |PARAVER| merger.

  **Only valid for mpimpi2prv**.

**-d**
  Sequentially dumps the contents of every intermediate file. This  is  very
  useful for debugging purposes.

**-dimemas**
  Forces the generation of a |DIMEMAS| tracefile regardless of the information
  contained on the intermediate trace files. Be careful when using this option
  because timing will **NOT** be accurate.

**-e** *file*
  If the intermediate file contains information about MPI callers, this option
  will search for information (function name, source file and line) of that
  caller in the given *file* (MPI application). The application must be compiled
  with the **-g** flag to obtain all the details for the callers.

**-extended-glop-info**
  Each global operation (MPI_Barrier, MPI_Alltoall, MPI_Reduce, ...) will
  generate additional information records on the tracefile. Such information
  usually contains: size of the global operation, who was the master, and the
  communicator used.

**-f** *file*, **-f-absolute** *file*, **-f-relative** *file*
  File with the names of the mpit files to merge. Files are searched with the
  absulute path and, if not found, in the current directory. The **f-absolute**
  and **f-relative** options only look for files using the absolute or the
  current path respectively.

**-h**
  Displays a simple help.

**-o** *file*
  Output file name. If the file exists, it will be overwritten.

**-paraver**
  Forces the generation of a |PARAVER| tracefile regardless of the information
  contained in the intermediate files. Be careful when using this option because
  timing will **NOT** be accurate.

**-s** *file*
  Symbol file for the mpit files. This is a good place to put information about
  the manually added events.

**-split-states**
  Do not join consecutive states that are the same into a single one.

**-syn**, **-syn-node**, **-no-syn**
  If different nodes are used in the execution of a tracing run, there can exist
  some clock differences on all the nodes. This option makes **mpi2prv** to
  recalculate all the timings based on the end of the **MPI_Init** call, and the
  node where they ran if **syn-node** is used. This will usually lead to
  synchronized tasks, but it will depend on how the clocks advance in time.

  It is not a perfect solution where all the clocks are different, but it can
  give a first impression on how the run worked.

  **Only applies to |PARAVER| traces. Use with caution. mpi2prv uses a simple
  but effective method to determine if this option must be applied.**

**-xyzt**
  On the BG/L system each task can be mapped in different points of a torus
  network. This option will produce a file with the coordinates on the network
  of each task.


ENVIRONMENT
-----------

mpi2prv uses the following environment variables:

**EXTRAE_LABELS**
  Location of the file with user custom information to add to the generated
  Paraver Configuration File (pcf). The format of this file is:

  | 
  | EVENT_TYPE
  | 0 [type1] [label1]
  | 0 [type2] [label2]
  | ...
  | 0 [typeK] [labelK]

  Where [typeN] is the event value and [labelN] is the description for the event
  with value [typeN]. It is also possible to link both, type and value, of an
  event:

  | 
  | EVENT_TYPE
  | 0 [type] [label]
  | VALUES
  | [value1] [label1]
  | [value2] [label2]
  | ...
  | [valueK] [labelK]

  With this information |PARAVER| can deal with both, type and value, when
  giving textual information to the end user. If |PARAVER| does not find any
  information for an event/type it will show it in numerical form.

**MPI2PRV_TMP_DIR**
  Location of the directory where all intermediate temporal files will be
  stored. These files will be removed as soon as the application ends.


EXAMPLES
--------

Merge all the intermediate files into a |PARAVER| trace file:

  $ mpi2prv \*.mpit -o out.prv

Merge all the intermediate files into a |DIMEMAS| trace file:

  $ mpi2dim \*.mpit -o out.dim

Merge all the intermediate files into a compressed |PARAVER| trace file:

  $ mpi2prv \*.mpit -o out.prv.gz

Merge all the intermediate files for a run on different nodes into a compressed
|PARAVER| trace file:

  $ mpi2prv -syn \*.mpit -o out.prv.gz

Merge all the intermediate files for a run on different nodes into a compressed
|PARAVER| trace file with additional information about MPI global operations and
MPI calls:

  $ mpi2prv -syn -extended-glop-info -e ./program -o out.prv.gz


REPORTING_BUGS
--------------

If you find any bug in the documentation or in the software, pelase send a
descriptive mail to: **tools@bsc.es**

SEE ALSO
--------
:manpage:`extrae(1)`

:manpage:`extrae_event(3)`, :manpage:`extrae_counters(3)`,
:manpage:`extrae_eventandcounters(3)`, :manpage:`extrae_shutdown(3)`,
:manpage:`extrae_restart(3)`, :manpage:`extrae_set_tracing_tasks(3)`,
:manpage:`extrae_set_options(3)`,
