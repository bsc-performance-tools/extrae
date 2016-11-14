.. _cha:QuickStart:

Quick start guide
=================


.. _sec:QuickStartInstrumentationPackage:

The instrumentation package
---------------------------


.. _subsec:QuickStartInstrumentationPackageUncompress:

Uncompressing the package
^^^^^^^^^^^^^^^^^^^^^^^^^

|TRACE| is a dynamic instrumentation package to trace programs compiled and run
with the shared memory model (like OpenMP and pthreads), the message passing
(MPI) programming model or both programming models (different MPI processes
using OpenMP or pthrads within each MPI process). |TRACE| generates trace files
that can be later visualized with |PARAVER|.

The package is distributed in compressed tar format (*e.g.,* extrae.tar.gz). To
unpack it, execute from the desired target directory the following command:

.. code-block:: sh

  gunzip -c extrae.tar.gz | tar -xvf -

The unpacking process will create different directories on the current directory
(see :numref:`tab:PackageDescription`).

.. tabularcolumns:: |l|p{10cm}|

.. table:: Package contents description
  :name: tab:PackageDescription

  +---------------+-----------------------------------------------------------+
  | **Directory** | **Contents**                                              |
  +===============+===========================================================+
  | bin           | Contains the binary files of the |TRACE| tool.            |
  +---------------+-----------------------------------------------------------+
  | etc           | Contains some scripts to set up environment variables and |
  |               | the |TRACE| internal files.                               |
  +---------------+-----------------------------------------------------------+
  | lib           | Contains the |TRACE| tool libraries.                      |
  +---------------+-----------------------------------------------------------+
  | share/man     | Contains the |TRACE| manual entries.                      |
  +---------------+-----------------------------------------------------------+
  | share/doc     | Contains the |TRACE| manuals (pdf, ps and html versions). |
  +---------------+-----------------------------------------------------------+
  | share/example | Contains examples to illustrate the |TRACE|               |
  |               | instrumentation.                                          |
  +---------------+-----------------------------------------------------------+


.. _sec:QuickStartQuickRunning:

Quick running
-------------

.. note::

  There are several included examples in the installation package. These
  examples are installed in :file:`${EXTRAE_HOME}/share/example` and cover
  different application types (including serial/MPI/OpenMP/CUDA/*etc.*). We
  suggest the user to look at them to get an idea on how to instrument their
  application.

Once the package has been unpacked, set the :envvar:`EXTRAE_HOME` environment
variable to the directory where the package was installed. Use the
:command:`export` or :command:`setenv` commands to set it up depending on the
shell you use. If you use sh-based shell (like :command:`sh`, :command:`bash`,
:command:`ksh`, :command:`zsh`, *...*), issue this command:

.. code-block:: sh

  export EXTRAE_HOME=<DIR>

however, if you use :command:`csh`-based shell (like :command:`csh`,
:command:`tcsh`), execute the following command:

.. code-block:: csh

  setenv EXTRAE_HOME <DIR>

where `<DIR>` refers where |TRACE| was installed.

.. note:: 
  Henceforth, all references to the usage of the environment variables will be
  used following the :command:`sh` format unless specified.

|TRACE| is offered in two different flavors: as a DynInst-based application, or
stand-alone application. DynInst is a dynamic instrumentation library that
allows the injection of code in a running application without the need to
recompile the target application. If the DynInst instrumentation library is not
installed, |TRACE| also offers different mechanisms to trace applications.


.. _subsec:RunningTraceDynInst:

Quick running |TRACE| - based on DynInst
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|TRACE| needs some environment variables to be setup on each session. Issuing
the command:

.. code-block:: sh

  source ${EXTRAE_HOME}/etc/extrae.sh

on a :command:`sh`-based shell, or

.. code-block:: sh

  source ${EXTRAE_HOME}/etc/extrae.csh

on a :command:`csh`-based shell will do the work. Then copy the default XML
configuration file [#QUICKXML]_ into the working directory by executing:

.. code-block:: sh

  cp ${EXTRAE_HOME}/share/example/MPI/extrae.xml .

If needed, set the application environment variables as usual (like
:envvar:`OMP_NUM_THREADS`, for example), and finally launch the application
using the :command:`${EXTRAE_HOME}/bin/extrae` instrumenter like:

.. code-block:: sh

  ${EXTRAE_HOME}/bin/extrae -config extrae.xml <PROGRAM>

where `<PROGRAM>` is the application binary.


.. _subsec:RunningTraceNOTDynInst:

Quick running |TRACE| - NOT based on DynInst
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|TRACE| needs some environment variables to be setup on each session. Issuing
the command:

.. code-block:: sh

  source ${EXTRAE_HOME}/etc/extrae.sh

on a :command:`sh`-based shell, or:

.. code-block:: sh

  source ${EXTRAE_HOME}/etc/extrae.csh

on a :command:`csh`-based shell will do the work. Then copy the default XML
configuration file [#QUICKXML]_ into the working directory by executing:

.. code-block:: sh

  cp ${EXTRAE_HOME}/share/example/MPI/extrae.xml .

and export :envvar:`EXTRAE_CONFIG_FILE` as:

.. code-block:: sh

  export EXTRAE_CONFIG_FILE=extrae.xml

If needed, set the application environment variables as usual (like
:envvar:`OMP_NUM_THREADS`, for example). Just before executing the target
application, issue the following command:

.. code-block:: sh

  export LD_PRELOAD=${EXTRAE_HOME}/lib/<LIB>

where `<LIB>` is one of the libraries listed in :numref:`tab:AvailableExtraeLIBS`.

.. tabularcolumns:: |l|c|c|c|c|c|c|c|c|c|

.. table:: Available libraries in |TRACE|. Their availability depends upon the
  configure process.
  :name: tab:AvailableExtraeLIBS

  +------------------------------+----------+-------+----------+-----------+---------+---------------+--------+----------+--------+
  | **Library**                  | **Application type**                                                                           |
  +==============================+==========+=======+==========+===========+=========+===============+========+==========+========+
  |                              | *Serial* | *MPI* | *OpenMP* | *pthread* | *SMPss* | *nanos/OMPss* | *CUDA* | *OpenCL* | *Java* |
  +------------------------------+----------+-------+----------+-----------+---------+---------------+--------+----------+--------+
  | libseqtrace                  | Yes      |       |          |           |         |               |        |          |        |
  +------------------------------+----------+-------+----------+-----------+---------+---------------+--------+----------+--------+
  | libmpitrace [#FORTRAN]_      |          | Yes   |          |           |         |               |        |          |        |
  +------------------------------+----------+-------+----------+-----------+---------+---------------+--------+----------+--------+
  | libomptrace                  |          |       | Yes      |           |         |               |        |          |        |
  +------------------------------+----------+-------+----------+-----------+---------+---------------+--------+----------+--------+
  | libpttrace                   |          |       |          | Yes       |         |               |        |          |        |
  +------------------------------+----------+-------+----------+-----------+---------+---------------+--------+----------+--------+
  | libsmpsstrace                |          |       |          |           | Yes     |               |        |          |        |
  +------------------------------+----------+-------+----------+-----------+---------+---------------+--------+----------+--------+
  | libnanostrace                |          |       |          |           |         | Yes           |        |          |        |
  +------------------------------+----------+-------+----------+-----------+---------+---------------+--------+----------+--------+
  | libcudatrace                 |          |       |          |           |         |               | Yes    |          |        |
  +------------------------------+----------+-------+----------+-----------+---------+---------------+--------+----------+--------+
  | libocltrace                  |          |       |          |           |         |               |        | Yes      |        |
  +------------------------------+----------+-------+----------+-----------+---------+---------------+--------+----------+--------+
  | javaseqtrace.jar             |          |       |          |           |         |               |        |          | Yes    |
  +------------------------------+----------+-------+----------+-----------+---------+---------------+--------+----------+--------+
  | libompitrace [#FORTRAN]_     |          | Yes   | Yes      |           |         |               |        |          |        |
  +------------------------------+----------+-------+----------+-----------+---------+---------------+--------+----------+--------+
  | libptmpitrace [#FORTRAN]_    |          | Yes   |          | Yes       |         |               |        |          |        |
  +------------------------------+----------+-------+----------+-----------+---------+---------------+--------+----------+--------+
  | libsmpssmpitrace [#FORTRAN]_ |          | Yes   |          |           | Yes     |               |        |          |        |
  +------------------------------+----------+-------+----------+-----------+---------+---------------+--------+----------+--------+
  | libnanosmpitrace [#FORTRAN]_ |          | Yes   |          |           |         | Yes           |        |          |        |
  +------------------------------+----------+-------+----------+-----------+---------+---------------+--------+----------+--------+
  | libcudampitrace [#FORTRAN]_  |          | Yes   |          |           |         |               | Yes    |          |        |
  +------------------------------+----------+-------+----------+-----------+---------+---------------+--------+----------+--------+
  | libcudaompitrace [#FORTRAN]_ |          | Yes   | Yes      |           |         |               | Yes    |          |        |
  +------------------------------+----------+-------+----------+-----------+---------+---------------+--------+----------+--------+
  | liboclmpitrace [#FORTRAN]_   |          | Yes   |          |           |         |               |        | Yes      |        |
  +------------------------------+----------+-------+----------+-----------+---------+---------------+--------+----------+--------+


.. _sec:QuickMerging:

Quick merging the intermediate traces
-------------------------------------

Once the intermediate trace files (:file:`*.mpit` files) have been created, they
have to be merged (using the :command:`mpi2prv` command) in order to generate
the final |PARAVER| trace file. Execute the following command to proceed with
the merge:

.. code-block:: sh

  ${EXTRAE_HOME}/bin/mpi2prv -f TRACE.mpits -o output.prv

The result of the merge process is a |PARAVER| tracefile called
:file:`output.prv`. If the :option:`-o` option is not given, the resulting
tracefile is called :file:`EXTRAE_Paraver_Trace.prv`. 



.. rubric:: Footnotes

.. [#QUICKXML] See section :ref:`cha:XML` for further details regarding this 
    file.

.. [#FORTRAN] If the application is Fortran append an f to the library. For
   example, if you want to instrument a Fortran application that is using MPI,
   use ``libmpitracef`` instead of ``libmpitrace``.
