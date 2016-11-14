.. _cha:pnmpi:

Running |TRACE| on top of PnMPI
===============================

.. _sec:pnmpiIntroduction:

Introduction
------------

Most tools targeting MPI rely on the MPI Profiling Interface (PMPI), which
allows tools to transparently intercept invocations to MPI routines and with
that to establish wrappers around MPI calls to gather execution information.
However, the usage of this interface is limited to a single tool. PnMPI
eliminates the restriction of a single PMPI tool layer per execution. It can
dynamically load and chain multiple PMPI tools into a single tool stack and then
interject this complete stack between the target application and the library
without changing the view for each individual tool. It enables the user to
combine arbitrary MPI tools without having to reimplement them. When |TRACE| is
operating through :envvar:`LD_PRELOAD` interposition it also supports to run on
top of PnMPI.


.. _sec:pnmpiInstructions:

Instructions to run with PnMPI
------------------------------

|TRACE| tracing libraries have to be processed with the :command:`patch` tool
that comes included with PnMPI. Just run this utility, passing as argument the
tracing library that you want to load under the PnMPI environment and the output
name for the patched library.

.. code-block:: sh

  $PNMPI_HOME/patch/patch libmpitrace.so libmpitrace-pnmpi.so

At execution time the :envvar:`PNMPI_CONF` environment variable has to be
defined, pointing to a file that specifies all the tools that will be loaded
with PnMPI. 

.. code-block:: sh

  export PNMPI_CONF=$PNMPI_HOME/demo/.pnmpi-conf

In this file we have to add the patched tracing library at the beginning of the
list.

.. code-block:: sh

  module libmpitrace-pnmpi
  module another-tool
  ...
