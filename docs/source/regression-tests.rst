.. _cha:RegressionTests:

Regression tests
================

|TRACE| includes a battery of regression tests to evaluate whether recent
versions of the instrumentation package keep their compatibility and that new
changes on it have not introduced new faults.

These tests are meant to be executed in the same machine on which |TRACE| was
compiled and they are not intended to support its execution through
batch-queuing systems nor cross-compilation processes.

To invoke the tests, simply run from the terminal the following command:

.. code-block:: sh

  make check

after the configuration and building process. It will automatically invoke all
the tests one after another and will produce several summaries.

These tests are divided into different categories that stress different parts of
|TRACE|.

The current categories tested include, but are not limited to:

* Clock routines
* Instrumentation support

  * Event definition in the PCF from the |TRACE| API
  * pthread instrumentation
  * MPI instrumentation
  * Java instrumentation

* Merging process (*i.e.*, :command:`mpi2prv`)
* Callstack unwinding (either using libunwind library or backtrace)
* Performance hardware counters through PAPI library
* XML parsing through libxml2

These tests may change during the development of |TRACE|.

If the reader has a particular suggestion on a particular test, please consider
sending it to tools@bsc.es for its consideration.
