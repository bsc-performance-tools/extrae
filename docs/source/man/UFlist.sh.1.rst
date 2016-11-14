:orphan:

.. _UFlist.sh(1):


UFlist.sh
=========

SYNOPSIS
--------

**UFlist.sh** *binary*


DESCRIPTION
-----------

Echo the available routines in the text section of *binary* with their
corresponding address. The output can be fed directly to |TRACE|
(EXTRAE_FUNCTIONS environment variable) to trace a selected set of user
functions. Instrumenting routins requires the source files to be compiled with
GCC and with **-finstrument-functions**.


ENVIRONMENT
-----------

UFlist.sh uses the following environment variables:

**TMPDIR**
  Location where all temporal files will be stored. Defaults to $PWD.


REPORTING_BUGS
--------------

If you find any bug in the documentation or in the software, pelase send a
descriptive mail to: **tools@bsc.es**

SEE ALSO
--------

:manpage:`mpi2prv(1)`

:manpage:`extrae_event(3)`, :manpage:`extrae_counters(3)`,
:manpage:`extrae_eventandcounters(3)`, :manpage:`extrae_shutdown(3)`,
:manpage:`extrae_restart(3)`, :manpage:`extrae_set_tracing_tasks(3)`,
:manpage:`extrae_set_options(3)`,
