.. _cha:introduction:

Introduction
============

|TRACE| is a dynamic instrumentation package to trace programs compiled and
run with the shared memory model (like OpenMP and pthreads), the message passing
(MPI) programming model or both programming models (different MPI processes
using OpenMP or pthreads within each MPI process). |TRACE| generates trace files
that can be visualized with |PARAVER|.

|TRACE| is currently available on different platforms and operating systems:
IBM PowerPC running Linux or AIX, and x86 and x86-64 running Linux. It also has
been ported to OpenSolaris and FreeBSD.

The combined use of |TRACE| and |PARAVER| offers an enormous analysis
potential, both qualitative and quantitative. With these tools the actual
performance bottlenecks of parallel applications can be identified. The
microscopic view of the program behavior that the tools provide is very useful
to optimize the parallel program performance.

This document tries to give the basic knowledge to use the |TRACE| tool. Chapter
:ref:`cha:configuration` explains how the package can be configured and
installed. Chapter :ref:`cha:examples` explains how to monitor an application to
obtain its trace file. At the end of this document there are appendices that
include: :ref:`cha:faq` and a list of :ref:`cha:InstrumentedRoutines` in the
package.


.. _sec:paraver:

What is the |PARAVER| tool?
---------------------------

|PARAVER| is a flexible parallel program visualization and analysis tool based
on an easy-to-use wxWidgets GUI. |PARAVER| was developed responding to the need
of having a qualitative global perception of the application behavior by visual
inspection and then to be able to focus on the detailed quantitative analysis of
the problems. |PARAVER| provides a large amount of information useful to decide
the points on which to invest the programming effort to optimize an application.

Expressive power, flexibility and the capability of efficiently handling large
traces are key features addressed in the design of |PARAVER|. The clear and
modular structure of |PARAVER| plays a significant role towers achieving these
targets.

Some |PARAVER| features are the support for:

* Detailed quantitative analysis of program performance,
* concurrent comparative analysis of several traces,
* fast analysis of very large traces,
* support for mixed message passing and shared memory (network of SMPs), and,
* customizable semantics of the visualized information.

One of the main features of |PARAVER| is the flexibility to represent traces
coming from different environments. Traces are composed of state records, events
and communications with associated timestamps. These three elements can be used
to build traces that capture the behavior along time of very different kind of
systems. The |PARAVER| distribution includes, either in its own distribution or
as additional packages, the following instrumentation modules:

#. Sequential application tracing: it is included in the |PARAVER| distribution.
   It can be used to trace the value of certain variables, procedure
   invocations, ... in a sequential program.
#. Parallel application tracing: a set of modules are optionally available to
   capture the activity of parallel applications using shared-memory,
   message-passing paradigms, or a combination of them.
#. System activity tracing in a multiprogrammed environment: an application to
   trace processor allocations and process migrations is optionally available in
   the |PARAVER| distribution.
#. Hardware counters tracing: an application to trace the hardware counter 
   values is optionally available in the |PARAVER| distribution.


.. _sec:whereparaver:

Where can the |PARAVER| tool be found?
--------------------------------------

The |PARAVER| distribution can be found at https://tools.bsc.es/downloads

|PARAVER| binaries are available for Linux/x86, Linux/x86-64 and Linux/ia64,
Windows.

In the Documentation section of the aforementioned URL you can find the
`|PARAVER| *Reference Manual*
<https://tools.bsc.es/sites/default/files/documentation/1364.pdf>`_ and
`|PARAVER| *Tutorial*
<https://tools.bsc.es/sites/default/files/documentation/1367.pdf>`_ in addition
to the documentation for other instrumentation packages.

|TRACE| and |PARAVER| tools e-mail support is tools@bsc.es.
