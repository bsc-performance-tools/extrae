.. _cha:online:

Extrae On-line User Guide
=========================

.. _sec:OnlineIntro:

Introduction
------------

|TRACE| On-line is a new module developed for the |TRACE| tracing toolkit,
available from version 3.0, that incorporates intelligent monitoring, analysis
and selection of the traced data. This tracing setup is tailored towards long
executions that are producing large traces. Applying automatic analysis
techniques based on clustering, signal processing and active monitoring, |TRACE|
gains the ability to inspect and filter the data while it is being collected to
minimize the amount of data emitted into the trace, while maximizing the amount
of relevant information presented.

|TRACE| On-line has been developed on top of Synapse [#SYNAPSE]_, a framework
that facilitates the deployment of applications that follow the master/slave
architecture based on the MRNet [#MRNET]_ software overlay network. Thanks to
its modular design, new types of automatic analysis can be added very easily as
new plug-ins into the on-line tracing system, just by defining new Synapse
protocols.

This document briefly describes the main features of the |TRACE| On-line module,
and shows how it has to be configured and the different options available.

.. _sec:OnlineAutomaticAnalysis:

Automatic analysis
------------------

|TRACE| On-line currently supports three types of automatic analysis: 

#. fine-grain structure detection based on clustering techniques,
#. periodicity detection based on signal processing techniques,
#. and multi-experiment analysis based on active monitoring techniques.
   
|TRACE| On-line has to be configured to apply one of these types of analysis,
and then the analysis will be performed periodically as new data is being
traced. 


.. _subsec:OnlineStructureDetection:

Structure detection
-------------------

This mechanism aims at identifying the fine-grain structure of the computing
regions of the program. Applying density-based clustering, this method is able
to expose the main performance trends in the computations, and this information
is useful to focus the analysis on the zones of real interest.  To perform the
cluster analysis, |TRACE| On-line relies on the ClusteringSuite tool
[#CLUSTERINGSUITE]_.

At each phase of analysis, several outputs are produced:

* A scatter-plot representation that illustrates the behavior of the main
  computing regions of the program, that enables a quick evaluation of potential
  imbalances.
* A summary of several performance metrics per cluster.
* On supported machines, a CPI stack model that attributes stall cycles to
  specific hardware components.
* And a trace that is augmented with the clusters information, that allows to
  identify patterns of performance and variabilites.

Subsequent clustering results can be used to study the evolution of the
application over time. In order to study how the clusters are evolving, the
xtrack tool [#XTRACK]_ can be used.


.. _subsec:OnlinePeriodicityDetection:

Periodicity detection
---------------------

This mechanism allows to detect iterative patterns over a wide region of time,
and precisely delimit where the iterations start. Once a period has been found,
those iterations presenting less perturbations are selected to produce a
representative trace, and the rest of the data is basically discarded.  The
result of applying this mechanism is a compact trace where only the
representative iterations are traced in full detail, and for the rest of the
execution we can optionally keep summarized information in the form of phase
profiles or a "low resolution" trace.

Please note that applying this technique to a very short execution, or if no
periodicity can be detected in the application, may result in an empty trace
depending on the configuration options selected (see section
:ref:`sec:onlineconfiguration`).


.. _subsec:OnlineMultiExperimentAnalysis:

Multi-experiment analysis
-------------------------

This mechanism employs active measurement techniques in order to simulate
different execution scenarios under the same execution. |TRACE| On-line is able
to add controlled interference into the program to simulate different
computation loads, network bandwidth, memory congestion and even tuning some
configurations of the parallel runtime (currently supports MPI Dynamic Load
Balance (DLB) runtime). Then, the application behavior can be studied under
different circumstances, and tracking can be used to analyze the impact of these
configurations on the program performance.  This technique aims at reducing the
number of executions necessary to evaluate different parameters and
characteristics of your program.


.. _sec:OnlineConfiguration:

Configuration
-------------

In order to activate the On-line tracing mode, the user has to enable the
corresponding configuration section in the |TRACE| XML configuration file. This
section is found under *trace-control > remote-control > online*. The default
configuration is already ready to use:

.. code-block:: xml

  <online enabled="yes"
          analysis="clustering"
          frequency="auto"
          topology="auto">

The available options for the ``<online>`` section are the following:

* :option:`enabled`
  Set to ``yes`` to activate the On-line tracing mode.
* :option:`analysis`
  Choose from ``clustering``, ``spectral`` and ``gremlins``.
* :option:`frequency`
  Set the time in seconds after which a new phase of analysis will be triggered,
  or ``auto`` to let |TRACE| decide this automatically.
* :option:`topology`
  Set the desired tree process tree topology, or ``auto`` to let |TRACE| decide
  this automatically.

Depending on the analysis selected, the following specific options become
available.


.. _subsec:ClusteringAnalysisOptions:

Clustering analysis options
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: xml

  <clustering config="cl.I.IPC.xml"/>

* :option:`config`
  Specify the path to the ClusteringSuite XML configuration file.


.. _subsec:SpectralAnalysisOptions:

Spectral analysis options
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: xml

  <spectral max_periods="0"
            num_iters="3"
            min_seen="0"
            min_likeness="85">
    <spectral_advanced enabled="no"
                       burst_threshold="80">
      <periodic_zone detail_level="profile" />
      <non_periodic_zone detail_level="bursts" min_duration="3s" />
    </spectral_advanced>
  </spectral>

The basic configuration options for the spectral analysis are the following:

* :option:`max_periods`
  Set to the maximum number of periods to trace, or ``all`` to explore the whole
  run.
* :option:`num_iters`
  Set to the number of iterations to trace per period.
* :option:`min_seen`
  Minimum repetitions of a period before tracing it (0 to trace the first time
  that you encounter it).
* :option:`min_likeness`
  Minimum percentage of similarity to compare two equivalent periods.

Also, some advanced settings are tunable in the ``<spectral_advanced>`` section:

* :option:`enabled`
  Set to ``yes`` to activate the spectral analysis advanced options.
* :option:`burst_threshold`
  Filter threshold to keep all CPU bursts that add up to the given total time
  percentage.
* :option:`detail_level`
  Specify the granularity of the data stored for the non-representative
  iterations of the periodic region, and in the non-periodic regions. Choose
  from none (everything is discarded), profile (phase profile at the start of
  each iteration/region) or bursts (trace in bursts mode).
* :option:`min_duration`
  Minimum duration in seconds of the non-periodic regions for emitting any
  information regarding that region into the trace.


.. _subsec:GremlinsAnalysisOptions:

Gremlins analysis options
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: xml

  <gremlins start="0"
            increment="2"
            roundtrip="no"
            loop="no" />

* :option:`start`
  Number of gremlins at the beginning of the execution.
* :option:`increment`
  Number of extra gremlins at each analysis phase. Can also be a negative value
  to indicate that you want to remove gremlins.
* :option:`roundtrip`
  Set to ``yes`` if you want to start adding gremlins after you decrease to 0,
  or vice-versa, start removing gremlins after you reach the maximum.
* :option:`loop`
  Set to ``yes`` if you want to go back to the initial number of gremlins and
  repeat the sequence of adding/removing gremlins after you have finished a
  complete sequence.



.. rubric:: Footnotes

.. [#SYNAPSE] https://github.com/gllort/synapse

.. [#MRNET] http://www.paradyn.org/mrnet/

.. [#CLUSTERINGSUITE] You can download it from https://tools.bsc.es/downloads.

.. [#XTRACK] You can download it from https://tools.bsc.es/downloads.
