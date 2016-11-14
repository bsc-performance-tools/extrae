.. _cha:configuration:

Configuration, build and installation
=====================================


.. _sec:configuration:

Configuration of the instrumentation package
--------------------------------------------

There are many options to be applied at configuration time for the
instrumentation package. We point out here some of the relevant options, sorted
alphabetically. To get the whole list run ``configure --help``. Options can be
enabled or disabled. To enable them use ``--enable-X`` or ``--with-X=``
(depending on which option is available), to disable them use ``--disable-X`` or
``--without-X``.

* .. option:: --enable-instrument-dynamic-memory

  Allows instrumentation of dynamic memory related calls (such as malloc, free,
  realloc).

* .. option:: --enable-merge-in-trace

  Embed the merging process in the tracing library so the final tracefile can be
  generated automatically from the application run.

* .. option:: --enable-parallel-merge

  Build the parallel mergers (``mpimpi2prv``/``mpimpi2dim``) based on MPI.

* .. option:: --enable-posix-clock

  Use POSIX clock (``clock_gettime`` call) instead of low level timing routines.
  Use this option if the system where you install the instrumentation package
  modifies the frequency of its processors at runtime.

* .. option:: --enable-single-mpi-lib

  Produces a single instrumentation library for MPI that contains both Fortran
  and C wrappers. Applications that call the MPI library from both C and Fortran
  languages need this flag to be enabled.

* .. option:: --enable-sampling
  
  Enable PAPI sampling support.

* .. option:: --enable-pmapi

  Enable PMAPI library to gather CPU performance counters. PMAPI is a base
  package installed in AIX systems since version 5.2.

* .. option:: --enable-openmp

  Enable support for tracing OpenMP on IBM, GNU and Intel runtimes. The IBM
  runtime instrumentation is only available for Linux/PowerPC systems.

* .. option:: --enable-openmp-gnu

  Enable support for tracing OpenMP on GNU runtime.

* .. option:: --enable-openmp-intel

  Enable support for tracing OpenMP on Intel runtime.

* .. option:: --enable-openmp-ibm

  Enable support for tracing OpenMP IBM runtime. The IBM runtime instrumentation
  is only available for Linux/PowerPC systems.

* .. option:: --enable-openmp-ompt

  Enables support for tracing OpenMP runtimes through the OMPT specification.

  .. note::
    Enabling this option disables the regular instrumentation system
    available through :option:`--enable-openmp-gnu`,
    :option:`--enable-openmp-intel` and :option:`--enable-openmp-gnu`.

* .. option:: --enable-smpss

  Enable support for tracing SMP-superscalar.

* .. option:: --enable-nanos

  Enable support for tracing Nanos run-time.

* .. option:: --enable-online

  Enables the on-line analysis module.

* .. option:: --enable-pthread

  Enable support for tracing pthread library calls.

* .. option:: --enable-xml

  Enable support for XML configuration (not available on BG/L, BG/P and BG/Q
  systems).

* .. option:: --enable-xmltest

  Do not try to compile and run a test LIBXML program.

* .. option:: --enable-doc

  Generates this documentation.

* .. option:: --prefix=<PATH>

  Location where the installation will be placed. After issuing ``make install``
  you will find under DIR the entries ``lib/``, ``include/``, ``share`` and
  ``bin`` containing everything needed to run the instrumentation package.

* .. option:: --with-bfd=<PATH>

  Specify where to find the Binary File Descriptor package. In conjunction with
  libiberty, it is used to translate addresses into source code locations.

* .. option:: --with-binary-type=<32|64|default>

  Specifies the type of memory address model when compiling (32bit or 64bit).

* .. option:: --with-boost=<PATH>

  Specify the location of the BOOST package. This package is required when using
  the DynInst instrumentation with versions newer than 7.0.1.

* .. option:: --with-binutils=<PATH>

  Specify the location for the binutils package. The binutils package is
  necessary to translate addresses into source code references.

* .. option:: --with-clustering=<PATH>

  If the on-line analysis module is enabled (see :option:`--enable-online`),
  specify where to find ClusteringSuite libraries and includes. This package
  enables support for on-line clustering analysis.

* .. option:: --with-cuda=<PATH>

  Enable support for tracing CUDA calls on nVidia hardware and needs to point to
  the CUDA SDK installation path. This instrumentation is only valid in binaries
  that use the shared version of the CUDA library. Interposition has to be done
  through the ``LD_PRELOAD`` mechanism. It is superseded by
  :option:`--with-cupti` which also supports instrmentation for static binaries.

* .. option:: --with-cupti=<PATH>

  Specify the location of the CUPTI libraries. CUPTI is used to instrument CUDA
  calls, and supersedes the :option:`--with-cuda`, although it still requires
  it.

* .. option:: --with-dyninst=<PATH>

  Specify the installation location for the DynInst package. |TRACE| also
  requires the DWARF package :option:`--with-dwarf` when using DynInst. Also,
  newer versions of DynInst (versions after 7.0.1) require the BOOST package
  :option:`--with-boost`. This flag is mandatory.  Requires a working
  installation of a C++ compiler.

* .. option:: --with-fft=<PATH>

  If the spectral analysis module is enabled (see :option:`--with-spectral`),
  specify where to find FFT libraries and includes. This library is a dependency
  of the Spectral libraries.

* .. option:: --with-java-jdk=<PATH>

  Specify the location of JAVA development kit (JDK). This is necessary to
  create the connectors between |TRACE| and Java applications.

* .. option:: --with-java-aspectj=<PATH>

  Specify the location of the AspectJ infrastructure. AspectJ is used to give
  support to dynamically instrumented Java applications.

* .. option:: --with-java-aspectj-weaver=<path/to/aspectjweaver.jar>

  AspectJ includes the ``aspectweaver.jar`` file that is responsible for the
  execution of dynamically instrumented Java applications. If
  :option:`--with-java-aspectj` cannot locate this file, use this option to
  tell |TRACE| where to find it.

* .. option:: --with-liberty=<PATH>

  Specify where to find the libiberty package. In conjunction with Binary File
  Descriptor, it is used to translate addresses into source code locations.

* .. option:: --with-mpi=<PATH>

  Specify the location of an MPI installation to be used for the instrumentation
  package. This flag is mandatory.

* .. option:: --with-mpi-name-mangling=<0u|1u|2u|upcase|auto>

  Choose the Fortran name decoration (0, 1 or 2 underscores) for MPI symbols, or
  auto to automatically detect the name mangling.

* .. option:: --with-synapse=<PATH>

  If the on-line analysis module is enabled (see :option:`--enable-online`),
  specify where to find Synapse libraries and includes. This library is a
  front-end of the MRNet library.

* .. option:: --with-opencl=<PATH>

  Specify the location for the OpenCL package, including library and include
  directories.

* .. option:: --with-openshmem=<PATH>

  Specify the location of the OpenSHMEM installation to be used for the
  instrumentation package.

* .. option:: --with-papi=<PATH>

  Specify where to find PAPI libraries and includes. PAPI is used to gather
  performance counters. This flag is mandatory.

* .. option:: --with-spectral=<PATH>

  If the on-line analysis module is enabled (see :option:`--enable-online`),
  specify where to find Spectral libraries and includes. This package enables
  support for on-line spectral analysis.

* .. option:: --with-unwind=<PATH>

  Specify where to find Unwind libraries and includes. This library is used to
  get callstack information on several architectures (including IA64 and Intel
  x86-64). This flag is mandatory.


.. sec:build:

Build
-----

To build the instrumentation package, just issue ``make`` after the
configuration.


.. sec:check:

Check
-----

The |TRACE| package contains some consistency checks. The aim of such checks is
to determine whether a functionality is operative in the target environment
and/or check whether the development of |TRACE| has introduced any misbehavior.
To run the checks, just issue ``make check`` after the compilation. Please,
notice that checks are meant to be run in the machine that the configure script
was run, thus the results of the checks on machines with back-end nodes
different to front-end nodes (like BG/* systems) are not representative at all.


.. sec:installation:

Installation
------------

To install the instrumentation package in the directory chosen at configure step
(through :option:`--prefix` option), issue ``make install``.


.. sec:exampleconfs:

Examples of configuration on different machines
-----------------------------------------------

All commands given here are given as an example to configure and install the
package, you may need to tune them properly (i.e., choose the appropriate
directories for packages and so). These examples assume that you are using a
sh/bash shell, you must adequate them if you use other shells (like csh/tcsh).


.. subsec:crayxc40:

Cray XC 40 - |TRACE| |release|
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before issuing the configure command, the following modules were loaded:

* PrgEnv-gnu/5.2.40
* cray-mpich/7.2.2
* cudatoolkit6.5/6.5.14-1.0502.9613.6.1
* libunwind/1.1-CrayGNU-5.2.4

Configuration command:

.. parsed-literal::

  ./configure --prefix=${PREFIX}/|tracepath|
              --with-papi=/opt/cray/papi/5.4.1.1
              --with-mpi=/opt/cray/mpt/7.2.2/gni/mpich2-gnu/48
              --with-unwind=/apps/daint/5.2.UP02/easybuild/software/libunwind/1.1-CrayGNU-5.2.40
              --with-cuda=/opt/nvidia/cudatoolkit6.5/6.5.14-1.0502.9613.6.1
              --enable-sampling
              --without-dyninst
              --with-binary-type=64
              CC=gcc CXX=g++ MPICC=cc

Build and installation commands:

.. parsed-literal::

  ``make``
  ``make install``


.. subsec:bglp

Bluegene (L and P variants)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configuration command:

.. parsed-literal::

  ./configure --prefix=/homec/jzam11/jzam1128/aplic/|tracepath|
              --with-papi=/homec/jzam11/jzam1128/aplic/papi/4.1.2.1
              --with-bfd=/bgsys/local/gcc/gnu-linux\_4.3.2/powerpc-linux-gnu/powerpc-bgp-linux
              --with-liberty=/bgsys/local/gcc/gnu-linux\_4.3.2/powerpc-bgp-linux
              --with-mpi=/bgsys/drivers/ppcfloor/comm
              --without-unwind
              --without-dyninst

Build and installation commands:

.. parsed-literal::

  ``make``
  ``make install``


.. subsec:bgq

BlueGene/Q
^^^^^^^^^^

To enable parsing the XML configuration file, the libxml2 must be installed. As
of the time of writing this user guide, we have been only able to install the
static version of the library in a BG/Q machine, so take this into consideration
if you install the libxml2 in the system. Similarly, the binutils package
(responsible for translating application addresses into source code locations)
that is available in the system may not be properly installed and we suggest
installing the binutils from the source code using the BG/Q cross-compiler.
Regarding the cross-compilers, we have found that using the IBM XL compilers may
require using the XL libraries when generating the final application binary with
|TRACE|, so we would suggest using the GNU cross-compilers
(``/bgsys/drivers/ppcfloor/gnu-linux/bin/powerpc64-bgq-linux-*``).

If you want to add libxml2 and binutils support into |TRACE|, your configuration
command may resemble to:

.. parsed-literal::

  ./configure --prefix=/homec/jzam11/jzam1128/aplic/juqueen/|tracepath|
              --with-mpi=/bgsys/drivers/ppcfloor/comm/gcc
              --without-unwind
              --without-dyninst
              --disable-openmp
              --disable-pthread
              --with-libz=/bgsys/local/zlib/v1.2.5
              --with-papi=/usr/local/UNITE/packages/papi/5.0.1
              --with-xml-prefix=/homec/jzam11/jzam1128/aplic/juqueen/libxml2-gcc
              --with-binutils=/homec/jzam11/jzam1128/aplic/juqueen/binutils-gcc
              --enable-merge-in-trace

Otherwise, if you do not want to add support for the libxml2 library, your
configuration may look like this:

.. parsed-literal::

  ./configure --prefix=/homec/jzam11/jzam1128/aplic/juqueen/|tracepath|
              --with-mpi=/bgsys/drivers/ppcfloor/comm/gcc
              --without-unwind
              --without-dyninst
              --disable-openmp
              --disable-pthread
              --with-libz=/bgsys/local/zlib/v1.2.5
              --with-papi=/usr/local/UNITE/packages/papi/5.0.1
              --disable-xml

In any situation, the build and installation commands are:

.. parsed-literal::

  ``make``
  ``make install``


.. subsec:aix

AIX
^^^

Some extensions of |TRACE| do not work properly (nanos, SMPss and OpenMP) on
AIX.  In addition, if using IBM MPI (aka POE) the make will complain when
generating the parallel merge if the main compiler is not xlc/xlC. So, you can
either change the compiler or disable the parallel merge at compile step. Also,
command ``ar`` can complain if 64bit binaries are generated. It's a good idea to
run make with ``OBJECT_MODE=64`` set to avoid this.


.. subsubsec:aix32ibm:

Compiling the 32bit package using the IBM compilers
"""""""""""""""""""""""""""""""""""""""""""""""""""

Configuration command:

.. parsed-literal::

  CC=xlc
  CXX=xlC
  ./configure --prefix=${PREFIX}/|tracepath|
              --disable-nanos               
              --disable-smpss
              --disable-openmp
              --with-binary-type=32
              --without-unwind
              --enable-pmapi
              --without-dyninst
              --with-mpi=/usr/lpp/ppe.poe

Build and installation commands:

.. parsed-literal::

  ``make``
  ``make install``


.. subsubsec:aix64ibmnomerge:

Compiling the 64bit package without the parallel merge
""""""""""""""""""""""""""""""""""""""""""""""""""""""

Configuration command:

.. parsed-literal::

  ./configure --prefix=${PREFIX}/|tracepath|
              --disable-nanos
              --disable-smpss
              --disable-openmp
              --disable-parallel-merge
              --with-binary-type=64
              --without-unwind
              --enable-pmapi
              --without-dyninst
              --with-mpi=/usr/lpp/ppe.poe

Build and installation commands:

.. parsed-literal::

  ``OBJECT_MODE=64 make``
  ``make install``


.. subsec:linux:

Linux
^^^^^

.. subsubsec:linuxmpichomppapi:

Compiling using default binary type using MPICH, OpenMP and PAPI
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Configuration command:

.. parsed-literal::

  ./configure --prefix=${PREFIX}/|tracepath|
              --with-mpi=/home/harald/aplic/mpich/1.2.7
              --with-papi=/usr/local/papi
              --enable-openmp
              --without-dyninst
              --without-unwind

Build and installation commands:

.. parsed-literal::

  ``make``
  ``make install``


.. subsubsec:linux32mixed:

Compiling 32bit package in a 32/64bit mixed environment
"""""""""""""""""""""""""""""""""""""""""""""""""""""""

Configuration command:

.. parsed-literal::

  ./configure --prefix=${PREFIX}/|tracepath|
              --with-mpi=/opt/osshpc/mpich-mx
              --with-papi=/gpfs/apps/PAPI/3.6.2-970mp
              --with-binary-type=32
              --with-unwind=${HOME}/aplic/unwind/1.0.1/32
              --with-elf=/usr
              --with-dwarf=/usr
              --with-dyninst=${HOME}/aplic/dyninst/7.0.1/32

Build and installation commands:

.. parsed-literal::

  ``make``
  ``make install``


.. subsubsec:linux64mixed:

Compiling 64bit package in a 32/64bit mixed environment
"""""""""""""""""""""""""""""""""""""""""""""""""""""""

Configuration command:

.. parsed-literal::

  ./configure --prefix=${PREFIX}/|tracepath|
              --with-mpi=/opt/osshpc/mpich-mx
              --with-papi=/gpfs/apps/PAPI/3.6.2-970mp
              --with-binary-type=64
              --with-unwind=${HOME}/aplic/unwind/1.0.1/64
              --with-elf=/usr
              --with-dwarf=/usr
              --with-dyninst=${HOME}/aplic/dyninst/7.0.1/64

Build and installation commands:

.. parsed-literal::

  ``make``
  ``make install``


.. subsubsec:linuxopenmpidyninstunwind:

Compiling using default binary type, using OpenMPI, DynInst and libunwind
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Configuration command:

.. parsed-literal::

  ./configure --prefix=${PREFIX}/|tracepath|
              --with-mpi=/home/harald/aplic/openmpi/1.3.1
              --with-dyninst=/home/harald/dyninst/7.0.1
              --with-dwarf=/usr
              --with-elf=/usr
              --with-unwind=/home/harald/aplic/unwind/1.0.1
              --without-papi

Build and installation commands:

.. parsed-literal::

  ``make``
  ``make install``


.. subsubsec:linuxcrayxt5_64sampling:

Compiling on CRAY XT5 for 64bit package and adding sampling
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Notice the :option:`--disable-xmltest`. As backends programs cannot be run in
the frontend, we skip running the XML test. Also using a local installation of
libunwind.

Configuration command:

.. parsed-literal::

  CC=cc
  CFLAGS='-O3 -g'
  LDFLAGS='-O3 -g'
  CXX=CC CXXFLAGS='-O3 -g'
  ./configure --prefix=${PREFIX}/|tracepath|
              --with-mpi=/opt/cray/mpt/4.0.0/xt/seastar/mpich2-gnu
              --with-binary-type=64
              --with-xml-prefix=/sw/xt5/libxml2/2.7.6/sles10.1\_gnu4.1.2
              --disable-xmltest
              --with-bfd=/opt/cray/cce/7.1.5/cray-binutils
              --with-liberty=/opt/cray/cce/7.1.5/cray-binutils
              --enable-sampling
              --enable-shared=no
              --with-papi=/opt/xt-tools/papi/3.7.2/v23
              --with-unwind=/ccs/home/user/lib
              --without-dyninst

Build and installation commands:

.. parsed-literal::

  ``make``
  ``make install``


.. subsubsec:linuxintelmic:

Compiling for the Intel MIC accelerator / Xeon Phi
""""""""""""""""""""""""""""""""""""""""""""""""""

The Intel MIC accelerators (also codenamed KnightsFerry - KNF and KnightsCorner
- KNC) or Xeon Phi processors are not binary compatible with the host (even if
it is an Intel x86 or x86/64 chip), thus the |TRACE| package must be compiled
specially for the accelerator (twice if you want |TRACE| for the host). While
the host configuration and installation has been shown before, in order to
compile |TRACE| for the accelerator you must configure |TRACE| like:

Configuration command:

.. parsed-literal::

  ./configure --prefix=/home/Computational/harald/extrae-mic
              --with-mpi=/opt/intel/impi/4.1.0.024/mic
              --without-dyninst
              --without-papi
              --without-unwind
              --disable-xml
              --disable-posix-clock
              --with-libz=/opt/extrae/zlib-mic
              --host=x86\_64-suse-linux-gnu
              --enable-mic
              CFLAGS="-O -mmic -I/usr/include"
              CC=icc
              CXX=icpc
              MPICC=/opt/intel/impi/4.1.0.024/mic/bin/mpiicc

To compile it, just issue:

.. parsed-literal::

  ``make``
  ``make install``


.. subsubsec:linuxarm:

Compiling on a ARM based processor machine using Linux
""""""""""""""""""""""""""""""""""""""""""""""""""""""

If using the GNU toolchain to compile the library, we suggest at least using
version 4.6.2 because of its enhaced in this architecture.

Configuration command:

.. parsed-literal::

  CC=/gpfs/APPS/BIN/GCC-4.6.2/bin/gcc-4.6.2
  ./configure --prefix=/gpfs/BSCTOOLS/|tracepath|
              --with-unwind=/gpfs/BSCTOOLS/libunwind/1.0.1-git
              --with-papi=/gpfs/BSCTOOLS/papi/4.2.0
              --with-mpi=/usr
              --enable-posix-clock
              --without-dyninst

Build and installation commands:

.. parsed-literal::

  ``make``
  ``make install``


.. subsubsec:linuxslurmmpich2:

Compiling in a Slurm/MOAB environment with support for MPICH2
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Configuration command:

.. parsed-literal::

  export MP_IMPL=anl2
  ./configure --prefix=${PREFIX}/|tracepath|
              --with-mpi=/gpfs/apps/MPICH2/mx/1.0.8p1..3/32
              --with-papi=/gpfs/apps/PAPI/3.6.2-970mp
              --with-binary-type=64
              --without-dyninst
              --without-unwind

Build and installation commands:

.. parsed-literal::

  ``make``
  ``make install``


.. subsubsec:linuxibmpoe:

Compiling in a environment with IBM compilers and POE
"""""""""""""""""""""""""""""""""""""""""""""""""""""

Configuration command:

.. parsed-literal::

  CC=xlc
  CXX=xlC
  ./configure --prefix=${PREFIX}/|tracepath|
              --with-mpi=/opt/ibmhpc/ppe.poe
              --without-dyninst
              --without-unwind
              --without-papi

Build and installation commands:

.. parsed-literal::

  ``make``
  ``make install``


.. subsubsec:linuxgnupoe:

Compiling in a environment with GNU compilers and POE
"""""""""""""""""""""""""""""""""""""""""""""""""""""

Configuration command:

.. parsed-literal::

  ./configure --prefix=${PREFIX}/|tracepath|
              --with-mpi=/opt/ibmhpc/ppe.poe
              --without-dyninst
              --without--unwind
              --without-papi

Build and installation commands:

.. parsed-literal::

  ``MP_COMPILER=gcc make``
  ``make install``


.. subsubsec:linuxhornetcrayxc40:

Compiling |TRACE| |release| in Hornet / Cray XC40 system
""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Configuration command, enabling MPI, PAPI and online analysis over MRNet.

.. parsed-literal::

  ./configure --prefix=/zhome/academic/HLRS/xhp/xhpgl/tools/|tracepath|/intel
              --with-mpi=/opt/cray/mpt/7.1.2/gni/mpich2-intel/140
              --with-unwind=/zhome/academic/HLRS/xhp/xhpgl/tools/libunwind
              --without-dyninst
              --with-papi=/opt/cray/papi/5.3.2.1
              --enable-online
              --with-mrnet=/zhome/academic/HLRS/xhp/xhpgl/tools/mrnet/4.1.0
              --with-spectral=/zhome/academic/HLRS/xhp/xhpgl/tools/spectral/3.1
              --with-synapse=/zhome/academic/HLRS/xhp/xhpgl/tools/synapse/2.0

Build and installation commands:

.. parsed-literal::

  ``make``
  ``make install``


.. subsubsec:linuxshaheeniicrayxc40:

Compiling |TRACE| |release| in Shaheen II / Cray XC40 system
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

With the following modules loaded:

.. parsed-literal::

  ``module swap PrgEnv-XXX/YYY PrgEnv-cray/5.2.40``
  ``module load cray-mpich``

Configuration command, enabling MPI, PAPI:

.. parsed-literal::

  ./configure --prefix=${PREFIX}/|tracepath|
              --with-mpi=/opt/cray/mpt/7.1.1/gni/mpich2-cray/83
              --with-binary-type=64
              --with-unwind=/home/markomg/lib
              --without-dyninst
              --disable-xmltest
              --with-bfd=/opt/cray/cce/default/cray-binutils
              --with-liberty=/opt/cray/cce/default/cray-binutils
              --enable-sampling
              --enable-shared=no
              --with-papi=/opt/cray/papi/5.3.2.1

Build and installation commands:

.. parsed-literal::

  ``make``
  ``make install``


.. subsec:howconfigured:

Knowing how a package was configured
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are interested on knowing how an |TRACE| package was configured execute
the following command after setting ``EXTRAE_HOME`` to the base location of an
installation.

.. parsed-literal::

  ``${EXTRAE_HOME}/etc/configured.sh``

this command will show the configure command itself and the location of some
dependencies of the instrumentation package.
