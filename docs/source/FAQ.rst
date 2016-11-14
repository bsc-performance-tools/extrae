.. _cha:FAQ:

Frequently Asked Questions
==========================


.. _sec:FAQconfigure:

Configure, compile and link FAQ
-------------------------------

.. admonition:: Question:

    The :command:`bootstrap` script claims :command:`libtool` errors like:

    * ``src/common/Makefile.am:9: Libtool library used but 'LIBTOOL' is undefined``
    * ``src/common/Makefile.am:9: The usual way to define 'LIBTOOL' is to add 'AC_PROG_LIBTOOL'``
    * ``src/common/Makefile.am:9: to 'configure.ac' and run 'aclocal' and 'autoconf' again.``
    * ``src/common/Makefile.am:9: If 'AC_PROG_LIBTOOL' is in 'configure.ac', make sure``
    * ``src/common/Makefile.am:9: its definition is in aclocal's search path.``

  Answer:
    Add to the ``aclocal`` (which is called in ``bootstrap``) the directory where
    it can find the M4-macro files from ``libtool``. Use the :option:`-I` option
    to add it.

.. admonition:: Question:

    The :command:`bootstrap` script claims that some macros are not found in the
    library, like:
  
    * ``aclocal:configure.ac:338: warning: macro 'AM_PATH_XML2' not found in library``

  Answer:
    Some M4 macros are not found. In this specific example, the libxml2 is not
    installed or cannot be found in the typical installation directory. To solve
    this issue, check whether the libxml2 is installed and modify the line in
    the :file:`bootstrap` script that reads ``aclocal -I config`` to ``aclocal
    -I config -I/path/to/xml/m4/macros`` where :file:`/path/to/xml/m4/macros` is
    the directory where the libxml2 M4 got installed (for example
    :file:`/usr/local/share/aclocal`).

.. admonition:: Question:

    The application cannot be linked succesfully. The link stage complains about,
    or something similar, to:
  
    * ``ld: 0711-317 ERROR: Undefined symbol: .__udivdi3``.
    * ``ld: 0711-317 ERROR: Undefined symbol: .__mulvsi3``.

  Answer:
    The instrumentation libraries have been compiled with GNU compilers whereas
    the application is compiled using IBM XL compilers. Add the libgcc's library
    to the link stage of the application. This library can be found under the
    installation directory of the GNU compiler.

.. admonition:: Question:

    The application cannot be linked. The linker misses some routines like:
  
    * ``src/common/utils.c:122: undefined reference to '__intel_sse2_strlen'``.
    * ``src/common/utils.c:125: undefined reference to '__intel_sse2_strdup'``.
    * ``src/common/utils.c:132: undefined reference to '__intel_sse2_strtok'``.
    * ``src/common/utils.c:100: undefined reference to '__intel_sse2_strncpy'``.
    * ``src/common/timesync.c:211: undefined reference to '__intel_fast_memset'``.

  Answer:
    The instrumentation libraries have been compiled using Intel compilers
    (*i.e.*, ``icc``, ``icpc``) whereas the application is being linked through
    non-Intel compilers or ``ld`` directly. You can proceed in three directions,
    you can either compile your application using the Intel compilers, or add an
    Intel library that provides these routines (:file:`libintlc.so` and
    :file:`libirc.so`, for instance), or even recompile |TRACE| using the GNU
    compilers. Note, nonetheless, that using Intel MPI compiler does not
    guarantee using the Intel compiler backends, just run the MPI compiler
    (:command:`mpicc`, :command:`mpiCC`, :command:`mpif77`, :command:`mpif90`,
    ...) with the :option:`-v` flag to get information on what compiler backend
    relies.

.. admonition:: Question:

    The make command dies when building libraries belonging |TRACE| in an AIX
    machine with messages like:
  
    * ``libtool: link: ar cru libcommon.a libcommon_la-utils.o libcommon_la-events.o``
    * ``ar: 0707-126 libcommon\_la-utils.o is not valid with the current object file mode.``
  
      ``Use the -X option to specify the desired object mode.``
    * ``ar: 0707-126 libcommon\_la-events.o is not valid with the current object file mode.``
  
      ``Use the -X option to specify the desired object mode.``

  Answer:
    ``libtool`` uses the :command:`ar` command to build static libraries.
    However, :command:`ar` does need special flags (:option:`-X64`) to deal with
    64 bit objects. To workaround this problem, just set the environment
    variable :envvar:`OBJECT_MODE` to 64 before executing :command:`gmake`. The
    :command:`ar` command honors this variable to properly handle the object
    files in 64 bit mode.

.. admonition:: Question:

    The :command:`configure` script dies saying:
  
    ``configure: error: Unable to determine pthread library support``.

  Answer:
    Some systems (like BG/L) does not provide a pthread library and
    :command:`configure` claims that cannot find it. Launch the
    :command:`configure` script with the :option:`-disable-pthread` parameter.

.. admonition:: Question:

    :command:`gmake` command fails when compiling the instrumentation package in
    a machine running AIX operating system, using 64 bit mode and IBM XL
    compilers complaining about Profile MPI (PMPI) symbols.

  Answer:
    Use the reentrant version of IBM compilers (``xlc_r`` and ``xlC_r``).  Non
    reentrant versions of MPI library do not include 64 bit MPI symbols, whereas
    reentrant versions do. To use these compilers, set the CC (C compiler) and
    CXX (C++ compiler) environment variables before running the
    :command:`configure` script.

.. admonition:: Question:

    The compiler fails complaining that some parameters can not be understood when
    compiling the parallel merge.

  Answer:
    If the environment has more than one compiler (for example, IBM and GNU
    compilers), is it possible that the parallel merge compiler is not the same
    as the rest of the package. There are two ways to solve this:
  
    * Force the package compilation with the same backend as the parallel
      compiler. For example, for IBM compiler, set ``CC=xlc`` and ``CXX=xlC`` at
      the configure step.
    * Tell the parallel compiler to use the same compiler as the rest of the
      package. For example, for IBM compiler mpcc, set ``MP_COMPILER=gcc`` when
      issuing the make command.

.. admonition:: Question:

    The instrumentation package does not generate the shared instrumentation
    libraries but generates the satatic instrumentation libraries.

  Answer 1:
    Check that the configure step was compiled without
    :option:`--disable-shared` or force it to be enabled through
    :option:`--enable-shared`.

  Answer 2:
    Some MPI libraries (like MPICH 1.2.x) do not generate the shared libraries
    by default. The instrumentation package rely on them to generate its shared
    libraries, so make sure that the shared libraries of the MPI library are
    generated.

.. admonition:: Question:

    In BlueGene systems where the libxml2 (or any optional library for extrae) the
    linker shows error messages like when compiling the final application with the
    |TRACE| library:
  
    * ``../libxml2/lib/libxml2.a(xmlschemastypes.o): In function '_xmlSchemaDateAdd'``
    * ``../libxml2-2.7.2/xmlschemastypes.c:3771: undefined reference to '__uitrunc'``
    * ``../libxml2-2.7.2/xmlschemastypes.c:3796: undefined reference to '__uitrunc'``
    * ``../libxml2-2.7.2/xmlschemastypes.c:3801: undefined reference to '__uitrunc'``
    * ``../libxml2-2.7.2/xmlschemastypes.c:3842: undefined reference to '__uitrunc'``
    * ``../libxml2-2.7.2/xmlschemastypes.c:3843: undefined reference to '__uitrunc'``
    * ``../libxml2/lib/libxml2.a(xmlschemastypes.o): In function 'xmlSchemaGetCanonValue'``
    * ``../libxml2-2.7.2/xmlschemastypes.c:5840: undefined reference to '__f64tou64rz'``
    * ``../libxml2-2.7.2/xmlschemastypes.c:5843: undefined reference to '__f64tou64rz'``
    * ``../libxml2-2.7.2/xmlschemastypes.c:5846: undefined reference to '__f64tou64rz'``
    * ``../libxml2-2.7.2/xmlschemastypes.c:5849: undefined reference to '__f64tou64rz'``
    * ``../libxml2/lib/libxml2.a(debugXML.o): In function 'xmlShell'``
    * ``../libxml2-2.7.2/debugXML.c:2802: undefined reference to '_fill'``
    * ``collect2: ld returned 1 exit status``

  Answer:
    The libxml2 library (or any other optional library) has been compiled using
    the IBM XL compiler. There are two alternatives to circumvent the problem:
    add the XL libraries into the link stage when building your application, or
    recompile the libxml2 library using the GNU gcc cross compiler for BlueGene.

.. admonition:: Question:

    Where do I get the procedure and constant declarations for Fortran?

  Answer:
    You can find a module (ready to be compiled) in
    :file:`${EXTRAE_HOME}/include/extrae_module.f`. To use the module, just
    compile it (do not link it), and then use it in your compiling / linking
    step. If you do not use the module, the trace generation (specially for
    those routines that expect parameters which are not ``INTEGER*4``) can
    result in type errors and thus generate a tracefile that does not honor the
    |TRACE| calls.


.. _sec:FAQexecution:

Execution FAQ
-------------

.. admonition:: Question:

    I executed my application instrumenting with |TRACE|, even though it appears
    that |TRACE| is not intrumenting anything. There is neither any |TRACE|
    message nor any |TRACE| output files (file:`set-X`/:file:`\*.mpit`).

  Answer 1:
    Check that environment variables are correctly passed to the application.

  Answer 2:
    If the code is Fortran, check that the number of underscores used to
    decorate routines in the instrumentation library matches the number of
    underscores added by the Fortran compiler you used to compile and link the
    application.  You can use the :command:`nm` and :command:`grep` commands to
    check it.

  Answer 3:
    If the code is MPI and Fortran, check that you're using the proper Fortran
    library for the instrumentation.

  Answer 4:
    If the code is MPI and you are using :envvar:`LD_PRELOAD`, check that the
    binary is linked against a shared MPI library (you can use the
    :command:`ldd` command).

.. admonition:: Question:

    Why are the environment variables not exported?

  Answer:
    MPI applications are launched using special programs (like
    :command:`mpirun`, :command:`poe`, :command:`mprun`, :command:`srun`, ...)
    that spawn the application for the selected resources. Some of these
    programs do not export all the environment variables to the spawned
    processes. Check if the the launching program does have special parameters
    to do that, or use the approach used on section :ref:`cha:Examples` based on
    launching scripts instead of MPI applications.

.. admonition:: Question:

    The instrumentation begins for a single process instead for several
    processes.

  Answer 1:
    Check that you place the appropriate parameter to indicate the number of
    tasks (typically :option:`-np`).

  Answer 2:
    Some MPI implementation require the application to receive special MPI
    parameters to run correctly. For example, MPICH based on CH-P4 device
    require the binary to receive som paramters. The following example is an
    sh-script that solves this issue:

    .. code-block:: sh
  
      #!/bin/sh
      EXTRAE_CONFIG_FILE=extrae.xml ./mpi_program $@ real_params

.. admonition:: Question:

    The application blocks at the beginning.

  Answer:
    The application may be waiting for all tasks to startup but only some of
    them are running. Check for the previous question.

.. admonition:: Question:

    The resulting traces do not contain the routines that have been
    instrumented.

  Answer 1:
    Check that the routines have been actually executed.

  Answer 2:
    Some compilers do automatic inlining of functions at some optimization
    levels (*e.g.*, Intel Compiler at :option:`-O2`). When functions are
    inlined, they do not have entry and exit blocks and cannot be instrumented.
    Turn off inlining or decrease the optimization level.

.. admonition:: Question:

    Number of threads = 1?

  Answer:
    Some MPI launchers (*i.e.,* :command:`mpirun`, :command:`poe`,
    :command:`mprun`, ...)

.. admonition:: Question:

    When running the instrumented application, the loader complains about:
    ``undefined symbol: clock_gettime``

  Answer:
    The instrumentation package was configured using
    :option:`--enable-posix-clock`` and on many systems this implies the
    inclusion of additional libraries (namely, :option:`-lrt`)


.. _sec:FAQcounters:

Performance monitoring counters FAQ
-----------------------------------

.. admonition:: Question:

    How do I know the available performance counters on the system?

  Answer 1:
    If using PAPI, check the :command:`papi_avail` or
    :command:`papi_native_avail` commands found in the PAPI installation
    directory.

  Answer 2:
    If using PMAPI (on AIX systems), check for the :command:`pmlist` command.
    Specifically, check for the available groups running :command:`pmlist -g -1`.

.. admonition:: Question:

    How do I know how many performance counters can I use?

  Answer:
    The |TRACE| package can gather up to eight (8) performance counters at the
    same time. This also depends on the underlying library used to gather them.

.. admonition:: Question:

    When using PAPI, I cannot read eight performance counters or the specified
    in :command:`papi_avail` output.

  Answer 1:
    There are some performance counters (those listed in :command:`papi_avail`)
    that are classified as derived. Such performance counters depend on more
    than one counter increasing the number of real performance counters used.
    Check for the derived column within the list to check whether a performance
    counter is derived or not.

  Answer 2:
    On some architectures, like the PowerPC, the performance counters are
    grouped in a such way that choosing a performance counter precludes others
    from being elected in the same set. A feasible work-around is to create as
    many sets in the XML file to gather all the required hardware counters and
    make sure that the sets change from time to time.


.. _sec:FAQmerge:

Merging traces FAQ
------------------

.. admonition:: Question:

    The :command:`mpi2prv` command shows the following messages at the start-up:

    ``PANIC! Trace file TRACE.0000011148000001000000.mpit is 16 bytes too big!``
    ``PANIC! Trace file TRACE.0000011147000002000000.mpit is 32 bytes too big!``
    ``PANIC! Trace file TRACE.0000011146000003000000.mpit is 16 bytes too big!``

    and it dies when parsing the intermediate files.

  Answer 1:
    The aforementioned messages are typically related with incomplete writes in
    disk. Check for enough disk space using the :command:`quota` and
    :command:`df` commands.

  Answer 2:
    If your system supports multiple ABIs (for example, linux x86-64 supports 32
    and 64 bits ABIs), check that the ABI of the target application and the ABI
    of the merger match.

.. admonition:: Question:

    The resulting |PARAVER| tracefile contains invalid references to the source
    code.
  
  Answer:
    This usually happens when the code has not been compiled and linked with the
    :option:`-g` flag. Moreover, some high level optimizations (which includes
    inlining, interprocedural analysis, and so on) can lead to generate bad
    references.

.. admonition:: Question:

    The resulting trace contains information regarding the stack (like callers)
    but their value does not coincide with the source code.
  
  Answer:
    Check that the same binary is used to generate the trace and referenced with
    the the :command:`-e` parameter when generating the |PARAVER| tracefile.
