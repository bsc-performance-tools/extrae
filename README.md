Extrae
======

A dynamic instrumentation package to trace programs compiled and run with the
shared memory model (like OpenMP and pthreads), the message passing (MPI)
programming model or both programming models (different MPI processes using
OpenMP or pthreads within each MPI process). **Extrae** generates trace files
that can be later visualized with **Paraver**.


Optional dependencies
---------------------

Please, make sure to install the development versions of all packages (-dev,
-devel) when available.

* [libunwind (>=1.0)](http://www.nongnu.org/libunwind)  
	Used to access the callstack within **Extrae**. This lets the analyst gather
	MPI call-sites and emit information on manually added events. Required for
	Intel x86_64 and ia64 architectures.
* [PAPI](http://icl.cs.utk.edu/papi)  
	Used to access the HW counters of the microprocessor. This increases the
	richness of the gathered traces. It is highly recommended to install and use
	PAPI with extrae, altough on some old kernels this may require patching.
* **libiberty and libbfd** (from the binutils package)  
	These two libraries are required to translate gathered application addresses
	into source code information (file name, address line and function name).
	Highly recommended.
* [libxml2 (>=2.5.0)](http://www.xmlsoft.org)  
	Used to parse the configuration of the instrumentation package instead of
	using environment variables. Highly recommended.
* [Dyninst](http://www.dyninst.org)
	Support is in experimental status. It is known to work on Linux PPC32/64
	systems and on Linux x86/x86_64 systems.
* [libz](http://www.zlib.net)  
	libiberty and libbfd may require libz. Also libz may be used to generate
	compressed traces directly.
* **MPI**  
  Execute MPI jobs. Tested with MPICH, MPICH2 and OpenMPI. Others may work.
* **OpenMP** runtime (gcc, ibm or icc)  
	Execute OpenMP jobs.
* **CUDA** / **CUPTI**  
	Execute CUDA-based applications.
* **OpenCL**  
	Execute OpenCL-based applications.


Installation instructions
-------------------------

If installing from Git, you will need to generate a configure file by running:

```sh
$ ./boostrap
```

then continue with installation as if from a release.

Refer to the [INSTALL](./INSTALL) file for general installation instructions.  
Refer to the [INSTALL-examples](./INSTALL-examples) file for examples of
specific installation instructions.
