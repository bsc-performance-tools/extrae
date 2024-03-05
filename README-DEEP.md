## Contact
**General support**: extrae@bsc.es

## Use Cases
### 1. Instrument applications and collect performance data during the program execution to generate a Paraver trace
* *Input:* Application binary and execution script
* *Output:* A [Paraver](https://github.com/bsc-performance-tools/wxparaver/blob/master/README-DEEP.md) trace file recording the activity each process/thread as a timestamped series of events, states and communications
* *Limits:*
    * Supported programming models: MPI, OpenMP, POSIX threads, OpenACC, OpenCL, CUDA, CUPTI, GASPI

### 2. Collect uncore counter data for PROFET memory analysis
* *Input:* Same as [1. Instrument applications and collect performance data during the program execution to generate a Paraver trace](#1-Instrument-applications-and-collect-performance-data-during-the-program-execution-to-generate-a-Paraver-trace)
* *Output:* Same as [1. Instrument applications and collect performance data during the program execution to generate a Paraver trace](#1-Instrument-applications-and-collect-performance-data-during-the-program-execution-to-generate-a-Paraver-trace), including read and write operations of the memory controllers
* *Limits:*
    * Extrae requires additional processes to collect this information, the user needs to allocate resources taking this into account.

The above use cases can be combined.

## Installation

The full documentation of how to install and use Extrae is available online as a [webpage](https://tools.bsc.es/doc/html/extrae) and as a
[downloadable PDF](https://tools.bsc.es/doc/pdf/extrae.pdf). Both versions are updated automatically with every public release of Extrae. Installation examples for different systems can be found in the ```INSTALL-examples``` file in the distribution. A basic installation guide follows:

### Dependencies

Extrae supports the most common parallel programming models, namely MPI, OpenMP, POSIX Threads, OmpSs, OpenCL, CUDA, OpenACC and GASPI. To install Extrae with support for any of them, a prior installation of the runtime is required, and the following dependencies are recommended:

* [PAPI](https://icl.utk.edu/papi/) (to read hardware counters)
* [libunwind](https://www.nongnu.org/libunwind/) (to record MPI and sampling call-sites)
* [GNU Binutils](https://www.gnu.org/software/binutils) (to translate call-sites into file, function and line number)
* [libxml2](http://xmlsoft.org/xslt/downloads.html) >= 2.5.0 (to enable fine-grain configuration options)
* [libz](https://zlib.net) (to generate compressed tracefiles)

In addition, the GNU Autotools build system is also required.

### EasyBuild
```
eb --software=Extrae,4.0.6 --toolchain={gpsmpi,2022a|ipsmpi,2022a}
```
### Spack
```
spack install extrae@4.0.6 mpi=psmpi +papi
```
### From sources

Extrae can be installed by cloning the official git repository:

```
git clone https://github.com/bsc-performance-tools/extrae
```

First run the ```bootstrap``` command in the repository root. This will generate the necessary files to compile the package. Once finished, run the ```configure``` command using the modifiers adapted to the support you want to install. The following example is tuned for the DEEP system using production Stages/2023 with compatibility for the GCC/11.3.0 and ParaStationMPI/5.8.1-1 toolchain:

```
  ./configure \
  --with-binutils=${STAGES}/2023/software/binutils/2.38-GCCcore-11.3.0 \
  --with-mpi=${STAGES}/2023/software/psmpi/5.8.1-1-GCC-11.3.0 \
  --with-papi=${STAGES}/2023/software/PAPI/7.0.0-GCCcore-11.3.0 \
  --with-unwind=${STAGES}/2023/software/libunwind/1.6.2-GCCcore-11.3.0 \
  --with-libz=${STAGES}/2023/software/zlib/1.2.12-GCCcore-11.3.0 \
  --with-cuda=${STAGES}/2023/software/CUDA/11.7 \
  --with-cupti=${STAGES}/2023/software/CUDA/11.7/extras/CUPTI \
  --without-dyninst \
  --enable-posix-clock --enable-openmp --enable-sampling \
  --prefix=<desired_install_path>
```
Then compile and install Extrae:
```
  make && make install
```
After the installation, the tracing libraries for the supported runtimes can be found under <desired_install_path>/lib. The following table shows the runtimes supported by the different libraries:


| Library          | MPI | OpenMP | CUDA | pthreads |
| ---------------- | --- | ------ | ---- | -------- |
| libmpitrace      |  ✅ |        |      |          |
| libomptrace      |     |   ✅   |      |          |
| libcudatrace     |     |        |  ✅  |          |
| libpttrace       |     |        |      |    ✅    |
| libompitrace     |  ✅ |   ✅   |      |          |
| libcudampitrace  |  ✅ |        |  ✅  |          |
| libptmpitrace    |  ✅ |        |      |    ✅    |
| libcudaompitrace |  ✅ |   ✅   |  ✅  |          |


## Getting Started
Extensive documentation can be found in the Extrae User Guide. See the ["Tools manuals"](https://tools.bsc.es/tools_manuals) sub-page on the ["BSC-Tools website"](https://tools.bsc.es) for ["HTML"](https://tools.bsc.es/doc/html/extrae/index.html) and ["PDF"](https://tools.bsc.es/doc/pdf/extrae.pdf) versions. A quick start guide is also available in the ["Getting your first trace"](https://tools.bsc.es/getting-your-first-trace) section. A more detailed guide on how to collect uncore counters data with Extrae to conduct PROFET memory analysis follows.

### Obtaining traces with optional data for PROFET

Extrae uses ```LD_PRELOAD``` interposition to capture calls to the parallel runtimes. This allows to instrument unmodified production binaries without the need to recompile nor relink them. The user needs to create an Extrae launching script, typically named ```trace.sh```, and make it executable. This script will be used to configure and run Extrae to capture the application's parallel activity. An example ```trace.sh``` script to instrument pure MPI applications may look like the example below (other examples are available under ```${EXTRAE_HOME}/share/examples}```):
```
    #!/usr/bin/env bash

    export EXTRAE_HOME=<path-to-extrae-installation>
    export EXTRAE_CONFIG_FILE=extrae.xml
    export LD_PRELOAD=${EXTRAE_HOME}/lib/libmpitrace.so

    $*
```
Where ```LD_PRELOAD``` points to the instrumentation library with MPI support, and ```EXTRAE_CONFIG_FILE``` points to Extrae's configuration file that can be copied from ```${EXTRAE_HOME}/share/example/MPI```.

At this point, the application can be launched by prepending the Extrae's launching script ```trace.sh``` just before the application's binary. Continuing with the MPI application example, the command would look like:
```
    srun ./trace.sh <app_binary>
```

To optionally collect uncore counter data for PROFET analysis, the user needs to:

1. Tune counters/uncore in extrae.xml configuration file to specify which counters to measure.
```
    <counters enabled="yes">
      <cpu ...
      </cpu>
      <uncore enabled="yes" period="10m">
        UNC_M_CAS_COUNT:RD, UNC_M_CAS_COUNT:WR
      </uncore>
      ...
    </counters>
```

2. Use the new launcher script ```extrae-uncore``` with the ```--dry-run``` flag to estimate the necessary additional resources to collect uncore data. We also recommend to add the ```--balance``` flag, which assigns the same number of processes to each socket of the node to avoid imbalances.
```
    ${EBROOTEXTRAE}/bin/extrae-uncore --dry-run --balance <extrae_xml_config_file> <extrae_tracing_library.so> <app_binary>

    Extrae: Will need <EXTRA_PROCS> processes/node to measure uncore counters
```

3. Extend the Slurm allocation to fit the new processes. If using ParaStationMPI also add the ```export SLURM_EXACT=1``` to allow the spawning of new processes at runtime, as well as ```export EXTRAE_SKIP_AUTO_LIBRARY_INITIALIZE=1```. Use the new launcher script to execute the application (this replaces the former ```trace.sh``` script).
```
    #!/usr/bin/env sh

    #SBATCH -A deep
    #SBATCH -N APP_NODES
    #SBATCH -n APP_PROCS+EXTRA_PROCS
    #SBATCH -p dp-cn

    # Use EasyBuild
    module load GCC/11.3.0 ParaStationMPI/5.8.1-1 Extrae/4.0.6

    # Use Spack
    # spack load extrae@4.0.6

    # Required for ParaStationMPI
    export EXTRAE_SKIP_AUTO_LIBRARY_INITIALIZE=1
    export SLURM_EXACT=1

    srun -n APP_PROCS ${EBROOTEXTRAE}/bin/extrae-uncore --balance <extrae_xml_config_file> <extrae_tracing_library.so> <app_binary>
```

The tracing libraries available for the argument ```<extrae_tracing_library>``` can be found under ```$EBROOTEXTRAE/lib```.
