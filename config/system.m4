# AX_SYSTEM_TYPE
# --------------------
AC_DEFUN([AX_SYSTEM_TYPE],
[
	AC_ARG_ENABLE(mic,
	   AC_HELP_STRING(
	      [--enable-mic],
	      [Enable compilation for the Intel MIC architecture (disabled by default; needed when cross-compiling for Intel MIC/Xeon Phi)]
	   ),
	   [enable_mic="${enableval}"],
	   [enable_mic="no"]
	)
	IS_MIC_MACHINE=${enable_mic}

	AC_ARG_ENABLE(arm,
	   AC_HELP_STRING(
	      [--enable-arm],
	      [Enable compilation for ARM architecture (disabled by default; needed when cross-compiling for ARM)]
	   ),
	   [enable_arm="${enableval}"],
	   [enable_arm="no"]
	)
	IS_ARM_MACHINE=${enable_arm}
	if test "${IS_ARM_MACHINE}" = "yes" ; then
		target_cpu="arm"
		target_os="linux"
	fi

	AC_ARG_ENABLE(arm64,
	   AC_HELP_STRING(
	      [--enable-arm64],
	      [Enable compilation for ARM64 architecture (disabled by default; needed when cross-compiling for ARM64/AARCH64)]
	   ),
	   [enable_arm64="${enableval}"],
	   [enable_arm64="no"]
	)
	IS_ARM64_MACHINE=${enable_arm64}
	if test "${IS_ARM64_MACHINE}" = "yes" ; then
		target_cpu="aarch64"
		target_os="linux"
	fi

	AC_ARG_ENABLE(sparc64,
	   AC_HELP_STRING(
	      [--enable-sparc64],
	      [Enable compilation for SPARC64 architecture (disabled by default; needed when cross-compiling for SPARC64)]
	   ),
	   [enable_sparc64="${enableval}"],
	   [enable_sparc64="no"]
	)
	IS_SPARC64_MACHINE=${enable_sparc64}
	if test "${IS_SPARC64_MACHINE}" = "yes" ; then
		target_cpu="sparc64"
		target_os="linux"
	fi

	AC_ARG_ENABLE(riscv64,
	   AC_HELP_STRING(
	      [--enable-riscv64],
	      [Enable compilation for RISCV64 architecture (disabled by default; needed when cross-compiling for RISCV64)]
	   ),
	   [enable_riscv64="${enableval}"],
	   [enable_riscv64="no"]
	)
	IS_RISCV64_MACHINE=${enable_riscv64}
	if test "${IS_RISCV64_MACHINE}" = "yes" ; then
		target_cpu="riscv64"
		target_os="linux"
	fi

    AC_ARG_ENABLE(powerpc64le,
       AC_HELP_STRING(
          [--enable-powerpc64le],
          [Enable compilation for powerpc64le architecture (disabled by default; needed when cross-compiling for powerpc64le)]
       ),
       [enable_powerpc64le="${enableval}"],
       [enable_powerpc64le="no"]
    )
    IS_POWERPC64LE_MACHINE=${enable_powerpc64le}
    if test "${IS_POWERPC64LE_MACHINE}" = "yes" ; then
        target_cpu="powerpc64le"
        target_os="linux"
    fi

	# Check if this is an Altix machine and if it has an /dev/mmtimer device
	# (which is a global clock!)
	AC_ARG_ENABLE(check-altix,
	   AC_HELP_STRING(
	      [--enable-check-altix],
	      [Enable check to known if this is an Altix machine (enabled by default)]
	   ),
	   [enable_check_altix="${enableval}"],
	   [enable_check_altix="yes"]
	)
	if test "${enable_check_altix}" = "yes" ; then
	   AX_IS_ALTIX_MACHINE
	   AX_HAVE_MMTIMER_DEVICE
	fi

	AX_IS_CRAY_XT
	if test "${IS_CXT_MACHINE}" = "yes" ; then
	  IS_CRAY_MACHINE="yes"
	fi
    AM_CONDITIONAL(IS_CRAY_MACHINE, test "${IS_CXT_MACHINE}" = "yes")

	AX_IS_BGL_MACHINE
	AX_IS_BGP_MACHINE
	AX_IS_BGQ_MACHINE
	if test "${IS_BGL_MACHINE}" = "yes" -o "${IS_BGP_MACHINE}" = "yes" -o "${IS_BGQ_MACHINE}" = "yes" ; then
	  AC_DEFINE([IS_BG_MACHINE], 1, [Defined if this machine is a BG machine])
	  IS_BG_MACHINE="yes"
      cross_compiling="no"
	fi
	AM_CONDITIONAL(IS_BG_MACHINE, test "${IS_BGL_MACHINE}" = "yes" -o "${IS_BGP_MACHINE}" = "yes" -o "${IS_BGQ_MACHINE}" = "yes")

	# Write defines in the output header file for the architecture and operating system
	case "${target_cpu}" in
	  arm*|aarch64*) Architecture="arm"
	             AC_DEFINE([ARCH_ARM], [1], [Define if architecture is ARM])
                 if test "${target_cpu}" == "aarch64" ; then
	                AC_DEFINE([ARCH_ARM64], [1], [Define if architecture is ARM64/AARCH64])
                 fi
                 ;;
	  i*86|x86_64|amd64)
	             Architecture="ia32"
	             AC_DEFINE([ARCH_IA32], [1], [Define if architecture is IA32])
	             if test "${target_cpu}" = "amd64" -o "${target_cpu}" = "x86_64" ; then
	                AC_DEFINE([ARCH_IA32_x64], [1], [Define if architecture is IA32 (with 64bit extensions)])
	             fi
	             ;;
	  powerpc* ) Architecture="powerpc"
	             AC_DEFINE([ARCH_PPC], [1], [Define if architecture is PPC]) ;;
	  ia64     ) Architecture="ia64"
	             AC_DEFINE([ARCH_IA64], [1], [Define if architecture is IA64]) ;;
	  alpha*   ) Architecture="alpha"
	             AC_DEFINE([ARCH_ALPHA], [1], [Define if architecture is ALPHA]) ;;
	  mips     ) Architecture="mips"
	             AC_DEFINE([ARCH_MIPS], [1], [Define if architecture is MIPS]) ;;
	  sparc64  ) Architecture="sparc64"
	             AC_DEFINE([ARCH_SPARC64], [1], [Define if architecture is SPARC64]) ;;
	  riscv64  )
	             Architecture="riscv"
				 if test "${target_cpu}" == "riscv64" ; then
	                AC_DEFINE([ARCH_RISCV64], [1], [Define if architecture is RISCV64])
	             fi
	             ;;
	esac
	
	case "${target_os}" in
      *android*) OperatingSystem="android"
                 AC_DEFINE([OS_ANDROID], [1], [Define if operating system is Android]) ;;
	  linux*   ) OperatingSystem="linux"
	             AC_DEFINE([OS_LINUX], [1], [Define if operating system is Linux]) ;;
	  aix*     ) OperatingSystem="aix"
	             AC_DEFINE([OS_AIX], [1], [Define if operating system is AIX]) ;;
	  osf*     ) OperatingSystem="dec"
	             AC_DEFINE([OS_DEC], [1], [Define if operating system is DEC]) ;;
	  irix*    ) OperatingSystem="irix"
	             AC_DEFINE([OS_IRIX], [1], [Define if operating system is IRIX]) ;;
	  freebsd* ) OperatingSystem="freebsd"
	             AC_DEFINE([OS_FREEBSD], [1], [Define if operating system is FreeBSD]) ;;
	  solaris* ) OperatingSystem="solaris"
	             AC_DEFINE([OS_SOLARIS], [1], [Define if operating system is Solaris]) ;;
	  darwin*  ) OperatingSystem="darwin"
	             AC_DEFINE([OS_DARWIN], [1], [Define if operating system is Darwin]) ;;
	esac
	
	# Publish these defines for conditional compilation 
	AM_CONDITIONAL(ARCH_IA32,    test "${Architecture}"    = "ia32"    )
	AM_CONDITIONAL(ARCH_POWERPC, test "${Architecture}"    = "powerpc" )
	AM_CONDITIONAL(ARCH_IA64,    test "${Architecture}"    = "ia64"    )
	AM_CONDITIONAL(ARCH_ALPHA,   test "${Architecture}"    = "alpha"   )
	AM_CONDITIONAL(ARCH_MIPS,    test "${Architecture}"    = "mips"    )
	AM_CONDITIONAL(ARCH_SPARC64, test "${Architecture}"    = "sparc64" )
	AM_CONDITIONAL(ARCH_RISCV64, test "${Architecture}"    = "riscv64" )
	 
    AM_CONDITIONAL(OS_ANDROID,   test "${OperatingSystem}" = "android" )
	AM_CONDITIONAL(OS_LINUX,     test "${OperatingSystem}" = "linux"   )
	AM_CONDITIONAL(OS_AIX,       test "${OperatingSystem}" = "aix"     )
	AM_CONDITIONAL(OS_DEC,       test "${OperatingSystem}" = "dec"     )
	AM_CONDITIONAL(OS_IRIX,      test "${OperatingSystem}" = "irix"    )
	AM_CONDITIONAL(OS_FREEBSD,   test "${OperatingSystem}" = "freebsd" )
	AM_CONDITIONAL(OS_DARWIN,    test "${OperatingSystem}" = "darwin" )
	AM_CONDITIONAL(OS_SOLARIS,   test "${OperatingSystem}" = "solaris" )

	# Special flags for specific systems or architectures	
	if test "${OperatingSystem}" = "freebsd" ; then
		CFLAGS="${CFLAGS} -I/usr/local/include"
	fi
])

