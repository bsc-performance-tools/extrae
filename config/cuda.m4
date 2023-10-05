# AX_CUDA
# --------------------
AC_DEFUN([AX_CUDA],
[
  AX_FLAGS_SAVE()
  AC_ARG_WITH(cuda,
  AC_HELP_STRING(
    [--with-cuda@<:@=DIR@:>@],
    [Enable support for tracing CUDA]
  ),
    [cuda_root_path="${withval}"],
    [cuda_root_path="no"]
  )

  if test "${cuda_root_path}" != "no" ; then
    AX_FIND_INSTALLATION([CUDA], [${cuda_root_path}], [nvcc], [], [cuda_runtime_api.h], [], [cudart], [],
      [ AC_MSG_NOTICE([CUDA tracing support enabled]) ],
      [ AC_MSG_ERROR([CUDA tracing support is enabled but could not find a valid CUDA installation. Check that --with-cuda points to the proper CUDA directory.]) ]
    )
  fi

  AM_CONDITIONAL(HAVE_CUDA, test "${CUDA_INSTALLED}" = "yes")
  AX_FLAGS_RESTORE()
])

# AX_CUPTI
# --------------------
AC_DEFUN([AX_CUPTI],
[
  AX_FLAGS_SAVE()
  AC_REQUIRE([AX_CUDA])

  if test "${CUDA_INSTALLED}" = "yes"; then
    AC_ARG_WITH(cupti,
      AC_HELP_STRING(
        [--with-cupti],
        [Enable support for tracing CUDA through CUPTI interface]
      ),
      [cuda_cupti_path="${withval}"],
      [cuda_cupti_path="${CUDA_HOME}/extras/CUPTI"]
    )

    if test "$cuda_cupti_path" != "no"; then
      AX_FIND_INSTALLATION([CUPTI], [${cuda_cupti_path}], [], [], [cupti.h cupti_events.h], [], [cupti], [],
        [ AC_MSG_NOTICE([CUDA tracing support enabled through CUPTI interface]) ],
        [ AC_MSG_ERROR([CUPTI tracing support is enabled but could not find valid CUPTI installation. Check that CUPTI is available for the CUDA installation pointed by --with-cuda.]) ]
      )
    fi
  fi

# Checks for deprecated CUPTI calls to define their parameter structures if not
# found. This is to allow compatibility with older CUDA versions even if Extrae
# is compiled with a newer one.
  AC_MSG_CHECKING([for cudaConfigureCall_v3020_params definition])
  if test "${CUPTI_INSTALLED}" = "yes"; then
    CFLAGS="$CFLAGS $CUDA_CFLAGS"
    AC_COMPILE_IFELSE(
      [AC_LANG_PROGRAM(
        [#include <cupti.h>],
        [
          cudaConfigureCall_v3020_params *p;
        ])
      ],
      [
        AC_MSG_RESULT([yes])
        AC_DEFINE([HAVE_CUDACONFIGURECALL_v3020], 1, [Define to 1 if cudaConfigureCall_v3020_params struct is available in the CUPTI installation.])
      ],
      [
        AC_MSG_RESULT([no])
      ]
    )
  fi

  AC_MSG_CHECKING([for cudaLaunch_v3020_params definition])
  if test "${CUPTI_INSTALLED}" = "yes"; then
    CFLAGS="$CFLAGS $CUDA_CFLAGS"
    AC_COMPILE_IFELSE(
      [AC_LANG_PROGRAM(
        [#include <cupti.h>],
        [
          cudaLaunch_v3020_params *p;
        ])
      ],
      [
        AC_MSG_RESULT([yes])
        AC_DEFINE([HAVE_CUDALAUNCH_v3020], 1, [Define to 1 if cudaLaunch_v3020_params struct is available in the CUPTI installation.])
      ],
      [
        AC_MSG_RESULT([no])
      ]
    )
  fi

  AC_MSG_CHECKING([for cudaStreamDestroy_v3020_params definition])
  if test "${CUPTI_INSTALLED}" = "yes"; then
    CFLAGS="$CFLAGS $CUDA_CFLAGS"
    AC_COMPILE_IFELSE(
      [AC_LANG_PROGRAM(
        [#include <cupti.h>],
        [
          cudaStreamDestroy_v3020_params *p;
        ])
      ],
      [
        AC_MSG_RESULT([yes])
        AC_DEFINE([HAVE_CUDASTREAMDESTROY_v3020], 1, [Define to 1 if cudaStreamDestroy_v3020_params struct is available in the CUPTI installation.])
      ],
      [
        AC_MSG_RESULT([no])
      ]
    )
  fi

  AM_CONDITIONAL(HAVE_CUPTI, test "${CUPTI_INSTALLED}" = "yes")
  AX_FLAGS_RESTORE()
])

# AX_CUDA_SHOW_CONFIGURATION
# --------------------
AC_DEFUN([AX_CUDA_SHOW_CONFIGURATION],
[
	if test "${CUPTI_INSTALLED}" = "yes" ; then
		echo CUDA instrumentation: yes, through CUPTI
		echo -e \\\tCUDA : ${CUDA_HOME}
		echo -e \\\tCUPTI: ${CUPTI_HOME}
	else
		if test "${CUDA_INSTALLED}" = "yes" ; then
			echo CUDA instrumentation: yes, through wrappers
			echo -e \\\tCUDA: ${CUDA_HOME}
		else
			echo CUDA instrumentation: no
		fi
	fi
])
