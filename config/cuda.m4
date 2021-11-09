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
