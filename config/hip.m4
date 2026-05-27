# AX_HIP
# --------------------
AC_DEFUN([AX_HIP],
[
  AX_FLAGS_SAVE()
  AC_ARG_WITH(hip,
  AC_HELP_STRING(
    [--with-hip@<:@=DIR@:>@],
    [Enable support for tracing HIP]
  ),
    [hip_root_path="${withval}"],
    [hip_root_path="no"]
  )

  if test "${hip_root_path}" != "no" ; then

    HIP_PLATFORM_DEFINE="-D__HIP_PLATFORM_AMD__"

    HIP_INCLUDES="-I${hip_root_path}/include"

    CFLAGS="${CFLAGS} ${HIP_PLATFORM_DEFINE} ${HIP_INCLUDES}"

    AX_FIND_INSTALLATION([HIP], [${hip_root_path}], [], [],
      [hip/hip_runtime.h hip/hip_runtime_api.h], [],
      [amdhip64], [],
      [ AC_MSG_NOTICE([HIP runtime support enabled]) ],
      [ AC_MSG_ERROR([HIP support is enabled but could not find a valid HIP installation.]) ]
    )

    HIP_CFLAGS="${HIP_CFLAGS} ${HIP_PLATFORM_DEFINE} ${HIP_INCLUDES}"
    AC_SUBST(HIP_CFLAGS)
  fi

  AM_CONDITIONAL(HAVE_HIP, test "${HIP_INSTALLED}" = "yes")

  AX_FLAGS_RESTORE()
])

# AX_ROCTRACER
# --------------------
AC_DEFUN([AX_ROCTRACER],
[
  AX_FLAGS_SAVE()
  AC_REQUIRE([AX_HIP])

  if test "${HIP_INSTALLED}" = "yes"; then
    CFLAGS="${HIP_CFLAGS} ${CFLAGS}"

    AC_ARG_WITH(roctracer,
      AC_HELP_STRING(
        [--with-roctracer@<:@=DIR@:>@],
        [Enable ROCtracer-based HIP tracing. Without DIR, searches the native path
         (HIP_HOME). Use --with-roctracer=DIR for a custom location,
         or --without-roctracer to disable ROCtracer and fall back to wrapper instrumentation.]
      ),
      [hip_roctracer_path="${withval}"],
      [hip_roctracer_path="auto"]
    )

    if test "${hip_roctracer_path}" = "auto"; then
      hip_roctracer_native="${HIP_HOME}"
      AX_FIND_INSTALLATION([ROCTRACER], [${hip_roctracer_native}], [], [],
        [roctracer/roctracer.h roctracer/roctracer_hip.h], [],
        [roctracer64], [],
        [ AC_MSG_NOTICE([HIP tracing support enabled through ROCtracer interface]) ],
        [ AC_MSG_ERROR([ROCtracer not found at native path '${hip_roctracer_native}'. Specify a custom path with --with-roctracer=DIR, or use --without-roctracer to disable ROCtracer and fall back to wrapper instrumentation.]) ]
      )
      if test "${ROCTRACER_INSTALLED}" = "yes"; then
        ROCTRACER_CFLAGS="${ROCTRACER_CFLAGS} -I${hip_roctracer_native}/include/roctracer"
        AC_SUBST(ROCTRACER_CFLAGS)
      fi
    elif test "${hip_roctracer_path}" != "no"; then
      AX_FIND_INSTALLATION([ROCTRACER], [${hip_roctracer_path}], [], [],
        [roctracer/roctracer.h roctracer/roctracer_hip.h], [],
        [roctracer64], [],
        [ AC_MSG_NOTICE([HIP tracing support enabled through ROCtracer interface]) ],
        [ AC_MSG_ERROR([ROCtracer not found at specified path '${hip_roctracer_path}'. Check the directory passed to --with-roctracer.]) ]
      )
      ROCTRACER_CFLAGS="${ROCTRACER_CFLAGS} -I${hip_roctracer_path}/include/roctracer"
      AC_SUBST(ROCTRACER_CFLAGS)
    fi
  fi

  AM_CONDITIONAL(HAVE_ROCTRACER, test "${ROCTRACER_INSTALLED}" = "yes")

  AX_FLAGS_RESTORE()
])

# AX_HIP_SHOW_CONFIGURATION
# --------------------
AC_DEFUN([AX_HIP_SHOW_CONFIGURATION],
[
  if test "${ROCTRACER_INSTALLED}" = "yes" ; then
    echo HIP instrumentation: yes, through ROCTRACER
    echo -e \\\tHIP     : ${HIP_HOME}
    echo -e \\\tROCTRACER: ${ROCTRACER_HOME}
  else
    if test "${HIP_INSTALLED}" = "yes" ; then
      echo HIP instrumentation: yes, through wrappers \(DEPRECATED\)
      echo -e \\\tHIP: ${HIP_HOME}
      echo -e \\\tWARNING: GPU wrapper instrumentation is deprecated and may be removed in a future release.
      echo -e \\\t         Enable ROCtracer for full GPU tracing support: --with-roctracer=DIR
    else
      echo HIP instrumentation: no
    fi
  fi
])