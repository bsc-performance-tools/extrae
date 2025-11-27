 # AX_HIP
# --------------------
AC_DEFUN([AX_HIP],
[
  AX_FLAGS_SAVE()
  # NOTE: These defines were necessary in older HIP versions, 
  # but it seems no longer required. Please verify if removing them
  # has no impact on functionality before deleting.
  AC_ARG_ENABLE(hip-for-nvidia,
      AC_HELP_STRING(
         [--enable-hip-for-nvidia],
         [Enable HIP library to target NVIDIA accelerators]
      ),
      [enable_hip_for_nvidia="${enableval}"],
      [enable_hip_for_nvidia="no"]
   )

  #AM_CONDITIONAL(HIP_FOR_NVIDIA, test "${enable_hip_for_nvidia}" = "yes")

  AC_ARG_WITH(hip,
  AC_HELP_STRING(
    [--with-hip@<:@=DIR@:>@],
    [Enable support for tracing HIP]
  ),
    [hip_root_path="${withval}"],
    [hip_root_path="no"]
  )

  if test "${hip_root_path}" != "no" ; then

    if test "${enable_hip_for_nvidia}" = "yes" ; then
      HIP_PLATFORM_DEFINE="-D__HIP_PLATFORM_NVIDIA__"
    else
      HIP_PLATFORM_DEFINE="-D__HIP_PLATFORM_AMD__"
    fi

    HIP_INCLUDES = "-I${hip_root_path}/hip/include -I${hip_root_path}/include/roctracer -I${hip_root_path}/include/hsa"

    CFLAGS = "${CFLAGS} ${HIP_PLATFORM_DEFINE} ${HIP_INCLUDES}"
    #CPPFLAGS="${CFLAGS}"

    AX_FIND_INSTALLATION([HIP], [${hip_root_path}], [], [], [hip/hip_runtime.h roctracer/roctracer_hip.h roctracer/roctracer_hsa.h], [], [roctracer64], [],
      [ AC_MSG_NOTICE([HIP tracing support enabled]) ],
      [ AC_MSG_ERROR([HIP tracing support is enabled but could not find a valid HIP installation. Check that --with-hip points to the proper HIP directory.]) ]
    )

    HIP_CFLAGS="${HIP_CFLAGS} ${HIP_PLATFORM_DEFINE} ${HIP_INCLUDES}"

    AC_SUBST(HIP_CFLAGS)

  fi

  AM_CONDITIONAL(HAVE_HIP, test "${HIP_INSTALLED}" = "yes")
  AX_FLAGS_RESTORE()
])

# AX_HIP_SHOW_CONFIGURATION
# --------------------
AC_DEFUN([AX_HIP_SHOW_CONFIGURATION],
[
  if test "${HIP_INSTALLED}" = "yes" ; then
		echo HIP instrumentation: yes, through ROCTRACER
		echo -e \\\tHIP : ${HIP_HOME}
  else
    echo HIP instrumentation: no
  fi
])
