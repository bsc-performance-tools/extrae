# AX_CHECK_UCONTEXT
# -----------------
AC_DEFUN([AX_CHECK_UCONTEXT],
[
	AC_MSG_CHECKING([for the name definition of struct ucontext])
	AX_FLAGS_SAVE()

	AC_COMPILE_IFELSE(
		[AC_LANG_PROGRAM(
			[#include <ucontext.h>],
			[
			  struct ucontext *uc;
			  void *p = &uc->uc_mcontext;
			])
		],
		[ STRUCT_UCONTEXT_TYPE="struct ucontext" ],
		[ STRUCT_UCONTEXT_TYPE="unknown" ]
	)

	if test "${STRUCT_UCONTEXT_TYPE}" = "unknown"; then
		AC_COMPILE_IFELSE(
			[AC_LANG_PROGRAM(
				[#include <ucontext.h>],
	 			[
				  ucontext_t *uc;
				  void *p = &uc->uc_mcontext;
				])
			],
	 		[ STRUCT_UCONTEXT_TYPE="ucontext_t" ],
	 		[ STRUCT_UCONTEXT_TYPE="unknown" ]
		)
	fi

	AC_MSG_RESULT([${STRUCT_UCONTEXT_TYPE}])
	if test "${STRUCT_UCONTEXT_TYPE}" = "unknown"; then
		AC_MSG_ERROR([Unknown definition of struct ucontext. Please check the definition in sys/ucontext.h or libc's ucontext.h and extend the configure macro])
	else
		AC_DEFINE_UNQUOTED([STRUCT_UCONTEXT], ${STRUCT_UCONTEXT_TYPE}, [Definition of struct ucontext])
	fi

	AX_FLAGS_RESTORE()
])

