         PROGRAM pi_instrumentedf
         USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_CHAR, C_NULL_CHAR, &
           C_PTR, C_LOC
         USE EXTRAE_MODULE
         IMPLICIT NONE

         INTEGER, PARAMETER :: n = 1000000
         INTEGER*8, PARAMETER, DIMENSION(2) :: values = (/ 0, 1 /)
         REAL*8, PARAMETER :: PI25DT = 3.141592653589793238462643
         CHARACTER(KIND=C_CHAR,LEN=6), DIMENSION(2), TARGET :: & 
           description_values
         TYPE(C_PTR), DIMENSION(2) :: description_values_ptrs
         CHARACTER(KIND=C_CHAR,LEN=20) :: evt_desc = &
           "Kernel execution" // C_NULL_CHAR

         INTEGER :: it
         REAL*8 :: pi, h, area, x

         description_values(1) = "End  " // C_NULL_CHAR
         description_values_ptrs(1) = C_LOC(description_values(1))
         description_values(2) = "Begin" // C_NULL_CHAR
         description_values_ptrs(2) = C_LOC(description_values(2))

         ! CALL extrae_init !! <- This is no longer necessary

         CALL extrae_define_event_type (1000, evt_desc, 2, values, &
           description_values_ptrs)

         h = 1.0 / n

         CALL extrae_event(1000, 1_8)
         DO it=0, 9
             area = pi_kernel (n, h)
         END DO

         CALL extrae_event(1000, 0_8)
         pi = h * area

         WRITE(*,"(A,F20.16,A,F20.16)") "pi is approximately", pi, &
           "Error is", ABS(pi - PI25DT)

         ! CALL extrae_fini !! <- This is no longer necessary

         CONTAINS

         REAL*8 FUNCTION pi_kernel( n, h )
         IMPLICIT NONE

         INTEGER, INTENT(IN) :: n
         REAL*8, INTENT(IN) :: h

         INTEGER :: i
         REAL*8 :: x, tmp

         tmp = 0

         DO i=1, n
             x = h * (i - 0.5)
             tmp = tmp + (4.0 / (1.0 + x*x))
         END DO

         pi_kernel = tmp
         RETURN
         END FUNCTION

         END PROGRAM
