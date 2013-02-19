      PROGRAM pi

      integer i
      real x, sum, h

      h = 1.0 / 10000000

!$OMP PARALLEL PRIVATE(X, i)
!$OMP DO REDUCTION (+:sum)
      DO i = 1, 10000000, 1
       x = h*(i-0.5)
       sum = sum + 4.0/(1.0+x**2)
      ENDDO
!$OMP END DO 
!$OMP END PARALLEL


!$OMP PARALLEL PRIVATE(X, i)
!$OMP DO REDUCTION (+:sum)
      DO i = 1, 10000000, 1
       x = h*(i-0.5)
       sum = sum + 4.0/(1.0+x**2)
      ENDDO
!$OMP END DO 
!$OMP END PARALLEL

      write (*,*) "pi = ",sum*h
 
      stop

      END
