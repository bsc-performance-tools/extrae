      program hello
      implicit none
      integer rank, size, ierror
      include 'mpif.h'
   
      call MPI_INIT(ierror)
      call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierror)
      call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierror)
      call MPI_FINALIZE(ierror)
      end
