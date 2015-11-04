      program hello
      implicit none
      integer rank, size, ierror
   
      call MPI_INIT(ierror)
      call MPI_FINALIZE(ierror)
      end
