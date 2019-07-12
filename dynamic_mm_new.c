#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define ROOT 0
#define FROM_ROOT 1
#define FROM_NODE 2

int main(int argc, char *argv[]) {
  //Variables, N is the number of rows and cols for matrix
  int N = 1000, rank, source, dest, numworkers, numtasks, mtype,
      rows, averow, extra, offset;
  
  // * allocation moved to per node to save memory
  double *a, *b, *c;

  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  numworkers = numtasks - 1;

  clock_t start = clock();
  
  //Root Tasks/Node
  if(rank == ROOT) {
    //Initialize the arrays, insert values
    printf("Initializing Arrays. Tasks: %d \n", numtasks);

    // * Allocate each array in one NxN block
    a = (double *)malloc(N * N * sizeof(double));
    b = (double *)malloc(N * N * sizeof(double));
    c = (double *)malloc(N * N * sizeof(double));

    for(int i = 0; i < N; i++) {
      for(int j = 0; j < N; j++) {
        // * Initialize values, use manual row-major addressing to resolve x, y
        a[i*N + j] = i + j;
        b[i*N + j] = i * j;
        c[i*N + j] = 0;
      }
    }

    clock_t mpi_start = clock();
    
    //Send the matrix data to the worker tasks
    averow = N / numworkers;
    extra = N % numworkers;
    offset = 0;

    // * Removed redundant mtype variable
    for(dest = 1; dest <= numworkers; dest++) {
      rows = (dest <= extra) ? averow + 1 : averow;
      //printf("Sending %d rows to ask %d offset=%d\n", rows, dest, offset);
      MPI_Send(&offset, 1, MPI_INT, dest, FROM_ROOT, MPI_COMM_WORLD);
      MPI_Send(&rows, 1, MPI_INT, dest, FROM_ROOT, MPI_COMM_WORLD);

      // * skip computation if rows is zero
      if (rows > 0) {
        // * fix offset for a
        MPI_Send(&a[offset], rows * N, MPI_DOUBLE, dest, FROM_ROOT, MPI_COMM_WORLD);
        MPI_Send(b, N * N, MPI_DOUBLE, dest, FROM_ROOT, MPI_COMM_WORLD);
      }
      offset = offset + (rows * N);
    }

    //Receive results from woker tasks
    // * changed i => source to fix increment
    for(int source = 1; source <= numworkers; source++) {
      MPI_Recv(&offset, 1, MPI_INT, source, FROM_NODE, MPI_COMM_WORLD, &status);
      MPI_Recv(&rows, 1, MPI_INT, source, FROM_NODE, MPI_COMM_WORLD, &status);
      // * fix offset for c
      MPI_Recv(&c[offset], rows * N, MPI_DOUBLE, source, FROM_NODE, MPI_COMM_WORLD, &status);
      printf("Received results from tasks %d\n", source);
    }
    clock_t mpi_end = clock() - mpi_start;
    clock_t end = clock() - start;
    double time_spent = (double)(end - start)/CLOCKS_PER_SEC;
    double mpi_time = (double)(mpi_end - mpi_start)/CLOCKS_PER_SEC;
    //Print results
    printf("Results: \n");

    for(int i = 0; i < N; i++) {
      printf("\n");

      for(int j = 0; j < N; j++) {
        printf("%6.2f   ", c[i*N+j]);
      }
    }
    
    printf("\n Done \n");
    printf("Total Time: %f seconds\n", time_spent);
    printf("MPI Communication: %f seconds\n",mpi_time);
  }

  //Workers
  if(rank > ROOT) {
    // * Removed redundant mtype variable
    //Receive data from ROOT
    MPI_Recv(&offset, 1, MPI_INT, ROOT, FROM_ROOT, MPI_COMM_WORLD, &status);
    MPI_Recv(&rows, 1, MPI_INT, ROOT, FROM_ROOT, MPI_COMM_WORLD, &status);

    // * skip computation if rows is zero
    if (rows > 0) {
      // * a and c arrays can be allocated with reduced size on compute nodes
      a = (double *)malloc(rows * N * sizeof(double));
      b = (double *)malloc(N * N * sizeof(double));
      c = (double *)malloc(rows * N * sizeof(double));

      MPI_Recv(a, rows * N, MPI_DOUBLE, ROOT, FROM_ROOT, MPI_COMM_WORLD, &status);
      MPI_Recv(b, N * N, MPI_DOUBLE, ROOT, FROM_ROOT, MPI_COMM_WORLD, &status);
      //Calculation/Multiplication
      for(int k = 0; k < N; k++) {
        // * Increment only inside the allocated work area, normalized to zero
        for(int i = 0; i < rows; i++) {
	    //c[i*N + k] = 0;
          for(int j = 0; j < N; j++) {
            c[i*N + k] += a[i*N + j] * b[j*N + k];
          }
        }
      }
    }
    //Send results back to ROOT
    MPI_Send(&offset, 1, MPI_INT, ROOT, FROM_NODE, MPI_COMM_WORLD);
    MPI_Send(&rows, 1, MPI_INT, ROOT, FROM_NODE, MPI_COMM_WORLD);
    MPI_Send(c, rows * N, MPI_DOUBLE, ROOT, FROM_NODE, MPI_COMM_WORLD);
  }

  MPI_Finalize();
}
