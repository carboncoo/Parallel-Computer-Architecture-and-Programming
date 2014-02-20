#include <algorithm>
#include <cassert>
#include <mpi.h>
#include <new>

using namespace std;

#define ROOT 0

//Single processor serial sort
double serialSort(float *data, int procs, int procId, size_t dataSize, size_t localSize) {
  float *serialData;
  if (procId == ROOT) {
    try {
      serialData = (float *)malloc(sizeof(float) * dataSize);
    } catch (bad_alloc&) {
      printf("@@@ Memory allocation failed for serialData!\n");
      printf("@@@ You have to use distributed sort for data Size %d\n", dataSize);
      return 0.f;
    }
  }

  assert( MPI_Gather(data, dataSize/procs, MPI_FLOAT, 
        serialData, dataSize/procs, MPI_FLOAT,
        0, MPI_COMM_WORLD) == MPI_SUCCESS );

  double startTime = MPI_Wtime();
  if (procId == ROOT) {
    sort(serialData, serialData + dataSize - procs);
  }
  double endTime = MPI_Wtime();
  if (procId == ROOT) {
    free(serialData);
  }
  return endTime - startTime;
}

//Parallel merge sort
double mergeSort(float *data, int procs, int procId, size_t dataSize, size_t localSize) {
  float *serialData;
  float *localData = (float *)malloc(sizeof(float) * localSize);
  memcpy(localData, data, sizeof(float) * localSize);
  if (procId == ROOT) {
    try {
      serialData = (float *)malloc(sizeof(float) * dataSize);
    } catch (bad_alloc&) {
      printf("@@@ Memory allocation failed for serialData!\n");
      printf("@@@ You have to use distributed sort for data Size %d\n", dataSize);
      return 0.f;
    }
  }

  double startTime = MPI_Wtime();

  sort(localData, localData + localSize);
  assert( MPI_Gather(localData, dataSize/procs, MPI_FLOAT,
        serialData, dataSize/procs, MPI_FLOAT,
        ROOT, MPI_COMM_WORLD) == MPI_SUCCESS );
  if (procId == ROOT) {
    for (size_t interval=dataSize/procs; interval<dataSize; interval*=2) {
      for (unsigned int start=0; start<dataSize; start+=(interval*2)) {
        inplace_merge( serialData + start, 
            serialData + min(start + interval, dataSize), 
            serialData + min(start + interval * 2, dataSize) );
      }
    }
  }

  double endTime = MPI_Wtime();
  if (procId == ROOT) {
    free(serialData);
  }
  free(localData);
  return endTime - startTime;
}
