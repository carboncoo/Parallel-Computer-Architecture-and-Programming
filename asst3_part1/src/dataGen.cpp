#include <random>
#include <cstdlib>
#include <algorithm>

using namespace std;

#define SEED 0

#define GET_DATA(data, size, dist, gen) \
  for (size_t i=0; i<size; i++) data[i] = distribution(generator);

extern int procId;

void uniform(float *data, size_t size, float min, float max, int procId) {
  default_random_engine generator(SEED+procId);
  uniform_real_distribution<float> distribution(min, max);

  GET_DATA(data, size, distribution, generator)
}

void exponential(float *data, size_t size, float lambda, int procId) {
  default_random_engine generator(SEED+procId);
  exponential_distribution<float> distribution(lambda);

  GET_DATA(data, size, distribution, generator)
}

void almostSort(float *data, size_t size, size_t swap) {
  for (size_t i=0; i<swap; i++) {
    size_t index = rand()%size;
    sort(data+index, data+index+1);
  }
}
