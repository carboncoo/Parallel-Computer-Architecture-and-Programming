#ifndef _DATAGEN_H_
#define _DATAGEN_H_

void uniform(float *data, size_t size, float min, float max, int procId);

void exponential(float *data, size_t size, float lambda, int procId);

void almostSort(float *data, size_t size, size_t swap);

#endif
