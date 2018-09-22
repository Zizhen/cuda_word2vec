#include <iostream>
#include <string.h>
#include <fstream>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <math.h>
#include "cublas_v2.h"

using namespace std;

int main(){
  cublasHandle_t handle;
  cublasCreate(&handle);

  int m = 3;
  int n = 4;
  int k = 2;
  float *A_h = new float[6];
  A_h[0] = 1;
  A_h[1] = 2;
  A_h[2] = 3;
  A_h[3] = 2;
  A_h[4] = 3;
  A_h[5] = 4;
  float *B_h = new float[8];
  B_h[0] = 1;
  B_h[1] = 4;
  B_h[2] = 2;
  B_h[3] = 3;
  B_h[4] = 3;
  B_h[5] = 2;
  B_h[6] = 4;
  B_h[7] = 1;
  float *C_h = new float[12];
  float *A_d, *B_d, *C_d;
  cudaMalloc((void **)&A_d, 6*sizeof(float));
  cudaMalloc((void **)&B_d, 8*sizeof(float));
  cudaMalloc((void **)&C_d, 12*sizeof(float));
  cudaMemcpy(A_d, A_h, 6*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, 8*sizeof(float), cudaMemcpyHostToDevice);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasSgemm(
    handle, CUBLAS_OP_N, CUBLAS_OP_N,
    m, n, k,
    &alpha,
    A_d, m,
    B_d, k,
    &beta,
    C_d, m
  );
  cudaMemcpy(C_h, C_d, 12*sizeof(float), cudaMemcpyDeviceToHost);
  for(int i = 0; i < 12; i ++){
    cout << C_h[i] << endl;
  }

  return 1;
}
