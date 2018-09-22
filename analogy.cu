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

// Compile with:
// nvcc -o example example.cu
#define ERROR_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__
void normalize_T(float* mat, float* normSum_d, float* matrixNorm_d, float* matrixNorm_T, int word_count, int dim){
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < word_count){
    for(int j = 0; j < dim; j++){
      matrixNorm_d[i*dim+j] = mat[i*dim+j] / normSum_d[i];
      matrixNorm_T[j*word_count+i] = mat[i*dim+j] / normSum_d[i];
    }
  }
}

__global__
void vectorManipulation(float* A, float* B, float* C, float* D, int len){
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < len)
    D[i] = A[i] + C[i] - B[i];
}

__global__
void fillABC(float* A_d, float* B_d, float* C_d, int* A_idx, int* B_idx, int* C_idx, float* matrixNorm_d, int len, int dim){
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < len){
    memcpy(&A_d[i*dim], &matrixNorm_d[A_idx[i]*dim], sizeof(float)*dim);
    memcpy(&B_d[i*dim], &matrixNorm_d[B_idx[i]*dim], sizeof(float)*dim);
    memcpy(&C_d[i*dim], &matrixNorm_d[C_idx[i]*dim], sizeof(float)*dim);
  }
}

__global__
void correctness(float* resVec_d, int* A_idx_d, int* B_idx_d, int* C_idx_d, int* D_idx_d, int word_count, int query_count){
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < query_count){
    int idx_a = A_idx_d[i];
    int idx_b = B_idx_d[i];
    int idx_c = C_idx_d[i];
    int max_idx = 0;
    int max = -1;
    for(int j = 0; j < word_count; j ++){
      if(j == idx_a || j == idx_b || j == idx_c)
        continue;
      if(resVec_d[i*word_count + j] > max){
        max = resVec_d[i*word_count + j];
        max_idx = j;
      }
    }
    D_idx_d[i] = max_idx;
  }
}

void fill_host_vector(string filename, unordered_map<string, int> &word2vec_map, string *dictionary, float *normSum_h, float *matrix_h, int dim){
  string str;
  ifstream infile;
  infile.open(filename);
  int i = 0;
  while(getline(infile,str)){
    string buf;
    stringstream ss(str);
    ss >> buf;
    word2vec_map[buf] = i;
    dictionary[i] = buf;
    int j = 0;
    while (ss >> buf){
      matrix_h[i*dim+j] = stof(buf);
      normSum_h[i] += pow(stof(buf),2);
      j++;
    }
    normSum_h[i] = sqrt(normSum_h[i]);
    i++;
  }
  infile.close();
}

std::tuple<int, int> find_dim_wordcount(string filename){
  int word_count = 0;
  string str;
  ifstream infile;
  infile.open(filename);
  getline(infile,str);
  word_count++;
  stringstream sss(str);
  int dim = -1;
  string buff;
  while(sss >> buff)
    dim ++;
  while(getline(infile,str)){
    word_count++;
  }
  infile.close();
  return std::make_tuple(dim, word_count);
}

int fill_host_query_vector(string queryFile, unordered_map<string, int> word2vec_map, int *A_idx, int *B_idx, int *C_idx, int *D_idx){
  string str;
  ifstream infile;
  infile.open(queryFile);
  int i = 0;
  int model_miss = 0;
  while(getline(infile, str)){
    string buf;
    stringstream ss(str);
    vector<string> queryWords;
    while (ss >> buf)
      queryWords.push_back(buf);
    int occurence_count;
    for(int j = 0; j < 4; j++){
      occurence_count = word2vec_map.count(queryWords[j]);
      if(occurence_count != 1){
        model_miss += 1;
        continue;
      }
    }
    A_idx[i] = word2vec_map[queryWords[1]];
    B_idx[i] = word2vec_map[queryWords[0]];
    C_idx[i] = word2vec_map[queryWords[2]];
    D_idx[i] = word2vec_map[queryWords[3]];
    i++;
  }
  infile.close();
  cout << "model doesn't cover " << model_miss << " of the total queries" << endl;
  return model_miss;
}

int analogy_single_query(string filename, vector<string> queryWords) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    int dim, word_count;
    tie(dim, word_count) = find_dim_wordcount(filename);
    int matrix_size = word_count*dim;

    unordered_map<string, int> word2vec_map;
    string dictionary[word_count];
    float *normSum_h = new float[word_count];
    float *matrix_h = new float[matrix_size];
    float *resVec_h = new float[word_count];
    fill_host_vector(filename, word2vec_map, dictionary, normSum_h, matrix_h, dim);

    // check if all query words are in model
    int occurence_count;
    for(int i = 0; i < 3; i++){
      occurence_count = word2vec_map.count(queryWords[i]);
      if(occurence_count != 1){
        cout << "model does not contain the word: " << queryWords[i] << endl;
        return -1;
      }
    }

    float* matrix_d;
    float* matrixNorm_d;
    float* matrixNorm_T;
    float* predict_d;
    float* resVec_d;
    float* normSum_d;

    ERROR_CHECK(cudaMalloc((void **)&matrix_d, matrix_size*sizeof(float)));
    ERROR_CHECK(cudaMalloc((void **)&matrixNorm_d, matrix_size*sizeof(float)));
    ERROR_CHECK(cudaMalloc((void **)&matrixNorm_T, matrix_size*sizeof(float)));
    ERROR_CHECK(cudaMalloc((void **)&predict_d, dim*sizeof(float)));
    ERROR_CHECK(cudaMalloc((void **)&resVec_d, word_count*sizeof(float)));
    ERROR_CHECK(cudaMalloc((void **)&normSum_d, word_count*sizeof(float)));

    ERROR_CHECK(cudaMemcpy(matrix_d, matrix_h, matrix_size*sizeof(float), cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemcpy(normSum_d, normSum_h, word_count*sizeof(float), cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemset(predict_d, 0.0, dim*sizeof(float)));

    dim3 dimGrid(ceil(word_count/1024.0), 1, 1);
    dim3 dimBlock(1024, 1, 1);
    normalize_T<<<dimGrid, dimBlock>>>(matrix_d, normSum_d, matrixNorm_d, matrixNorm_T, word_count, dim);
    ERROR_CHECK(cudaDeviceSynchronize());

    int idx_1 = word2vec_map[queryWords[0]];
    int idx_2 = word2vec_map[queryWords[1]];
    int idx_3 = word2vec_map[queryWords[2]];

    const float alpha = 1.0f;
    const float beta = 0.0f;
    dim3 dimGrid1(1, 1, 1);
    dim3 dimBlock1(dim, 1, 1);
    vectorManipulation<<<dimGrid1, dimBlock1>>>(&matrixNorm_d[idx_1*dim],
              &matrixNorm_d[idx_2*dim], &matrixNorm_d[idx_3*dim], predict_d, dim);
    ERROR_CHECK(cudaDeviceSynchronize());

    cublasSgemv(
      handle, CUBLAS_OP_N,
      word_count, dim,
      &alpha, matrixNorm_T, std::max(word_count, dim),
      predict_d, 1, &beta,
      resVec_d, 1
    );

    ERROR_CHECK(cudaMemcpy(resVec_h, resVec_d, word_count*sizeof(float), cudaMemcpyDeviceToHost));

    resVec_h[idx_1] = 0;
    resVec_h[idx_2] = 0;
    resVec_h[idx_3] = 0;
    int max = std::max_element(resVec_h, resVec_h + word_count) - resVec_h;
    cout << dictionary[max] << endl;

    cublasDestroy(handle);
    ERROR_CHECK(cudaFree(matrix_d));
    ERROR_CHECK(cudaFree(matrixNorm_d));
    ERROR_CHECK(cudaFree(matrixNorm_T));
    ERROR_CHECK(cudaFree(predict_d));
    ERROR_CHECK(cudaFree(resVec_d));
    ERROR_CHECK(cudaFree(normSum_d));
    delete [] normSum_h;
    delete [] matrix_h;
    delete [] resVec_h;

    return 0;
}

int analogy_batch_query(string filename, string queryFile) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    int dim, word_count, query_count;
    tie(dim, query_count) = find_dim_wordcount(queryFile);
    tie(dim, word_count) = find_dim_wordcount(filename);
    int matrix_size = word_count*dim;

    unordered_map<string, int> word2vec_map;
    string dictionary[word_count];
    float *normSum_h = new float[word_count];
    float *matrix_h = new float[matrix_size];
    int *A_idx = new int[query_count];
    int *B_idx = new int[query_count];
    int *C_idx = new int[query_count];
    int *D_idx = new int[query_count];
    int *D_res = new int[query_count];
    fill_host_vector(filename, word2vec_map, dictionary, normSum_h, matrix_h, dim);
    query_count -= fill_host_query_vector(queryFile, word2vec_map, A_idx, B_idx, C_idx, D_idx);
    int query_matrix_size = query_count*dim;

    float* matrix_d;
    float* matrixNorm_d;
    float* matrixNorm_T;
    float* predict_d;
    float* resVec_d;
    float* normSum_d;
    int* A_idx_d;
    int* B_idx_d;
    int* C_idx_d;
    int* D_idx_d;
    float* A_d;
    float* B_d;
    float* C_d;

    ERROR_CHECK(cudaMalloc((void **)&matrix_d, matrix_size*sizeof(float)));
    ERROR_CHECK(cudaMalloc((void **)&matrixNorm_d, matrix_size*sizeof(float)));
    ERROR_CHECK(cudaMalloc((void **)&matrixNorm_T, matrix_size*sizeof(float)));
    ERROR_CHECK(cudaMalloc((void **)&predict_d, query_matrix_size*sizeof(float)));
    ERROR_CHECK(cudaMalloc((void **)&resVec_d, word_count*query_count*sizeof(float)));
    ERROR_CHECK(cudaMalloc((void **)&normSum_d, word_count*sizeof(float)));
    ERROR_CHECK(cudaMalloc((void **)&A_idx_d, query_count*sizeof(int)));
    ERROR_CHECK(cudaMalloc((void **)&B_idx_d, query_count*sizeof(int)));
    ERROR_CHECK(cudaMalloc((void **)&C_idx_d, query_count*sizeof(int)));
    ERROR_CHECK(cudaMalloc((void **)&D_idx_d, query_count*sizeof(int)));
    ERROR_CHECK(cudaMalloc((void **)&A_d, query_matrix_size*sizeof(float)));
    ERROR_CHECK(cudaMalloc((void **)&B_d, query_matrix_size*sizeof(float)));
    ERROR_CHECK(cudaMalloc((void **)&C_d, query_matrix_size*sizeof(float)));

    ERROR_CHECK(cudaMemcpy(matrix_d, matrix_h, matrix_size*sizeof(float), cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemcpy(normSum_d, normSum_h, word_count*sizeof(float), cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemcpy(A_idx_d, A_idx, query_count*sizeof(int), cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemcpy(B_idx_d, B_idx, query_count*sizeof(int), cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemcpy(C_idx_d, C_idx, query_count*sizeof(int), cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemset(predict_d, 0.0, query_matrix_size*sizeof(float)));

    dim3 dimGrid(ceil(word_count/1024.0), 1, 1);
    dim3 dimBlock(1024, 1, 1);
    normalize_T<<<dimGrid, dimBlock>>>(matrix_d, normSum_d, matrixNorm_d, matrixNorm_T, word_count, dim);
    ERROR_CHECK(cudaDeviceSynchronize());

    // float* matrixNorm_h = new float[matrix_size];
    // ERROR_CHECK(cudaMemcpy(matrixNorm_h, matrixNorm_d, matrix_size*sizeof(float), cudaMemcpyDeviceToHost));
    // int idxA = word2vec_map["athens"];
    // int idxB = word2vec_map["greece"];
    // int idxC = word2vec_map["baghdad"];
    // for(int i = 0; i < 10; i++){
    //   cout << matrixNorm_h[idxA * dim + i] << endl;;
    // }
    // for(int i = 0; i < 10; i++){
    //   cout << matrixNorm_h[idxB * dim + i] << endl;;
    // }
    // for(int i = 0; i < 10; i++){
    //   cout << matrixNorm_h[idxC * dim + i] << endl;;
    // }

    // in case fillABC does not work
    for(int i = 0; i < query_count; i++){
      ERROR_CHECK(cudaMemcpy(&A_d[i*dim], &matrixNorm_d[A_idx[i]*dim], dim*sizeof(float), cudaMemcpyDeviceToDevice));
      ERROR_CHECK(cudaMemcpy(&B_d[i*dim], &matrixNorm_d[B_idx[i]*dim], dim*sizeof(float), cudaMemcpyDeviceToDevice));
      ERROR_CHECK(cudaMemcpy(&C_d[i*dim], &matrixNorm_d[C_idx[i]*dim], dim*sizeof(float), cudaMemcpyDeviceToDevice));
    }

    // dim3 dimGrid1(ceil(query_count/1024.0), 1, 1);
    // dim3 dimBlock1(1024, 1, 1);
    // fillABC<<<dimGrid1, dimBlock1>>>(A_d, B_d, C_d, A_idx_d, B_idx_d, C_idx_d, matrixNorm_d, query_count, dim);
    // ERROR_CHECK(cudaDeviceSynchronize());

    // ERROR_CHECK(cudaMemcpy(A_h, A_d, query_matrix_size*sizeof(float), cudaMemcpyDeviceToHost));
    // ERROR_CHECK(cudaMemcpy(B_h, B_d, query_matrix_size*sizeof(float), cudaMemcpyDeviceToHost));
    // ERROR_CHECK(cudaMemcpy(C_h, C_d, query_matrix_size*sizeof(float), cudaMemcpyDeviceToHost));
    // for(int i = 0; i < 10; i++){
    //   cout << A_h[i] << endl;;
    // }
    // for(int i = 0; i < 10; i++){
    //   cout << B_h[i] << endl;;
    // }
    // for(int i = 0; i < 10; i++){
    //   cout << C_h[i] << endl;;
    // }

    const float alpha = 1.0f;
    const float negative_alpha = -1.0f;
    const float beta = 0.0f;
    cublasSaxpy(
      handle, query_matrix_size,
      &alpha,
      A_d, 1,
      predict_d, 1
    );
    ERROR_CHECK(cudaDeviceSynchronize());
    cublasSaxpy(
      handle, query_matrix_size,
      &negative_alpha,
      B_d, 1,
      predict_d, 1
    );
    ERROR_CHECK(cudaDeviceSynchronize());
    cublasSaxpy(
      handle, query_matrix_size,
      &alpha,
      C_d, 1,
      predict_d, 1
    );
    ERROR_CHECK(cudaDeviceSynchronize());

    // float* predict_h = new float[query_matrix_size];
    // ERROR_CHECK(cudaMemcpy(predict_h, predict_d, query_matrix_size*sizeof(float), cudaMemcpyDeviceToHost));
    // for(int i = 0; i < dim; i++){
    //   cout << predict_h[i] << endl;
    // }

    cublasSgemm(
      handle, CUBLAS_OP_N, CUBLAS_OP_N,
      word_count, query_count, dim,
      &alpha,
      matrixNorm_T, std::max(word_count, dim),
      predict_d, std::max(query_count, dim),
      &beta,
      resVec_d, std::max(word_count, query_count)
    );
    ERROR_CHECK(cudaDeviceSynchronize());

    float* resVec_h = new float[query_count*word_count];
    ERROR_CHECK(cudaMemcpy(resVec_h, resVec_d, query_count*word_count*sizeof(float), cudaMemcpyDeviceToHost));
    // float* matrixNorm_h = new float[matrix_size];
    // ERROR_CHECK(cudaMemcpy(matrixNorm_h, matrixNorm_d, matrix_size*sizeof(float), cudaMemcpyDeviceToHost));
    // for(int i = 0; i < dim; i++){
    //   cout << matrixNorm_h[dim+i] << endl;
    // }

    // cout << resVec_h[1] << endl;
    // cout << resVec_h[query_count] << endl;
    // int idx_a = A_idx[0];
    // int idx_b = B_idx[0];
    // int idx_c = C_idx[0];
    // int max_idx = 0;
    // int max = -1;
    // cout << dictionary[idx_a] << endl;
    // cout << dictionary[idx_b] << endl;
    // cout << dictionary[idx_c] << endl;
    // cout << resVec_h[word2vec_map["germany"]] << endl;
    // cout << dictionary[word2vec_map["germany"]] << endl;
    // cout << resVec_h[word2vec_map["zetan"]] << endl;
    // for(int j = 0; j < word_count; j ++){
    //   if(j == idx_a || j == idx_b || j == idx_c)
    //     continue;
    //   cout << max << endl;
    //   if(resVec_h[j] > max){
    //     // cout << dictionary[j] << " " << resVec_h[j] << endl;
    //     max = resVec_h[j];
    //     max_idx = j;
    //   }
    // }
    // cout << max_idx << endl;
    // cout << dictionary[max_idx] << endl;

    // dim3 dimGrid2(ceil(query_count/1024.0), 1, 1);
    // dim3 dimBlock2(1024, 1, 1);
    // correctness<<<dimGrid2, dimBlock2>>>(resVec_d, A_idx_d, B_idx_d, C_idx_d, D_idx_d, word_count, query_count);
    // ERROR_CHECK(cudaDeviceSynchronize());
    // ERROR_CHECK(cudaMemcpy(D_res, D_idx_d, query_count*sizeof(float), cudaMemcpyDeviceToHost));

    int idx_a = A_idx[2];
    int idx_b = B_idx[2];
    int idx_c = C_idx[2];
    cout << dictionary[idx_a] << endl;
    cout << dictionary[idx_b] << endl;
    cout << dictionary[idx_c] << endl;
    float* resVec_tmp = &resVec_h[2*word_count];
    resVec_tmp[idx_a] = 0;
    resVec_tmp[idx_b] = 0;
    resVec_tmp[idx_c] = 0;
    int max = std::max_element(resVec_tmp, resVec_tmp + word_count) - resVec_tmp;
    cout << dictionary[max] << endl;

    // int total_correct = 0;
    // for(int i = 0; i < query_count; i++){
    //   int idx_a = A_idx[i];
    //   int idx_b = B_idx[i];
    //   int idx_c = C_idx[i];
    //   float* resVec_tmp = &resVec_h[i*word_count];
    //   resVec_tmp[idx_a] = 0;
    //   resVec_tmp[idx_b] = 0;
    //   resVec_tmp[idx_c] = 0;
    //   int max = std::max_element(resVec_tmp, resVec_tmp + word_count) - resVec_tmp;
    //   cout << dictionary[max] << endl;
    //
    //   if(max == D_idx[i]){
    //     total_correct ++;
    //   }
    // }
    // cout << "total_correct is: " << total_correct << endl;
    // cout << "Correctness is: " << (total_correct*1.0)/query_count << endl;

    cublasDestroy(handle);
    ERROR_CHECK(cudaFree(matrix_d));
    ERROR_CHECK(cudaFree(matrixNorm_d));
    ERROR_CHECK(cudaFree(matrixNorm_T));
    ERROR_CHECK(cudaFree(predict_d));
    ERROR_CHECK(cudaFree(resVec_d));
    ERROR_CHECK(cudaFree(normSum_d));
    delete [] normSum_h;
    delete [] matrix_h;
    delete [] resVec_h;
    delete [] A_idx;
    delete [] B_idx;
    delete [] C_idx;
    delete [] D_idx;
    delete [] D_res;

    return 0;
}
