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
void normalize(float* mat, float* normSum_d, float* matrixNorm_d, int dim, int max){
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < max){
    for(int j = 0; j < dim; j++){
      matrixNorm_d[i*dim+j] = mat[i*dim+j] / normSum_d[i];
    }
  }
}

__global__
void vectorManipulation(float* A, float* B, float* C, float* D, int len){
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  printf("test\n");
  if (i < len)
    D[i] = A[i] + C[i] - B[i];
}

__global__
void vecMatMultiplication(float* mat, float* vec, float* res, int len, int max){
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < max){
    for(int j = 0; j < len; j ++){
      res[i] += mat[i*len+j] * vec[j];
    }
  }
}

int main(int argc, char* argv[]) {
  if(argc == 2){
    if(strcmp(argv[1], "analogy") == 0){
      cout << "usage: ./a.out analogy path/to/your/model dim path/to/your/testfile" << endl;
      cout << "   or: ./a.out analogy path/to/your/model dim word1 word2 word3" << endl;
    }
    else
      cout << "function not supported" << endl;
  }
  else if (argc > 2){
    unordered_map<string, int> word2vec_map;
    string str;
    ifstream infile;
    infile.open(argv[2]);
    int word_count = 0;
    int dim = stod(argv[3]);
    while(getline(infile,str)){
      word_count++;
    }
    infile.close();
    string dictionary[word_count];
    float normSum_h[word_count];

    infile.open(argv[2]);
    int matrix_size = word_count*dim;
    float *matrix_h = new float[matrix_size];
    float *resVec_h = new float[word_count];
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

    float* matrix_d;
    float* matrixNorm_d;
    float* D;
    float* resVec_d;
    cudaMalloc((void **)&matrix_d, matrix_size*sizeof(float));
    cudaMalloc((void **)&matrixNorm_d, matrix_size*sizeof(float));
    cudaMalloc((void **)&D, dim*sizeof(float));
    cudaMalloc((void **)&resVec_d, word_count*sizeof(float));
    cudaMemcpy(matrix_d, matrix_h, matrix_size*sizeof(float), cudaMemcpyHostToDevice);
    float* normSum_d;
    cudaMalloc((void **)&normSum_d, word_count*sizeof(float));
    cudaMemcpy(normSum_d, normSum_h, word_count*sizeof(float), cudaMemcpyHostToDevice);
    dim3 dimGrid(ceil(word_count/1024.0), 1, 1);
    dim3 dimBlock(1024, 1, 1);
    normalize<<<dimGrid, dimBlock>>>(matrix_d, normSum_d, matrixNorm_d, dim, word_count);

    // float *matRes = new float[matrix_size];
    // cudaMemcpy(matRes, matrixNorm_d, matrix_size*sizeof(float), cudaMemcpyDeviceToHost);
    // for(int j = 0; j < 150; j++){
    //   cout << matRes[word2vec_map["king"]*150+j] << endl;
    // }
    // cout << endl;
    // for(int j = 0; j < 150; j++){
    //   cout << matRes[word2vec_map["man"]*150+j] << endl;
    // }
    // cout << endl;
    // for(int j = 0; j < 150; j++){
    //   cout << matRes[word2vec_map["woman"]*150+j] << endl;
    // }

    if(strcmp(argv[1],"analogy") == 0){
      if(argc == 7){
        int count[3];
        for(int i = 0; i < 3; i++){
          count[i] = word2vec_map.count(argv[4+i]);
          if(count[i] != 1){
              cout << "map does not contain the word: " << argv[4+i] << endl;
              return -1;
          }
        }
        int idx_1 = word2vec_map[argv[4]];
        int idx_2 = word2vec_map[argv[5]];
        int idx_3 = word2vec_map[argv[6]];
        dim3 dimGrid1(1, 1, 1);
        dim3 dimBlock1(dim, 1, 1);
        vectorManipulation<<<dimGrid1, dimBlock1>>>(&matrixNorm_d[idx_1*dim],
                  &matrixNorm_d[idx_2*dim], &matrixNorm_d[idx_3*dim], D, dim);
        cudaDeviceSynchronize();
        // float *matRes = new float[dim];
        // cudaMemcpy(matRes, D, dim*sizeof(float), cudaMemcpyDeviceToHost);
        // for(int i = 0; i < 150; i ++){
        //   cout << matRes[i] << endl;
        // }
        dim3 dimGrid2(ceil(word_count/1024.0), 1, 1);
        dim3 dimBlock2(1024, 1, 1);
        vecMatMultiplication<<<dimGrid2, dimBlock2>>>(matrixNorm_d, D, resVec_d, dim, matrix_size);
        cudaMemcpy(resVec_h, resVec_d, word_count*sizeof(float), cudaMemcpyDeviceToHost);
        resVec_h[idx_1] = 0;
        resVec_h[idx_2] = 0;
        resVec_h[idx_3] = 0;
        int max = std::max_element(resVec_h, resVec_h + word_count) - resVec_h;
        cout << dictionary[max] << endl;
      }
    }

    cudaFree(matrix_d);
    cudaFree(matrixNorm_d);
    cudaFree(D);
    cudaFree(resVec_d);
  }

  return 0;
}
