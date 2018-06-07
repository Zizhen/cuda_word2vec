#include <iostream>
#include <string.h>
#include <fstream>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <cuda.h>
#include <cuda_runtime_api.h>

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


int main(int argc, char* argv[]) {
  if(argc == 2){
    if(strcmp(argv[1], "analogy") == 0)
      cout << "usage: ./a.out analogy path/to/your/model dimension path/to/your/testfile" << endl;
    else
      cout << "function not supported" << endl;
  }
  else if (argc > 2){
    // size_t f, t;
    // cudaSetDevice(0);
    // cudaMemGetInfo(&f, &t);
    // cout << f << " " << t << endl;

    unordered_map<string, int> word2vec_map;
    string str;
    ifstream infile;
    infile.open(argv[2]);
    int word_count = 0;
    int dimension = stod(argv[4]);
    while(getline(infile,str)){
      word_count++;
    }
    int matrix_size = word_count*dimension;
    float *word_matrix_h = new float[matrix_size];
    int i = 0;
    while(getline(infile,str)){
      string buf;
      stringstream ss(str);
      ss >> buf;
      cout << buf << endl;
      word2vec_map[buf] = i;
      int j = 0;
      while (ss >> buf){
        cout << buf << endl;
        word_matrix_h[i*dimension+j] = stod(buf);
        j++;
      }
      i++;
    }
    infile.close();
    float* word_matrix_d;
    ERROR_CHECK(cudaMalloc((void **)&word_matrix_d, matrix_size*sizeof(float)));
    cudaMemcpy(word_matrix_d, word_matrix_h, matrix_size*sizeof(float), cudaMemcpyHostToDevice);

    if(strcmp(argv[1],"analogy") == 0){
      cout << "test" << endl;
      if(argc == 6){
        int idx_1 = word2vec_map[argv[4]];
        int idx_2 = word2vec_map[argv[5]];
        int idx_3 = word2vec_map[argv[6]];
        cout << idx_1 << " " << idx_2 << " " << idx_3 << endl;
      }
    }

    cudaFree(word_matrix_d);
  }

  return 0;
}
