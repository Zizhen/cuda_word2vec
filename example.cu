#include <iostream>
#include <string.h>
#include <fstream>
#include <vector>
#include <sstream>
#include <unordered_map>

using namespace std;

// Compile with:
// nvcc -o example example.cu
#define N 10

__global__
void add(int *a, int *b) {
  int i = blockIdx.x;
  if (i<N) {
    b[i] = 2*a[i];
  }
}

int main(int argc, char* argv[]) {
  if(argc == 2){
    if(strcmp(argv[1], "analogy") == 0)
      cout << "usage: ./a.out analogy path/to/your/model path/to/your/testfile dimension" << endl;
    else
      cout << "function not supported" << endl;
  }
  else if (argc > 2){
    unordered_map<string, int> word2vec_map;
    string str;
    ifstream infile;
    infile.open(argv[2]);
    int word_count = 0;
    int dimension = stod(argv[4]);
    cout << dimension << endl;
    while(getline(infile,str)){
      word_count++;
    }
    int matrix_size = word_count*dimension/2;
    cout << matrix_size << endl;
    double word_matrix[matrix_size];
    cudaMalloc((void **)&word_matrix, matrix_size*sizeof(double));
    cout << matrix_size << endl;

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
        word_matrix[i*dimension+j] = stod(buf);
        j++;
      }
      i++;
    }
    infile.close();

    // for (int i = 0; i < 10; i++){
    //   for (int j = 0; j < 10; j++){
    //     cout << word_matrix[i*dimension+j] << " ";
    //   }
    //   cout << endl;
    // }


  }
  ////////////////////////////////
    int ha[N], hb[N];
    int *da, *db;
    cudaMalloc((void **)&da, N*sizeof(int));
    cudaMalloc((void **)&db, N*sizeof(int));

    for (int i = 0; i<N; ++i) {
        ha[i] = i;
    }

    cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice);

    add<<<N, 1>>>(da, db);

    cudaMemcpy(hb, db, N*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i<N; ++i) {
      cout << hb[i] << endl;
    }

    cudaFree(da);
    cudaFree(db);

    return 0;
}
