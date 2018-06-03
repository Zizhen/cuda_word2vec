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
    double* word_vector;
    unordered_map<string, double*> word2vec_map;
    string str;
    ifstream infile;
    infile.open(argv[2]);
    int word_count = 0;
    int dimension = stod(argv[4]);

    while(getline(infile,str)){
      word_count++;
    }

    cout << word_count << " " << dimension << endl;
    // while(getline(infile,str)){
    //   word_vector.clear();
    //   string buf;
    //   stringstream ss(str);
    //   vector<string> tokens;
    //   ss >> buf;
    //   word_list.push_back(buf);
    //   while (ss >> buf){
    //     word_matrix_1d.push_back(stod(buf));
    //     word_vector.push_back(stod(buf));
    //   }
    //   word2vec_map[word_list[i]] = word_vector;
    // }
    infile.close();



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
