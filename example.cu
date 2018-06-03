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

double** vec2arr(vector<vector<double>> vals, int N, int M){
   double** temp;
   temp = new double*[N];
   for(unsigned i=0; i < N; i++)
   {
      temp[i] = new double[M];
      for(unsigned j=0; j < M; j++)
      {
          temp[i][j] = vals[i][j];
      }
   }
   return temp;
}

int main(int argc, char* argv[]) {
  if(argc == 2){
    if(strcmp(argv[1], "analogy") == 0)
      cout << "usage: ./a.out analogy path/to/your/model path/to/your/testfile" << endl;
    else
      cout << "function not supported" << endl;
  }
  else if (argc > 2){
    vector<string> word_list;
    vector<double> word_vector;
    vector<double> word_matrix_1d;
    unordered_map<string, vector<double>> word2vec_map;
    string str;
    ifstream infile;
    infile.open(argv[2]);
    int i = 0;
    while(getline(infile,str)){
      word_vector.clear();
      string buf;
      stringstream ss(str);
      vector<string> tokens;
      ss >> buf;
      word_list.push_back(buf);
      while (ss >> buf){
        word_matrix_1d.push_back(stod(buf));
        word_vector.push_back(stod(buf));
      }
      word2vec_map[word_list[i]] = word_vector;
      i++;
    }
    int word_count = word_list.size();
    int dimension = word_vector.size();
    infile.close();
    double* word_matrix_1d_arr = &word_matrix_1d[0];

    for(int i = 0; i < 100; i++){
      cout << word_matrix_1d[i];
    }


    // if(strcmp(argv[1],"analogy") == 0){
    //   if(argc == 6){
    //     string w1 = argv[3];
    //     string w2 = argv[4];
    //     string w3 = argv[5];
    //     vector<nn_entry> NNs = analogy(w1, w2, w3, word_matrix, word2vec_map);
    //     for(auto i: NNs){
    //       cout << std::get<0>(i) << " " << std::get<1>(i) << endl;
    //     }
    //   }
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
