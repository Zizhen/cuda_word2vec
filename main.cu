#include <iostream>
#include "analogy.cu"

using namespace std;

int main(int argc, char* argv[]) {
  if(argc <= 2){
    cout << "usage: ./a.out analogy path/to/your/model path/to/your/testfile" << endl;
    cout << "   or: ./a.out analogy path/to/your/model word1 word2 word3" << endl;
  }
  else if(argc == 6 && strcmp(argv[1], "analogy") == 0){
    vector<string> queryWords;
    const char *vinit[] = {argv[3], argv[4], argv[5]};
    vector<string> words(vinit, end(vinit));
    analogy_single_query(argv[2], words);
  }
  else if(argc == 4 && strcmp(argv[1], "analogy") == 0){
    analogy_batch_query(argv[2], argv[3]);
  }

}
