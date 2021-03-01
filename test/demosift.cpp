#include "../include/cphdr.h"

int main(int argv, char** args)
{	
  std::string kpFile;
  std::vector<KeyPoints> kp;
  
  kpFile = args[1];
  
  kp = loadKeypoints(kpFile);
  
  std::cout << "Quantidade de KeyPoints lidos:" << kp.size() << "\n";  
}
