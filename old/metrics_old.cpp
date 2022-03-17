#include "../include/evaluation/metrics.h"

//g++ -std=c++11 -ggdb `pkg-config --cflags opencv` -o `basename repeatability.cpp .cpp` repeatability.cpp `pkg-config --libs opencv`

//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/imgproc.hpp"
//Incluindo todas as bibliotecas std do c/c++
//#include <bits/stdc++.h>

const int INF = (int) 1e9;

struct Line{
  std::pair<int, int> p1, p2;
};

std::vector<struct Line> linha;

FILE *inKP;
//cv::Matriz de Homografia, vetor base, vetor resultado
cv::Mat homografia, pointBase, resultPoint;

std::vector<std::pair<float, std::pair<int, int> > > keyPointB, keyPointB2;
std::vector<std::pair<float, std::pair<int, int> > > keyPointS;

double NR[] = {0, 0, 0}; // Socv::Matorio do Numero de Repeticoes em para cada uma das 3 regioes de interesse
double NU[] = {0, 0, 0}; // Socv::Matorio do Numero total util para cada uma das 3 regioes de interesse

const int MAXKP = 600;

int quantMaxKP = MAXKP;
int quantPositiveKP = 0; //quantidade de KeyPoints que é encontrado em ambas imagens 




void readMatriz(std::string argv){
  // Abrindo arquivos
  std::ifstream  in(argv);
    std::streambuf  *cinbuf = std::cin.rdbuf();
    std::cin.rdbuf(in.rdbuf());
    
  float value;
  homografia = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 3; j++){
      std::cin>>value;
      homografia.at<float>(i, j) = value;
    }
  }
  in.close();
}

//Lendo keyPoints do arquivo de texto 
//Se val = 1 salva os pontos base, se for igual a 2 salva os pontos de saida
void readKeyPoints(int val){
  int x, y, n;
  float response; //Por enquanto n serve pra nada
  fscanf(inKP, "%d", &n);

  for(int i = 0; i < n; i++){
    fscanf(inKP, "%d %d %f", &y, &x, &response);
    if(val == 1)
      keyPointB.push_back(std::make_pair(-response, std::make_pair(x, y)));
    else if(val == 2)
      keyPointS.push_back(std::make_pair(-response, std::make_pair(x, y)));
  }
}

//Função que faz a transformação dos keypoints da imagem base
void transformacao(){
  int siz = (int)keyPointB.size();
  keyPointB2 = keyPointB;
  
  for(int i = 0; i < siz && i < quantMaxKP; i++){
    pointBase.at<float>(0, 0) = keyPointB[i].second.first;
    pointBase.at<float>(1, 0) = keyPointB[i].second.second;
    pointBase.at<float>(2, 0) = 1;
    
    //Calculando multiplicação entre cv::Matrizes
    resultPoint = homografia * pointBase;
    
    //Pegando as novas coordenadas do pontoBase
    float aux1 = resultPoint.at<float>(0, 0) / resultPoint.at<float>(2, 0);
    float aux2 = resultPoint.at<float>(1, 0) / resultPoint.at<float>(2, 0);
    
    std::pair<int, int> point;
    
    point.first = round(aux1); //x
    point.second = round(aux2); //y
    
    keyPointB[i].second = point;
  }
}

//Função para calcular o Repeatability Rate retornando a quantidade de keypoins que deu Positivo (encontrado na outra imagem)
void calculandoRR(int quantK){
  int visited[MAXKP];
  memset(visited, 0, sizeof visited);
  
  for(int i = 0; i < quantK; i++){
    int x = keyPointB[i].second.first;
    int y = keyPointB[i].second.second;
    for(int j = 0; j < quantK; j++){
      if(visited[j]) continue;
      
      int x1 = keyPointS[j].second.first;
      int y1 = keyPointS[j].second.second;
      // Calculando a distância em pixel entre os dois pontos
      int dist = std::max(abs(x - x1) , abs(y - y1)); 
      
      //Se estiver dentro do raio de 19 pixels
      if(dist < 19){ 
        //visited[j] = 1;
        struct Line linhaAux;
        linhaAux.p1 = std::make_pair(keyPointB2[i].second.first, keyPointB2[i].second.second);
        linhaAux.p2 = std::make_pair(x1, y1);
        
        linha.push_back(linhaAux);
        
        quantPositiveKP++;
        break;
      }
    }
  }	
}

void showPointsCorrelation(char *img1, char *img2)
{
  cv::Mat im1, im2; //imagens com keypoints correlatos
  std::string aux1 = (std::string) img1;
  std::string aux2 = (std::string) img2;
  aux1 = "../dataset/2D/distance/100/100.gLarson97R.jpg";
  aux2 = "../dataset/2D/distance/103/103.gLarson97R.jpg";

  im1 = cv::imread(aux1, cv::IMREAD_UNCHANGED);
  im2 = cv::imread(aux2, cv::IMREAD_UNCHANGED);

  cv::Size sz1 = im1.size();
  cv::Size sz2 = im2.size();

  //im3 = Junção das duas imagens lado a lado
  cv::Mat im3(sz1.height, sz1.width+sz2.width, CV_8UC3);
  cv::Mat left(im3, cv::Rect(0, 0, sz1.width, sz1.height));    
  im1.copyTo(left);
  cv::Mat right(im3, cv::Rect(sz1.width, 0, sz2.width, sz2.height));
  im2.copyTo(right);
    
  //Desenhando linhas entre keypoints
  for(int i = 0; i < (int)linha.size(); i++)
  {
    int x = linha[i].p1.first;
    int y = linha[i].p1.second;
    int x1 = linha[i].p2.first + im1.cols;
    int y1 = linha[i].p2.second;
    
    line(im3, cv::Point (x, y), cv::Point (x1 , y1), cv::Scalar(0, 0, 255), 1, 8, 0);
  }

  //resize(im3, im3, Size(), 0.2, 0.2, CV_INTER_LINEAR);
  imwrite("im.jpg", im3);
}

int lookFor(std::string x, std::string str){
  //cout<<str<<"|"<<x<<endl;
  for(int i = 0; i < (int) str.size(); i++){
    if(str[i] == x[0]){
      if(str[i+1] == x[1])
        return i;
    }
  }
  return -1;
}

void execute(char** argv, std::string base, std::string saida){
  
  std::string roiN[] = {"1", "2", "3"};
  
  for(int k = 0; k < 3; k++){
    quantPositiveKP = 0;
    keyPointB.clear();
    keyPointB2.clear();
    keyPointS.clear();
    
    std::string txt0(argv[1]), txt1(argv[2]), txt2(argv[3]);
      
    //processando std::string base e saida
    int posi;
    //para o arquivo H
    posi = lookFor("BX", txt0);
    txt0.replace(posi, 2, base);
    posi = lookFor("SX", txt0);
    txt0.replace(posi, 2, saida);
    
    //para os arquivos com os KP
    posi = lookFor("BX", txt1);
    txt1.replace(posi, 2, base);
    posi = lookFor("BX", txt1);
    txt1.replace(posi, 2, base);
    
    txt1.replace(txt1.size()-5, 1, roiN[k]);
    
    posi = lookFor("SX", txt2);
    txt2.replace(posi, 2, saida);
    posi = lookFor("SX", txt2);
    txt2.replace(posi, 2, saida);
    
    txt2.replace(txt2.size()-5, 1, roiN[k]);
    
    readMatriz(txt0);
    
    inKP = fopen(txt1.c_str(), "r");
    readKeyPoints(1);
    
    fclose(inKP);
    
    inKP = fopen(txt2.c_str(), "r");
    readKeyPoints(2);
    
    fclose(inKP);
    
    //Inicializando cv::Matriz coordenada
    pointBase = cv::Mat::zeros(cv::Size(1, 3), CV_32F);
    resultPoint = cv::Mat::zeros(cv::Size(1, 3), CV_32F);	
    
    //Ordenando pelo maior response para selecionar os x maiores para comparação
    sort(keyPointB.begin(), keyPointB.end());
    sort(keyPointB2.begin(), keyPointB2.end());
    sort(keyPointS.begin(), keyPointS.end());
    
    //Fazendo a transformação para todos os pontos da base
    transformacao();
    
    //Setando a quantidade max de keypoints para ser avaliado
    int quantK = std::min(keyPointB.size(), keyPointS.size()) < quantMaxKP ? std::min(keyPointB.size(), keyPointS.size()) : quantMaxKP;
    
    if(quantK != 0)
      calculandoRR(quantK);	
    
    NR[k]+= (double)quantPositiveKP;
    NU[k]+= (double)quantK;
    
    
  }
}

float sumDist = 0;

void executeUniformity(char** argv, std::string base){	
  
  std::string txt0(argv[1]), txt1(argv[2]), txt2(argv[3]);
    
  //processando std::string base e saida
  int posi;
  
  //para os arquivos com os KP
  posi = lookFor("BX", txt1);
  txt1.replace(posi, 2, base);
  posi = lookFor("BX", txt1);
  txt1.replace(posi, 2, base);
  
  inKP = fopen(txt1.c_str(), "r");
  
  float val;
  fscanf(inKP, "%f", &val);	
  if(std::isnan(val)) val = 0.0;
  
  sumDist += val;
}

//Função Principal
//Chamada: 

void distanceUniformity(char ** argv){
  std::string distance[] = {"100", "103", "109", "122", "147", "197", "297"};
  int siz = 7;
  for(int i = 0; i < siz; i++){
    executeUniformity(argv, distance[i]);
  }
  
  double medDist = sumDist/siz;
  
  printf("%.2f\n", medDist);	
}
void lightUniformity(char **argv){
  std::string light[] = {"001", "010", "011", "100", "101", "110", "111"};
  int siz = 7;
  for(int i = 0; i < siz; i++){
    executeUniformity(argv, light[i]);
  }
  
  double medDist = sumDist/siz;
  
  printf("%.2f\n", medDist);	
}
void viewpointUniformity(char **argv){
  std::string view[] = {"00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"};
  int siz = 21;
  for(int i = 0; i < siz; i++){
    executeUniformity(argv, view[i]);
  }
  
  double medDist = sumDist/siz;
  
  printf("%.2f\n", medDist);	
}
void distanceRR(char **argv){
  std::string distance[] = {"100", "103", "109", "122", "147", "197", "297"};
  int cont = 0;
  int siz = 7;
  for(int i = 0; i < siz; i++){
    for(int j = 0; j < siz; j++){
      if(i != j){				
        execute(argv, distance[i], distance[j]);
      }
    }
  }
  
  if(NU[0] == 0) NU[0]=1;
  if(NU[1] == 0) NU[1]=1;
  if(NU[2] == 0) NU[2]=1;

  double RR1 = NR[0]/NU[0];
  double RR2 = NR[1]/NU[1];
  double RR3 = NR[2]/NU[2];
  
  double RR = std::min(RR1, std::min(RR2, RR3));
  //double RR = (RR1 + RR2 + RR3)/3;
  //printf("RR1  %.8f RR2  %.8f RR2  %.8f\n", RR1, RR2, RR3);
  
  printf("RR %.8f\n", RR);
}
void lightRR(char **argv){
  std::string ligh[] = {"001", "010", "011", "100", "101", "110", "111"};
  int cont = 0;
  int siz = 7;
  for(int i = 0; i < siz; i++){
    for(int j = 0; j < siz; j++){
      if(i != j){				
        execute(argv, ligh[i], ligh[j]);
      }
    }
  }
  if(NU[0] == 0) NU[0]=1;
  if(NU[1] == 0) NU[1]=1;
  if(NU[2] == 0) NU[2]=1;
  
  double RR1 = NR[0]/NU[0];
  double RR2 = NR[1]/NU[1];
  double RR3 = NR[2]/NU[2];
  
  double RR = std::min(RR1, std::min(RR2, RR3));
  
  //printf("RR1  %.8f RR2  %.8f RR2  %.8f\n", RR1, RR2, RR3);
  
  printf("RR %.8f\n", RR);
}
void viewpointRR(char **argv){
  std::string view[] = {"00", "01", "02", "03", "04", "05", "06", "07", "08", "09",
          "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"};
          
  int siz = 21;
  for(int i = 0; i < siz; i++){
    for(int j = 0; j < siz; j++){
      if(i != j){				
        execute(argv, view[i], view[j]);
      }
    }
  }
  if(NU[0] == 0) NU[0]=1;
  if(NU[1] == 0) NU[1]=1;
  if(NU[2] == 0) NU[2]=1;
  
  double RR1 = NR[0]/NU[0];
  double RR2 = NR[1]/NU[1];
  double RR3 = NR[2]/NU[2];
  
  double RR = std::min(RR1, std::min(RR2, RR3));
  
  //printf("RR1  %.8f RR2  %.8f RR2  %.8f\n", RR1, RR2, RR3);
  
  printf("RR %.8f\n", RR);
}

void projectRoomRR(char **argv){
  std::string projectRoom[] = {"1", "2", "3", "4", "5", "6", "7"};
  int cont = 0;
  int siz = 7;

  for(int i = 0; i < siz; i++){
    for(int j = 0; j < siz; j++){
      if(i != j){
        execute(argv, projectRoom[i], projectRoom[j]);
      }
    }
  }
  
  if(NU[0] == 0) NU[0]=1;
  //if(NU[1] == 0) NU[1]=1;
  if(NU[2] == 0) NU[2]=1;

  double RR1 = NR[0]/NU[0];
  //double RR2 = NR[1]/NU[1];
  double RR3 = NR[2]/NU[2];
  
  double RR = std::min(RR1, RR3);
  //double RR = (RR1 + RR2 + RR3)/3;
  //printf("RR1  %.8f RR2  %.8f RR2  %.8f\n", RR1, RR2, RR3);
  
  printf("RR %.8f\n", RR);
}

void projectRoomUniformity(char ** argv){
  std::string projectRoom[] = {"1", "2", "3", "4", "5", "6", "7"};
  int siz = 7;
  for(int i = 0; i < siz; i++){
    executeUniformity(argv, projectRoom[i]);
  }
  
  double medDist = sumDist/siz;
  
  printf("%.2f\n", medDist);	
}

int main(int, char** argv ){
  
  std::string tipo(argv[4]);
    
  if(tipo == "1") distanceUniformity(argv);
  else if(tipo == "2") lightUniformity(argv);
  else if(tipo == "3") viewpointUniformity(argv);
  else if(tipo == "4") distanceRR(argv);
  else if(tipo == "5") lightRR(argv);
  else if(tipo == "6") viewpointRR(argv);
  else if(tipo == "7") projectRoomUniformity(argv);
  else if(tipo == "8") projectRoomRR(argv);
  
  return 0;
}
