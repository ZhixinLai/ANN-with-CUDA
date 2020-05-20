/********************************************************************

sequential.cu the sequential version of NN 

Input: 
/usr/local/cuda-10.1/bin/nvcc -arch=compute_52 -o sequential.out sequential.cu
./sequential.out block_size activationtype     // block_size = 0; activationtype=1 means sigomid and 2 means ReLU

Output: 
elapsed_time - the elapsed time to perform the multiplication.
accuracy on training set and test set.

********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>

#include <iostream>  
#include <string>  
#include <vector>  
#include <fstream>  
#include <sstream> 
#include<random> 
using namespace std; 


#define X_trn(x, y) X_trn[x * size_train + y] // 196 * 964
#define X_tst(x, y) X_tst[x * size_test + y]  // 196 * 414
#define Y_trn(x, y) Y_trn[x * size_train + y] // 1   * 964
#define Y_tst(x, y) Y_tst[x * size_test + y]  // 1   * 414
#define X(x, y) X[x * size_batch + y]  // 196 * 964
#define Y(x, y) Y[x * size_batch + y]  // 1   * 414


#define W1(x, y) W1[x * size_input + y]       // 20 * 196
#define b1(x, y) b1[x * 1 + y]                // 20 * 1
#define W2(x, y) W2[x * size_hidden + y]      // 2  * 20
#define b2(x, y) b2[x * 1 + y]                // 2  * 1

#define dW1(x, y) dW1[x * size_input + y]     // 20 * 196
#define db1(x, y) db1[x * 1 + y]              // 20 * 1
#define dW2(x, y) dW2[x * size_hidden + y]    // 2  * 20
#define db2(x, y) db2[x * 1 + y]              // 2  * 1

#define Z1(x, y) Z1[x * size_batch + y]       // 20 * 964
#define A1(x, y) A1[x * size_batch + y]       // 20 * 964
#define Z2(x, y) Z2[x * size_batch + y]       // 2  * 964
#define A2(x, y) A2[x * size_batch + y]       // 2  * 964

#define dZ1(x, y) dZ1[x * size_batch + y]     // 20 * 964
#define dA1(x, y) dA1[x * size_batch + y]     // 20 * 964
#define dZ2(x, y) dZ2[x * size_batch + y]     // 2  * 964
#define dA2(x, y) dA2[x * size_batch + y]     // 2  * 964


#define dev_X_trn(x, y) dev_X_trn[x * size_train + y] // 196 * 964
#define dev_X_tst(x, y) dev_X_tst[x * size_test + y]  // 196 * 414
#define dev_Y_trn(x, y) dev_Y_trn[x * size_train + y] // 1   * 964
#define dev_Y_tst(x, y) dev_Y_tst[x * size_test + y]  // 1   * 414
#define dev_X(x, y) dev_X[x * size_batch + y] // 196 * 964
#define dev_Y(x, y) dev_Y[x * size_batch + y]  // 1   * 414


#define dev_W1(x, y) dev_W1[x * size_input + y]       // 20 * 196
#define dev_b1(x, y) dev_b1[x * 1 + y]                // 20 * 1
#define dev_W2(x, y) dev_W2[x * size_hidden + y]      // 2  * 20
#define dev_b2(x, y) dev_b2[x * 1 + y]                // 2  * 1

#define dev_dW1(x, y) dev_dW1[x * size_input + y]     // 20 * 196
#define dev_db1(x, y) dev_db1[x * 1 + y]              // 20 * 1
#define dev_dW2(x, y) dev_dW2[x * size_hidden + y]    // 2  * 20
#define dev_db2(x, y) dev_db2[x * 1 + y]              // 2  * 1

#define dev_Z1(x, y) dev_Z1[x * size_batch + y]       // 20 * 964
#define dev_A1(x, y) dev_A1[x * size_batch + y]       // 20 * 964
#define dev_Z2(x, y) dev_Z2[x * size_batch + y]       // 2  * 964
#define dev_A2(x, y) dev_A2[x * size_batch + y]       // 2  * 964

#define dev_dZ1(x, y) dev_dZ1[x * size_batch + y]     // 20 * 964
#define dev_dA1(x, y) dev_dA1[x * size_batch + y]     // 20 * 964
#define dev_dZ2(x, y) dev_dZ2[x * size_batch + y]     // 2  * 964
#define dev_dA2(x, y) dev_dA2[x * size_batch + y]     // 2  * 964

#define max_index(x, y) max_index[y] // 1  * 964

int size_train  = 964;
int size_test   = 414;
int size_batch  = 0;

int size_input  = 196;
int size_hidden = 20;
int size_output = 2;

int size_X_trn = 196*964;
int size_Y_trn = 1*964;
int size_X_tst = 196*414;
int size_Y_tst = 1*414;
int size_Xbatch = 0;
int size_Ybatch = 0;


int size_W1 = size_hidden*size_input;
int size_b1 = size_hidden*1;
int size_W2 = size_output*size_hidden;
int size_b2 = size_output*1;

int size_dW1 = size_hidden*size_input;
int size_db1 = size_hidden*1;
int size_dW2 = size_output*size_hidden;
int size_db2 = size_output*1;

#define size_Z1 size_hidden*size_batch
#define size_A1 size_hidden*size_batch
#define size_Z2 size_output*size_batch
#define size_A2 size_output*size_batch

#define size_dZ1 size_hidden*size_batch
#define size_dA1 size_hidden*size_batch
#define size_dZ2 size_output*size_batch
#define size_dA2 size_output*size_batch

#define size_max_index 1*size_batch

double *X_trn, *X_tst;
int *Y_trn, *Y_tst;
double *W1, *b1, *W2, *b2;
double *dW1, *db1, *dW2, *db2;
double *Z1, *A1, *Z2, *A2;
double *dZ1, *dA1, *dZ2, *dA2;
int *max_index;



void HiddenLayer(double* dev_X, double* dev_W1, double* dev_b1, double* dev_A1, double* dev_Z1, int size_input, int size_batch, int acti_type, int max_row, int max_col)

{

  for(int i = 0; i < max_row; i++) {
    for(int j = 0; j < max_col; j++) {
      double partial = 0.0;
    	for (int k = 0; k < size_input; k++){
    		partial += dev_W1(i,k) * dev_X(k,j);
      }
    	dev_Z1(i,j) = partial + dev_b1(i,0);
	    // Sigmoid
    	if (acti_type == 1)
    		dev_A1(i,j) = 1 / (1 + exp(0 - dev_Z1(i,j)));
    	// ReLU
    	if (acti_type == 2) {
    		if (dev_Z1(i,j) < 0)
    			dev_A1(i,j) = 0;
    		if (dev_Z1(i,j) >= 0)
    			dev_A1(i,j) = dev_Z1(i,j);
    	}
    }
  }
}

void OutputLayer(double* dev_A1, double* dev_W2, double* dev_b2, double* dev_Z2, int size_hidden, int size_batch, int max_row, int max_col)

{

  for(int i = 0; i < max_row; i++) {
    for(int j = 0; j < max_col; j++) {
	    double partial = 0.0;
    	for (int k = 0; k < size_hidden; k++){
    		partial += dev_W2(i,k) * dev_A1(k,j);
      }
    	dev_Z2(i,j) = partial + dev_b2(i,0);
    }
  }
 
}

void Softmax(double* Z2, int row, int col, double* A2, int* max_index)
{

  int c, r;  
	double max = 0, sum = 0;
	for (c = 0; c < col; c++) {
    max = Z2(0, c);
    max_index[c] = 1;    
		for (r = 1; r < row; r++) {   
			if (Z2(r, c) > max){      
				max = Z2(r, c);        
        max_index[c] = 0;        
      }
		}
		sum = 0;
		for (r = 0; r < row; r++)
			sum += exp(Z2(r, c));
		for (r = 0; r < row; r++)
			A2(r, c) = exp(Z2(r, c)) / sum;
  }
  return;

}


double cross_entropy_loss(int* Y, double* A2, int col) 
{
  
  int c;
  double loss = 0;
  for(c = 0; c < col; c++) {
    loss += -log(A2(0, c)) * Y(0, c) - log(A2(1, c)) * (1-Y(0, c));
  }
  return loss/col;
  
}

/* init Z and A in the host */
void initialize_ZA(int size_batch) {

  Z1 = (double *) malloc(size_Z1*sizeof(double));   // 20*964
  A1 = (double *) malloc(size_A1*sizeof(double));   // 20*964
  Z2 = (double *) malloc(size_Z2*sizeof(double));   // 2*964
  A2 = (double *) malloc(size_A2*sizeof(double));   // 2*964

  dZ1 = (double *) malloc(size_dZ1*sizeof(double));  // 20*964
  dA1 = (double *) malloc(size_dA1*sizeof(double));  // 20*964
  dZ2 = (double *) malloc(size_dZ2*sizeof(double));  // 2*964
  dA2 = (double *) malloc(size_dA2*sizeof(double));  // 2*964
  
  max_index = (int *) malloc(size_max_index*sizeof(int));             // 1*964
    
  memset (Z1,0,  size_Z1);
  memset (A1,0,  size_A1);
  memset (Z2,0,  size_Z2);
  memset (A2,0,  size_A2);
  
  memset (dZ1,0, size_dZ1);
  memset (dA1,0, size_dA1);
  memset (dZ2,0, size_dZ2);
  memset (dA2,0, size_dA2);
  
  memset (max_index,0,size_max_index);

}

void forward(double* X, int* Y, string type, int acti_type,  int block_size){

  if(type == "train"){
    size_batch  = size_train;
    size_Xbatch = size_X_trn;
    size_Ybatch = size_Y_trn;        
  }
  else{
    size_batch = size_test;
    size_Xbatch = size_X_tst;
    size_Ybatch = size_Y_tst;    
  }

  // init Z and A in the host
  initialize_ZA(size_batch);

  // hidden layer and activation function to get Z1 and A1
  HiddenLayer(X, W1, b1, A1, Z1, size_input, size_batch, acti_type, size_hidden, size_batch); 

  // output layer to get Z2
  OutputLayer(A1, W2, b2, Z2, size_hidden, size_batch, size_output, size_batch);
 
  // softmax layer to get A2
  Softmax(Z2, size_output, size_batch, A2, max_index);

}


void Back_dZ2 (double* dev_A2, int* dev_Y_trn, double* dev_dZ2, int size_train, int size_batch, int max_row, int max_col)

{

  for(int i = 0; i < max_row; i++) {
    for(int j = 0; j < max_col; j++) {
    	// int c = threadIdx.x; // column of Z2
      dev_dZ2(0, j) = (dev_A2(0, j) - dev_Y_trn(0, j)) / size_train;
      dev_dZ2(1, j) = (dev_Y_trn(0, j) - dev_A2(0, j)) / size_train;
     }
  }

}

// dW1(20*196) = dZ1(20*964) * X(196*964)
// dW2(2*20) = dZ2(2*964) * A1(20*964)
void Back_dW (double* dev_A, double* dev_dZ, double* dev_dW, int size_batch, int W_col, int max_row, int max_col)

{  

  for(int i = 0; i < max_row; i++) {
    for(int j = 0; j < max_col; j++) {
      double tmp = 0.0;
    	for (int k = 0; k < size_batch; k++) {
    		tmp += dev_dZ[i*size_batch+k] * dev_A[j*size_batch+k];
      }
    	dev_dW[i*W_col+j] = tmp;
    }
  }

}

// db1(20*1) is from dZ1(20*964)
// db2(2*1) is from dZ1(2*964)
void Back_db(double* dZ, double* db, int row, int col, int size_batch)

{
  int r, c;
  for(r = 0; r < row; r++) {
    double tmp = 0;
    for(c = 0; c < col; c++) {
      tmp += dZ[r*size_batch+c];
    }
    db[r*1+0] = tmp;
  }
}
    
void Back_dA1 (double* dev_W2, double* dev_dZ2, double* dev_dA1, int size_batch, int size_hidden, int size_output, int max_row, int max_col)

{  
    
  // dA1(20*964) = dZ2(2*964) * W2(2*20)
  for(int i = 0; i < max_row; i++) {
    for(int j = 0; j < max_col; j++) { 
      double partial = 0.0;
    	for (int k = 0; k < size_output; k++) {
    		partial += dev_W2(k,i) * dev_dZ2(k,j);
      }
    	dev_dA1(i,j) = partial;
    }
  }

}


void Back_dZ1 (double* dev_dA1, double* dev_A1, double* dev_Z1, double* dev_dZ1, int size_batch, int acti_type, int max_row, int max_col)

{  

  for(int i = 0; i < max_row; i++) {
    for(int j = 0; j < max_col; j++) { 
      if(acti_type == 1){ // Sigmoid
          dev_dZ1(i, j) = dev_dA1(i, j) * dev_A1(i, j) * (1-dev_A1(i, j)); // dZ1 = dA1*A1*(1-A1)
      } 
      else if(acti_type == 2) { // ReLU
        if(dev_Z1(i, j) < 0) 
          dev_dZ1(i, j) = 0;
        else
          dev_dZ1(i, j) = dev_dA1(i, j); //dZ1 = dA1*Z1_mask
      }
    }
  }

}

void backprop(int acti_type, int block_size) { // type = 1 is Sigmoid

  // get dZ2
  Back_dZ2(A2, Y_trn, dZ2, size_train, size_train, 1, size_train);

  // get dw2
  Back_dW(A1, dZ2, dW2, size_train, size_hidden, size_output, size_hidden);


  // get db2
  Back_db(dZ2, db2, size_output, size_train, size_train);

  // get dA1
  Back_dA1(W2, dZ2, dA1, size_train, size_hidden, size_output, size_hidden, size_train);    


  // get dZ1
  Back_dZ1(dA1, A1, Z1, dZ1, size_train, acti_type, size_hidden, size_train);


  // get dW1
  Back_dW(X_trn, dZ1, dW1, size_train, size_input, size_hidden, size_input);

  // get b1
  Back_db(dZ1, db1, size_hidden, size_train, size_train);

}

void update_Wb(double* dev_dWb, double* dev_Wb, int col, double learn_rate, int max_row, int max_col)
{

  for(int i = 0; i < max_row; i++) {
    for(int j = 0; j < max_col; j++) { 
      dev_Wb[i*col+j] = dev_Wb[i*col+j] - learn_rate * dev_dWb[i*col+j];
    }
  }
  
}

void updateParameter(double learn_rate, int block_size)
{

  // update w1
  update_Wb(dW1, W1, size_input, learn_rate, size_hidden, size_input);

  // update b1
  update_Wb(db1, b1, 1, learn_rate, size_hidden, 1);

  
  // update w2
  update_Wb(dW2, W2, size_hidden, learn_rate, size_output, size_hidden);
  
  // update b2
  update_Wb(db2, b2, 1, learn_rate, size_output, 1);
 

}


void read_X(string data_path, double* array)
{  
  ifstream inFile(data_path);  
  string row;   
  int p;
  p = 0;
  string value;
  while (getline(inFile, row)){  
    stringstream col(row);    
    while (getline(col, value, ',')){
      array[p] = stod(value);      
      p++;
    }   
  }  
}


void read_Y(string data_path, int* array)
{  
  ifstream inFile(data_path);  
  string row;   
  int p;
  p = 0;
  string value;
  while (getline(inFile, row)){  
    stringstream col(row);    
    while (getline(col, value, ',')){
      array[p] = stod(value);      
      p++;
    }   
  }  
}

/* Set the value and reading data */
void read_data()
{

  X_trn = (double *) malloc(size_X_trn * sizeof(double));  // 196*964
  Y_trn = (int *)    malloc(size_Y_trn * sizeof(int));     // 1*964
  X_tst = (double *) malloc(size_X_tst * sizeof(double));  // 196*414
  Y_tst = (int *)    malloc(size_Y_tst * sizeof(int));     // 1*414
  
  
  string X_trn_path = "X_trn.csv"; // Defined the name of cvs file
  string Y_trn_path = "Y_trn.csv";
  string X_tst_path = "X_tst.csv";
  string Y_tst_path = "Y_tst.csv";
        
  read_X(X_trn_path, X_trn); //Execution 
  read_Y(Y_trn_path, Y_trn);  
  read_X(X_tst_path, X_tst);  
  read_Y(Y_tst_path, Y_tst);  

}

void initialize_Wb() {
  
  W1 = (double *) malloc(size_W1*sizeof(double));   // 20*196
  b1 = (double *) malloc(size_b1*sizeof(double));   // 20*1
  W2 = (double *) malloc(size_W2*sizeof(double));   // 2*20
  b2 = (double *) malloc(size_b2*sizeof(double));   // 2*1
  
  dW1 = (double *) malloc(size_dW1*sizeof(double)); // 20*196
  db1 = (double *) malloc(size_db1*sizeof(double)); // 20*1
  dW2 = (double *) malloc(size_dW2*sizeof(double)); // 2*20
  db2 = (double *) malloc(size_db2*sizeof(double)); // 2*1

  memset (W1,0.5,size_W1);
  memset (b1,0,  size_b1);
  memset (W2,0.5,size_W2);
  memset (b2,0,  size_b2);
  
  memset (dW1,0, size_dW1);
  memset (db1,0, size_db1);
  memset (dW2,0, size_dW2);
  memset (db2,0, size_db2);
  
	default_random_engine e;
	uniform_real_distribution<double> u(-1,1);
 
  for (int i = 0; i < size_W1; i++) {
    W1[i] = u(e);
  }  
  for (int i = 0; i < size_W2; i++) {
    W2[i] = u(e);
  }   
  for (int i = 0; i < size_b1; i++) {
    b1[i] = 0;
  } 
  for (int i = 0; i < size_b2; i++) {
    b2[i] = 0;
  } 
  
}

double accuracy(int* max_index, int* Y, int size_batch) 
{
  
  int i;
  double count = 0;
  for(i = 0; i < size_batch; i++) {
    if(Y(0, i) == max_index(0, i))
      count += 1;
  }  
  return count/double(size_batch);
  
}

double train(double* X_trn, int* Y_trn, int acti_type, int block_size) {

  forward(X_trn, Y_trn, "train", acti_type, block_size);
  backprop(acti_type, block_size); // 1 Sigmoid 2 ReLU 
  updateParameter(0.01, block_size);
  return cross_entropy_loss(Y_trn, A2, size_train);
  
}

double test(double* X, int* Y, string type, int acti_type, int block_size) {

  forward(X, Y, type, acti_type, block_size);
  if(type == "train")
    return accuracy(max_index, Y, size_train);
  else
    return accuracy(max_index, Y, size_test);
  
}

int main(int argc, char *argv[])
{

  int block_size;
  double loss;
  double acc_trn, acc_tst;
  int e;
  int epochs = 20000;
  int acti_type = 1;
   
  if ( argc < 3 ){
    printf(" Usage: first argument: block size \n");
    printf(" second argument: activation type \n");
    return -1;
  } else if ( argc > 3 ) {
    printf("\n Too many arguments. \n");
    return -1;
  } else {
    block_size = atoi(argv[1]);
    acti_type = atoi(argv[2]);
  }
  

  initialize_Wb();
  read_data();
  float elapsed_time = 0.0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for(e = 0; e < epochs; e++) {
    loss = train(X_trn, Y_trn, acti_type, block_size);
    // printf("%f \n", loss);
    // printf("the %d epoch, the training loss is: %f \n", e, loss);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  printf( "Elapsed Time: %.4e msec. \n", elapsed_time );
  
  acc_trn = test(X_trn, Y_trn, "train", acti_type, block_size);
  acc_tst = test(X_tst, Y_tst, "test", acti_type, block_size);
  printf("the %d epoch, the training accuracy is: %f, the test accuracy is: %f\n", e, acc_trn, acc_tst);
  
}

