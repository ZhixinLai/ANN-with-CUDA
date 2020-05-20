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
#define max_index(x, y) max_index[y] // 1  * 964

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
#define dev_max_index(x, y) dev_max_index[y] // 1  * 964

#define size_train  964
#define size_test   414
#define size_input  196
#define size_hidden 20
#define size_output 2

#define size_X size_input*size_batch
#define size_Y size_batch
#define size_W1 size_hidden*size_input
#define size_b1 size_hidden*1
#define size_W2 size_output*size_hidden
#define size_b2 size_output*1
#define size_dW1 size_hidden*size_input
#define size_db1 size_hidden*1
#define size_dW2 size_output*size_hidden
#define size_db2 size_output*1
#define size_Z1 size_hidden*size_batch
#define size_A1 size_hidden*size_batch
#define size_Z2 size_output*size_batch
#define size_A2 size_output*size_batch
#define size_dZ1 size_hidden*size_batch
#define size_dA1 size_hidden*size_batch
#define size_dZ2 size_output*size_batch
#define size_dA2 size_output*size_batch
#define size_max_index 1*size_batch
#define size_dev_max_index 1*size_batch

int size_batch = 0;
int *Y_trn, *Y_tst, *max_index, *dev_Y, *dev_max_index;
double *X_trn, *X_tst, *X, *W1, *b1, *W2, *b2, *dW1, *db1, *dW2, *db2, *Z1, *A1, *Z2, *A2, *dZ1, *dA1, *dZ2, *dA2;
double *dev_X, *dev_W1, *dev_b1, *dev_W2, *dev_b2, *dev_dW1, *dev_db1, *dev_dW2, *dev_db2, *dev_Z1, *dev_A1, *dev_Z2, *dev_A2, *dev_dZ1, *dev_dA1, *dev_dZ2, *dev_dA2;

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

  X_trn = (double *) malloc(size_input*size_train * sizeof(double));  // 196*964
  Y_trn = (int *)    malloc(size_train * sizeof(int));     // 1*964
  X_tst = (double *) malloc(size_input*size_test * sizeof(double));  // 196*414
  Y_tst = (int *)    malloc(size_train * sizeof(int));     // 1*414
  
  string X_trn_path = "X_trn.csv"; // Defined the name of cvs file
  string Y_trn_path = "Y_trn.csv";
  string X_tst_path = "X_tst.csv";
  string Y_tst_path = "Y_tst.csv";
        
  read_X(X_trn_path, X_trn); //Execution 
  read_Y(Y_trn_path, Y_trn);  
  read_X(X_tst_path, X_tst);  
  read_Y(Y_tst_path, Y_tst);  

}

/* init W b */
void initialize_Wb() {
  
  W1 = (double *) malloc(size_W1*sizeof(double));   // 20*196
  b1 = (double *) malloc(size_b1*sizeof(double));   // 20*1
  W2 = (double *) malloc(size_W2*sizeof(double));   // 2*20
  b2 = (double *) malloc(size_b2*sizeof(double));   // 2*1
  
  dW1 = (double *) malloc(size_dW1*sizeof(double)); // 20*196
  db1 = (double *) malloc(size_db1*sizeof(double)); // 20*1
  dW2 = (double *) malloc(size_dW2*sizeof(double)); // 2*20
  db2 = (double *) malloc(size_db2*sizeof(double)); // 2*1
  
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

/* init Z and A in the host */
void initialize_ZA(int size_batch) 
{

  Z1 = (double *) malloc(size_Z1*sizeof(double));   // 20*964
  A1 = (double *) malloc(size_A1*sizeof(double));   // 20*964
  Z2 = (double *) malloc(size_Z2*sizeof(double));   // 2*964
  A2 = (double *) malloc(size_A2*sizeof(double));   // 2*964
  dZ1 = (double *) malloc(size_dZ1*sizeof(double));  // 20*964
  dA1 = (double *) malloc(size_dA1*sizeof(double));  // 20*964
  dZ2 = (double *) malloc(size_dZ2*sizeof(double));  // 2*964
  dA2 = (double *) malloc(size_dA2*sizeof(double));  // 2*964
  max_index = (int *) malloc(size_max_index*sizeof(int));             // 1*964

}

/* init Z and A in the device */
void initialize_dev_ZA(int size_batch) 
{

  cudaMalloc((void**)&dev_X,  size_X *  sizeof(double));
  cudaMalloc((void**)&dev_Y,  size_Y *  sizeof(int));
  cudaMalloc((void**)&dev_max_index,  size_dev_max_index *  sizeof(int));
  
  cudaMalloc((void**)&dev_Z1, size_Z1 * sizeof(double));
  cudaMalloc((void**)&dev_A1, size_A1 * sizeof(double));  
  cudaMalloc((void**)&dev_Z2, size_Z2 * sizeof(double));
  cudaMalloc((void**)&dev_A2, size_A2 * sizeof(double)); 

}

/* free Z and A in the device */
void free_dev_ZA() 
{

  cudaFree(dev_X);
  cudaFree(dev_Y);   
  cudaFree(dev_max_index);
  cudaFree(dev_Z1); 
  cudaFree(dev_A1);
  cudaFree(dev_Z2);
  cudaFree(dev_A2);

}

/* init W and b in the device */
void initialize_dev_Wb()
{
  
  cudaMalloc((void**)&dev_W1, size_W1 * sizeof(double));
  cudaMalloc((void**)&dev_b1, size_b1 * sizeof(double));
  cudaMalloc((void**)&dev_W2, size_W2 * sizeof(double));
  cudaMalloc((void**)&dev_b2, size_b2 * sizeof(double));
  cudaMalloc((void**)&dev_dW1, size_dW1 * sizeof(double));
  cudaMalloc((void**)&dev_db1, size_db1 * sizeof(double));
  cudaMalloc((void**)&dev_dW2, size_dW2 * sizeof(double));
  cudaMalloc((void**)&dev_db2, size_db2 * sizeof(double));

}

/* free W and b in the device */
void free_dev_Wb()
{
  
  cudaFree(dev_W1);
  cudaFree(dev_b1);   
  cudaFree(dev_W2);
  cudaFree(dev_b2); 
  cudaFree(dev_dW1);
  cudaFree(dev_db1);
  cudaFree(dev_dW2);
  cudaFree(dev_db2);  

}

/* init dZ and dA in the host */
void initialize_dev_dZA(int size_batch)
{
  
  cudaMalloc((void**)&dev_dZ1, size_dZ1 * sizeof(double));
  cudaMalloc((void**)&dev_dA1, size_dA1 * sizeof(double));  
  cudaMalloc((void**)&dev_dZ2, size_dZ2 * sizeof(double));
  cudaMalloc((void**)&dev_dA2, size_dA2 * sizeof(double));

}

/* free dZ and dA in the device */
void free_dev_dZA()
{
  
  cudaFree(dev_dZ1);
  cudaFree(dev_dA1);   
  cudaFree(dev_dZ2);
  cudaFree(dev_dA2); 

}


__global__ void HiddenLayer_Sigmoid(double* dev_X, double* dev_W1, double* dev_b1, double* dev_A1, double* dev_Z1, int K, int size_batch, int max_row, int max_col)

{
   
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int j = threadIdx.x + blockIdx.x * blockDim.x; 
  if(i >= max_row || j >= max_col)
    return;
  
	double partial = 0.0;
	for (int k = 0; k < K; k++)
		partial += dev_W1(i,k) * dev_X(k,j);
	dev_Z1(i,j) = partial + dev_b1(i,0);
 
  dev_A1(i,j) = 1 / (1 + exp(0 - dev_Z1(i,j)));
  
}

__global__ void HiddenLayer_ReLU(double* dev_X, double* dev_W1, double* dev_b1, double* dev_A1, double* dev_Z1, int K, int size_batch, int max_row, int max_col)

{
   
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int j = threadIdx.x + blockIdx.x * blockDim.x; 
  if(i >= max_row || j >= max_col)
    return;
  
	double partial = 0.0;
	for (int k = 0; k < K; k++)
		partial += dev_W1(i,k) * dev_X(k,j);
	dev_Z1(i,j) = partial + dev_b1(i,0);

	dev_A1(i,j) = dev_Z1(i,j) * (dev_Z1(i,j) > 0);
  
}

__global__ void OutputLayer(double* dev_A1, double* dev_W2, double* dev_b2, double* dev_Z2, int K, int size_batch, int max_row, int max_col)

{

  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int j = threadIdx.x + blockIdx.x * blockDim.x; 
  if(i >= max_row || j >= max_col)
    return;

	double partial = 0.0;
 
	for (int k = 0; k < K; k++)
		partial += dev_W2(i,k) * dev_A1(k,j);
	dev_Z2(i,j) = partial + dev_b2(i,0);
 
}

// parallel for column part
__global__ void Softmax(double* dev_Z2, double* dev_A2, int* dev_max_index, int size_batch, int max_row, int max_col)
{

  int j = threadIdx.x + blockIdx.x * blockDim.x; 
  if(j >= max_col)
    return;  

	double max = dev_Z2(0, j), sum = 0;
  dev_max_index[j] = 1;    
  
	for (int i = 1; i < max_row; i++) {   
		if (dev_Z2(i, j) > max){      
			max = dev_Z2(i, j);        
      dev_max_index[j] = 0;        
    }
	}
  
	for (int i = 0; i < max_row; i++)
		sum += exp(dev_Z2(i, j));
	for (int i = 0; i < max_row; i++)
		dev_A2(i, j) = exp(dev_Z2(i, j)) / sum;

}

__global__ void Back_dZ2 (double* dev_A2, int* dev_Y_trn, double* dev_dZ2, int size_batch, int max_row, int max_col)

{
  int i = threadIdx.y + blockIdx.y * blockDim.y;  // y == row; x == col
  int j = threadIdx.x + blockIdx.x * blockDim.x;  // column of Z2
  if(i >= max_row || j >= max_col)
    return;

  dev_dZ2(0, j) = (dev_A2(0, j) - dev_Y_trn(0, j)) / size_batch;
  dev_dZ2(1, j) = (dev_Y_trn(0, j) - dev_A2(0, j)) / size_batch;

}

// dW1(20*196) = dZ1(20*964) * X(196*964)
// dW2(2*20) = dZ2(2*964) * A1(20*964)
__global__ void Back_dW (double* dev_A, double* dev_dZ, double* dev_dW, int size_batch, int W_col, int max_row, int max_col)

{  

	int k;
  double tmp = 0.0;
  int i = threadIdx.y + blockIdx.y * blockDim.y;  // i/y -> row 
  int j = threadIdx.x + blockIdx.x * blockDim.x;  // j/x -> col
  if(i >= max_row || j >= max_col)
    return;
	
	for (k = 0; k < size_batch; k++)
		tmp += dev_dZ[i*size_batch+k] * dev_A[j*size_batch+k];
	dev_dW[i*W_col+j] = tmp;

}

// db1(20*1) is from dZ1(20*964)
// db2(2*1) is from dZ1(2*964)
__global__ void Back_db(double* dev_dZ, double* dev_db, int size_batch, int max_row, int max_col)

{
  
  int i = threadIdx.y + blockIdx.y * blockDim.y;  // i/y -> row
  if(i >= max_row)
    return;
  
  double tmp = 0;
  for(int j = 0; j < max_col; j++) {
    tmp += dev_dZ[i*size_batch+j];
  }
  dev_db[i*1+0] = tmp;

}
 
    
__global__ void Back_dA1 (double* dev_W2, double* dev_dZ2, double* dev_dA1, int size_batch, int K, int max_row, int max_col)

{  
    
  // dA1(20*964) = dZ2(2*964) * W2(2*20)
	int k;
	double partial = 0.0;
  int i = threadIdx.y + blockIdx.y * blockDim.y;  // i/y -> row
  int j = threadIdx.x + blockIdx.x * blockDim.x;  // j/x -> col
  if(i >= max_row || j >= max_col)
    return;
 
	for (k = 0; k < K; k++)
		partial += dev_W2(k,i) * dev_dZ2(k,j);
	dev_dA1(i,j) = partial;

}


__global__ void Back_dZ1_Sigmoid (double* dev_dA1, double* dev_A1, double* dev_Z1, double* dev_dZ1, int size_batch, int max_row, int max_col)

{  

  int i = threadIdx.y + blockIdx.y * blockDim.y;  // y == row; x == col
  int j = threadIdx.x + blockIdx.x * blockDim.x;  // column of Z2
  if(i >= max_row || j >= max_col)
    return;

  dev_dZ1(i, j) = dev_dA1(i, j) * dev_A1(i, j) * (1-dev_A1(i, j)); // dZ1 = dA1*A1*(1-A1)

}

__global__ void Back_dZ1_ReLU (double* dev_dA1, double* dev_A1, double* dev_Z1, double* dev_dZ1, int size_batch, int max_row, int max_col)

{  

  int i = threadIdx.y + blockIdx.y * blockDim.y;  // y == row; x == col
  int j = threadIdx.x + blockIdx.x * blockDim.x;  // column of Z2
  if(i >= max_row || j >= max_col)
    return;

  if(dev_Z1(i, j) < 0) 
    dev_dZ1(i, j) = 0;
  else
    dev_dZ1(i, j) = dev_dA1(i, j); //dZ1 = dA1*Z1_mask

}


__global__ void update_Wb(double* dev_dWb, double* dev_Wb, int col, double learn_rate, int max_row, int max_col)
{

  int i = threadIdx.y + blockIdx.y * blockDim.y;  // y == row; x == col
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  if(i >= max_row || j >= max_col)
    return;
  
  dev_Wb[i*col+j] = dev_Wb[i*col+j] - learn_rate * dev_dWb[i*col+j];
  
}


/* forward to calculate A Z preY */
void forward(double* X, int* Y, string type, int acti_type,  int block_size){

  if(type == "train"){
    size_batch  = size_train;       
  }
  else{
    size_batch = size_test;   
  }

  // init Z and A in the host
  initialize_ZA(size_batch);
  // init X Y W b Z A in the device
  initialize_dev_ZA(size_batch);

  dim3 dimBlock(block_size, block_size);
  // hidden layer and activation function to get Z1 and A1
  dim3 dimGrid1((size_batch + dimBlock.x - 1)/ dimBlock.x, (size_hidden+ dimBlock.y - 1)/ dimBlock.y);       
  cudaMemcpy(dev_W1, W1, size_W1 * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b1, b1, size_b1 * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_X,  X,  size_X  * sizeof(double), cudaMemcpyHostToDevice);  
  if(acti_type == 1)
    HiddenLayer_Sigmoid<<<dimGrid1, dimBlock>>>(dev_X, dev_W1, dev_b1, dev_A1, dev_Z1, size_input, size_batch, size_hidden, size_batch); 
  else if(acti_type == 2)
    HiddenLayer_ReLU<<<dimGrid1, dimBlock>>>(dev_X, dev_W1, dev_b1, dev_A1, dev_Z1, size_input, size_batch, size_hidden, size_batch); 
  cudaMemcpy(Z1, dev_Z1, size_Z1 * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(A1, dev_A1, size_A1 * sizeof(double), cudaMemcpyDeviceToHost);
 
  // output layer to get Z2
  dim3 dimGrid2((size_batch + dimBlock.x - 1)/ dimBlock.x, (size_output + dimBlock.y - 1)/ dimBlock.y);   
  cudaMemcpy(dev_W2, W2, size_W2 * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b2, b2, size_b2 * sizeof(double), cudaMemcpyHostToDevice);
  OutputLayer<<<dimGrid2, dimBlock>>>(dev_A1, dev_W2, dev_b2, dev_Z2, size_hidden, size_batch, size_output, size_batch);
  cudaMemcpy(Z2, dev_Z2, size_Z2 * sizeof(double), cudaMemcpyDeviceToHost);

  // softmax layer to get A2 and max_index
  dim3 dimGrid3((size_batch + dimBlock.x - 1)/ dimBlock.x, (size_output + dimBlock.y - 1)/ dimBlock.y);
  Softmax<<<dimGrid3, dimBlock>>>(dev_Z2, dev_A2, dev_max_index, size_batch, size_output, size_batch);
  cudaMemcpy(A2, dev_A2, size_A2 * sizeof(double), cudaMemcpyDeviceToHost);  
  cudaMemcpy(max_index, dev_max_index, size_max_index * sizeof(int), cudaMemcpyDeviceToHost);  

  free_dev_ZA();
}

/* calculate loss  */
double cross_entropy_loss(int* Y, double* A2, int col) 
{

  double loss = 0;
  for(int c = 0; c < col; c++) {
    loss += -log(A2(0, c)) * Y(0, c) - log(A2(1, c)) * (1-Y(0, c));
  }
  return loss/col;
  
}

/* backward to calculate dW db  */
void backprop(double* X, int* Y, int acti_type, int block_size) { // type = 1 is Sigmoid
  
  size_batch = size_train;
  initialize_dev_ZA(size_batch);

  dim3 dimBlock(block_size, block_size);
  // get dZ2
  dim3 dimGrid1((size_batch + dimBlock.x - 1)/ dimBlock.x, (1 + dimBlock.y - 1)/ dimBlock.y); 
  cudaMemcpy(dev_A2, A2, size_A2 * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_Y, Y, size_Y * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_dZ2,  dZ2,  size_dZ2  * sizeof(double), cudaMemcpyHostToDevice);
  Back_dZ2<<<dimGrid1, dimBlock>>>(dev_A2, dev_Y, dev_dZ2, size_batch, 1, size_batch);
  cudaMemcpy(dZ2, dev_dZ2, size_dZ2 * sizeof(double), cudaMemcpyDeviceToHost);

  // get dw2
  dim3 dimGrid2((size_hidden + dimBlock.x - 1)/ dimBlock.x, (size_output + dimBlock.y - 1)/ dimBlock.y); 
  cudaMemcpy(dev_A1, A1, size_A1 * sizeof(double), cudaMemcpyHostToDevice);
  Back_dW<<<dimGrid2, dimBlock>>>(dev_A1, dev_dZ2, dev_dW2, size_batch, size_hidden, size_output, size_hidden);
  cudaMemcpy(dW2, dev_dW2, size_dW2 * sizeof(double), cudaMemcpyDeviceToHost);

  // get db2
  dim3 dimGrid3((1 + dimBlock.x - 1)/ dimBlock.x, (size_output + dimBlock.y - 1)/ dimBlock.y);   
  Back_db<<<dimGrid3, dimBlock>>>(dev_dZ2, dev_db2, size_batch, size_output, size_batch);
  cudaMemcpy(db2, dev_db2, size_db2 * sizeof(double), cudaMemcpyDeviceToHost); 

  // get dA1
  dim3 dimGrid4((size_batch + dimBlock.x - 1)/ dimBlock.x, (size_hidden + dimBlock.y - 1)/ dimBlock.y);  
  cudaMemcpy(dev_W2, W2, size_W2 * sizeof(double), cudaMemcpyHostToDevice);
  Back_dA1<<<dimGrid4, dimBlock>>> (dev_W2, dev_dZ2, dev_dA1, size_batch, size_output, size_hidden, size_batch);      
  cudaMemcpy(dA1, dev_dA1, size_dA1 * sizeof(double), cudaMemcpyDeviceToHost);

  // get dZ1
  dim3 dimGrid5((size_batch + dimBlock.x - 1)/ dimBlock.x, (size_hidden + dimBlock.y - 1)/ dimBlock.y); 
  cudaMemcpy(dev_A1, A1, size_A1 * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_Z1, Z1, size_Z1 * sizeof(double), cudaMemcpyHostToDevice);
  if(acti_type == 1)
    Back_dZ1_Sigmoid<<<dimGrid5, dimBlock>>>(dev_dA1, dev_A1, dev_Z1, dev_dZ1, size_batch, size_hidden, size_batch);
  else if(acti_type == 2)
    Back_dZ1_ReLU<<<dimGrid5, dimBlock>>>(dev_dA1, dev_A1, dev_Z1, dev_dZ1, size_batch, size_hidden, size_batch);    
  cudaMemcpy(dZ1, dev_dZ1, size_dZ1 * sizeof(double), cudaMemcpyDeviceToHost);

  // get dW1
  dim3 dimGrid6((size_input + dimBlock.x - 1)/ dimBlock.x, (size_hidden + dimBlock.y - 1)/ dimBlock.y); 
  cudaMemcpy(dev_X, X, size_X * sizeof(double), cudaMemcpyHostToDevice);
  Back_dW<<<dimGrid6, dimBlock>>>(dev_X, dev_dZ1, dev_dW1, size_batch, size_input, size_hidden, size_input);
  cudaMemcpy(dW1, dev_dW1, size_dW1 * sizeof(double), cudaMemcpyDeviceToHost);

  // get b1
  dim3 dimGrid7((1 + dimBlock.x - 1)/ dimBlock.x, (size_hidden + dimBlock.y - 1)/ dimBlock.y);   
  Back_db<<<dimGrid7, dimBlock>>>(dev_dZ1, dev_db1, size_batch, size_hidden, size_batch);
  cudaMemcpy(db1, dev_db1, size_db1 * sizeof(double), cudaMemcpyDeviceToHost); 

  // free ZA on device
  free_dev_ZA();
  
}

/* update W b  */
void updateParameter(double learn_rate, int block_size)
{

  dim3 dimBlock(block_size, block_size);
  
  // update w1
  cudaMemcpy(dev_dW1,  dW1, size_dW1  * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_W1,   W1,  size_W1   * sizeof(double), cudaMemcpyHostToDevice);
  dim3 dimGrid1((size_input + dimBlock.x - 1)/ dimBlock.x, (size_hidden + dimBlock.y - 1)/ dimBlock.y); 
  update_Wb<<<dimGrid1, dimBlock>>>(dev_dW1, dev_W1, size_input, learn_rate, size_hidden, size_input);
  cudaMemcpy(W1, dev_W1, size_W1 * sizeof(double), cudaMemcpyDeviceToHost);

  // update b1
  cudaMemcpy(dev_db1,  db1, size_db1  * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b1,   b1,  size_b1   * sizeof(double), cudaMemcpyHostToDevice);
  dim3 dimGrid2((1 + dimBlock.x - 1)/ dimBlock.x, (size_hidden + dimBlock.y - 1)/ dimBlock.y); 
  update_Wb<<<dimGrid2, dimBlock>>>(dev_db1, dev_b1, 1, learn_rate, size_hidden, 1);
  cudaMemcpy(b1, dev_b1, size_b1 * sizeof(double), cudaMemcpyDeviceToHost);
  
  // update w2
  cudaMemcpy(dev_dW2,  dW2, size_dW2  * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_W2,   W2,  size_W2   * sizeof(double), cudaMemcpyHostToDevice);
  dim3 dimGrid3((size_hidden + dimBlock.x - 1)/ dimBlock.x, (size_output + dimBlock.y - 1)/ dimBlock.y); 
  update_Wb<<<dimGrid3, dimBlock>>>(dev_dW2, dev_W2, size_hidden, learn_rate, size_output, size_hidden);
  cudaMemcpy(W2, dev_W2, size_W2 * sizeof(double), cudaMemcpyDeviceToHost);
  
  // update b2
  cudaMemcpy(dev_db2,  db2, size_db2  * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b2,   b2,  size_b2   * sizeof(double), cudaMemcpyHostToDevice);
  dim3 dimGrid4((1 + dimBlock.x - 1)/ dimBlock.x, (size_output + dimBlock.y - 1)/ dimBlock.y); 
  update_Wb<<<dimGrid4, dimBlock>>>(dev_db2, dev_b2, 1, learn_rate, size_output, 1);
  cudaMemcpy(b2, dev_b2, size_b2 * sizeof(double), cudaMemcpyDeviceToHost);

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

void train(double* X_trn, int* Y_trn, int acti_type, int block_size) 
{

  forward(X_trn, Y_trn, "train", acti_type, block_size);
  backprop(X_trn, Y_trn, acti_type, block_size); // 1 Sigmoid 2 ReLU 
  updateParameter(0.01, block_size);
  
}

double test(double* X, int* Y, string type, int acti_type, int block_size) 
{

  forward(X, Y, type, acti_type, block_size);
  if(type == "train")
    return accuracy(max_index, Y, size_train);
  else
    return accuracy(max_index, Y, size_test);
  
}

int main(int argc, char *argv[])
{

  int block_size;
  int epochs = 20000;
  int acti_type;
  double acc_trn, acc_tst;

  if ( argc < 3 ){
    printf(" Usage: first argument: dimension of square matrix \n");
    printf("        second argument: size of CUDA block \n");
    return -1;
  } else if ( argc > 3 ) {
    printf("\n Too many arguments. \n");
    return -1;
  } else {
    block_size = atoi(argv[1]);
    acti_type = atoi(argv[2]);
  }
  

  initialize_Wb();
  initialize_dev_Wb();
  initialize_dev_dZA(size_train);
  read_data();
  float elapsed_time = 0.0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for(int e = 0; e < epochs; e++) {
    train(X_trn, Y_trn, acti_type, block_size);
    // double loss = cross_entropy_loss(Y_trn, A2, size_train);
    // printf("%f\n", loss);    
    // printf("the %d epoch, the training loss is: %f \n", e, loss);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  printf( "Elapsed Time: %.4e msec. \n", elapsed_time );
  
  acc_trn = test(X_trn, Y_trn, "train", acti_type, block_size);
  acc_tst = test(X_tst, Y_tst, "test", acti_type, block_size);
  printf("the training accuracy is: %f, the test accuracy is: %f\n", acc_trn, acc_tst);
  
  free_dev_Wb();
  free_dev_dZA();
  
}