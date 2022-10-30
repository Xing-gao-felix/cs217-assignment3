#include <stdio.h>

#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    /*************************************************************************/
    // INSERT KERNEL CODE HERE

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int bx,by,tx,ty;
    bx = blockIdx.x;
    by = blockIdx.y;
    tx = threadIdx.x;
    ty = threadIdx.y;

    int row = by * blockDim.y + threadIdx.y;
    int col = bx * blockDim.x + threadIdx.x;

    float c_v = 0;

    // if (row < m && col < n){
    //     C[row*n + col] = 0.0;
    // }
    
    for (int p = 0; p < k/TILE_SIZE+1; p++ ){

        if (row < m && p*TILE_SIZE + tx < k){
            // A[row][p*TILE_SIZE+tx]
            tileA[ty][tx] = A[row*k + p*TILE_SIZE+tx];
        }else{
            tileA[ty][tx] = 0;
        }

        if (col < n && p*TILE_SIZE + ty < k){
            // B[p*TILE_SIZE+ty][col]
            tileB[ty][tx] = B[(p*TILE_SIZE+ty)*n + col];
        }else{
            tileB[ty][tx] = 0;
        }
        __syncthreads();

        if (row < m && col < n){
            for (int i = 0; i < TILE_SIZE; i++){
                c_v += tileA[ty][i] * tileB[i][tx];
            }
        }
        // C[row*n + col] += c_v;
        __syncthreads();

    }

    if (row < m && col < n){
        C[row*n + col] = c_v;
    }
    
    
    // int row = blockIdx.y * blockDim.y + threadIdx.y; 
    // int col = blockIdx.x * blockDim.x + threadIdx.x;

    // if (row < m && col < n){
    //     float p = 0;
    //     for (int i = 0; i < k; k++){
    //         //A[row][i] * B[i][col]
    //         p += A[row*k+i] * B[i*n+col];  
    //     }
    //     C[row*k+col] = p;

    // }


        
    /*************************************************************************/
}

void basicSgemm(int m, int n, int k, const float *A, const float *B, float *C)
{
    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;
	
    /*************************************************************************/
    //INSERT CODE HERE

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(n/BLOCK_SIZE+1, m/BLOCK_SIZE+1);


    /*************************************************************************/

    // Invoke CUDA kernel -----------------------------------------------------

    /*************************************************************************/
    //INSERT CODE HERE
	mysgemm<<<dimGrid,dimBlock>>>(m,n,k,A,B,C);
    /*************************************************************************/
}


