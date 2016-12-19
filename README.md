AUTHORS
=======
	
	Vincent	Garcia
	Eric	Debreuve
	Michel	Barlaud

	

REFERENCE & BIBTEX
==================

    * V. Garcia and E. Debreuve and F. Nielsen and M. Barlaud.
      k-nearest neighbor search: fast GPU-based implementations and application to high-dimensional feature matching.
      In Proceedings of the IEEE International Conference on Image Processing (ICIP), Hong Kong, China, September 2010

	* V. Garcia and E. Debreuve and M. Barlaud.
	  Fast k nearest neighbor search using GPU.
	  In Proceedings of the CVPR Workshop on Computer Vision on GPU, Anchorage, Alaska, USA, June 2008.
		
	* Vincent Garcia
	  Ph.D. Thesis: Suivi d'objets d'intérêt dans une séquence d'images : des points saillants aux mesures statistiques
	  Université de Nice - Sophia Antipolis, Sophia Antipolis, France, December 2008

		
REQUIREMENTS
============

	- The computer must have a CUDA-enabled graphic card (c.f. NVIDIA website)
	- CUDA has to be installed (CUDA drivers and CUDA toolkit)
	- A C compiler has to be installed
	- For using the CUDA code in Matlab, please refer to the CUDA Matlab plug-in webpage for requirements.


			
COMPILATION & EXECUTION
=======================

	The provided code can be used for C and Matlab applications.
	We provide bellow the C and Matlab procedures to compile and execute our code.
	The user must have a basic knowledge of compiling and executing standard examples before trying to compile and execute our code.
    We will consider here that we want to compile the file "knn_cuda_with_indexes.cu".

	For C
		1.	Set the global variable MATLAB_CODE to 0 in file knn_cuda_with_indexes.cu
		2.	Compile the CUDA file with the command line:
			nvcc -o knn_cuda_with_indexes.exe knn_cuda_with_indexes.cu -lcuda -D_CRT_SECURE_NO_DEPRECATE
		3.	Execute the knn_cuda_with_indexes.exe with the command line:
			./knn_cuda_with_indexes.exe
			
			
	For MATLAB
		1.	Set the global variable MATLAB_CODE to 1 in file knn_cuda_with_indexes.cu
		2.	Compile the CUDA file with the Matlab command line:
			nvmex -f nvmexopts.bat knn_cuda_with_indexes.cu -I'C:\CUDA\include' -L'C:\CUDA\lib' -lcufft -lcudart -lcuda -D_CRT_SECURE_NO_DEPRECATE
		3.	Execute the run_matlab.m script

		
		
ORGANISATION OF DATA
====================
	
	In CUDA, it is usual to use the notion of array.
	For our kNN search program, the following array
		
		A = | 1 3 5 |
		    | 2 4 6 |
	
	corresponds to the a set of 3 points of dimension 2:
	
		p1 = (1, 2)
		p2 = (3, 4)
		p3 = (5, 6)
	
	The array A is actually stored in memory as a linear vector:
	
		A = (1, 3, 5, 2, 4, 6)

	The organisation of data is different in Matlab and in CUDA. For Matlab, the previous linear vector
	corresponds to an array of 3 lines and 2 columns:
	
		    | 1 2 |
		A = | 3 4 |
		    | 5 6 |


Speed
=====



  ```
  CUDA

  Number of reference points      :   4096
  Number of query points          :   4096
  Dimension of points             :   32
  Number of neighbors to consider :   20
  Processing kNN search           : done in 15.662722 s for 100 iterations (0.156627 s by iteration)


  cuBLAS

  Number of reference points      :   4096   
  Number of query points          :   4096 
  Dimension of points             :   32 
  Number of neighbors to consider :   20 
  Processing kNN search           : done in 12.456899 s for 100 iterations (0.124569 s by iteration)
  ```


## use_texture

```
Time(%)      Time     Calls       Avg       Min       Max  Name                                           
 78.15%  1.23364s       100  12.336ms  12.279ms  12.389ms  cuInsertionSort(float*, int, int*, int, int, in
t, int)                                                                                                   
 20.09%  317.18ms       100  3.1718ms  3.1694ms  3.1745ms  cuComputeDistanceTexture(int, float*, int, int,
 int, float*)                                                                                             
  0.63%  10.023ms       200  50.116us  49.761us  57.730us  [CUDA memcpy DtoH]                             
  0.55%  8.6914ms       100  86.914us  86.466us  91.651us  [CUDA memcpy HtoA]                             
  0.55%  8.6361ms       100  86.361us  85.794us  92.802us  [CUDA memcpy HtoD]                             
  0.02%  382.09us       100  3.8200us  3.6480us  4.0960us  cuParallelSqrt(float*, int, int, int)          
```


##  no use_texture
```
==30641== Profiling result:                                                                               
Time(%)      Time     Calls       Avg       Min       Max  Name                                           
 88.46%  1.19670s       100  11.967ms  11.921ms  11.993ms  cuInsertionSort(float*, int, int*, int, int, in
t, int)                                                                                                   
  9.49%  128.35ms       100  1.2835ms  1.2822ms  1.2846ms  cuComputeDistanceGlobal(float*, int, int, float
*, int, int, int, float*)                                                                                 
  1.28%  17.344ms       200  86.717us  86.306us  93.058us  [CUDA memcpy HtoD]                             
  0.74%  10.022ms       200  50.110us  49.761us  61.282us  [CUDA memcpy DtoH]                             
  0.03%  380.90us       100  3.8080us  3.6160us  4.1280us  cuParallelSqrt(float*, int, int, int)          
```

## CUBLAS

```
Time(%)      Time     Calls       Avg       Min       Max  Name                                           
 89.75%  1.21083s       100  12.108ms  12.054ms  12.166ms  cuInsertionSort(float*, int, int*, int, int, in
t, int)                                                                                                   
  4.50%  60.706ms       100  607.06us  598.16us  612.56us  cuAddRNorm(float*, int, int, int, float*)      
  2.64%  35.649ms       100  356.49us  342.28us  375.15us  maxwell_sgemm_128x128_raggedMn_nt              
  1.29%  17.369ms       201  86.413us  1.2160us  92.674us  [CUDA memcpy HtoD]                             
  1.02%  13.788ms       100  137.88us  137.73us  138.05us  cuAddQNormAndSqrt(float*, int, int, float*, int
)                                                                                                         
  0.74%  10.023ms       200  50.115us  49.505us  52.354us  [CUDA memcpy DtoH]                             
  0.05%  679.41us       200  3.3970us  3.1360us  3.7440us  cuComputeNorm(float*, int, int, int, float*)   
                                                                                                    ```



