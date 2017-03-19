"""

 Date         12/07/2009
 ====

 Authors      Vincent Garcia
 =======      Eric    Debreuve
              Michel  Barlaud

 Description  Given a reference point set and a query point set, the program returns
 ===========  firts the distance between each query point and its k nearest neighbors in
              the reference point set, and second the indexes of these k nearest neighbors.
              The computation is performed using the API NVIDIA CUDA.

 Paper        Fast k nearest neighbor search using GPU
 =====

 BibTeX       @INPROCEEDINGS{2008_garcia_cvgpu,
 ======         author = {V. Garcia and E. Debreuve and M. Barlaud},
                title = {Fast k nearest neighbor search using GPU},
                booktitle = {CVPR Workshop on Computer Vision on GPU},
                year = {2008},
                address = {Anchorage, Alaska, USA},
                month = {June}
              }
"""


import numpy as np
import cupy
import string




code = '''

/**
  * Gathers k-th smallest distances for each column of the distance matrix in the top.
  *
  * @param dist        distance matrix
  * @param dist_pitch  pitch of the distance matrix given in number of columns
  * @param ind         index matrix
  * @param ind_pitch   pitch of the index matrix given in number of columns
  * @param width       width of the distance matrix and of the index matrix
  * @param height      height of the distance matrix and of the index matrix
  * @param k           number of neighbors to consider
  */
#include <stdio.h>

extern "C"
__global__ void cuInsertionSort(float *dist, int dist_pitch, int *ind, int ind_pitch, int width, int height, int k){

	// Variables
    int l, i, j;
    float *p_dist;
	int   *p_ind;
    float curr_dist, max_dist;
    int   curr_row,  max_row;
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	
    if (xIndex<width){
        
        // Pointer shift, initialization, and max value
        p_dist   = dist + xIndex;
		p_ind    = ind  + xIndex;
        max_dist = p_dist[0];
        p_ind[0] = 1;
        
        // Part 1 : sort kth firt elementZ
        for (l=1; l<k; l++){
            curr_row  = l * dist_pitch;
			curr_dist = p_dist[curr_row];
            if (curr_dist<max_dist){
                i=l-1;
                for (int a=0; a<l-1; a++){
                    if (p_dist[a*dist_pitch]>curr_dist){
                        i=a;
                        break;
                    }
                }
                for (j=l; j>i; j--){
                    p_dist[j*dist_pitch] = p_dist[(j-1)*dist_pitch];
                    p_ind[j*ind_pitch]   = p_ind[(j-1)*ind_pitch];
                }
                p_dist[i*dist_pitch] = curr_dist;
                p_ind[i*ind_pitch]   = l+1;
            }
            else {
                p_ind[l*ind_pitch] = l+1;
            }
            max_dist = p_dist[curr_row];
        }
        
        // Part 2 : insert element in the k-th first lines
        max_row = (k-1)*dist_pitch;
        for (l=k; l<height; l++){
            curr_dist = p_dist[l*dist_pitch];
            if (curr_dist<max_dist){
                i=k-1;
                for (int a=0; a<k-1; a++){
                    if (p_dist[a*dist_pitch]>curr_dist){
						i=a;
						break;
					}
				}
                for (j=k-1; j>i; j--){
					p_dist[j*dist_pitch] = p_dist[(j-1)*dist_pitch];
					p_ind[j*ind_pitch]   = p_ind[(j-1)*ind_pitch];
                }
				        p_dist[i*dist_pitch] = curr_dist;
                p_ind[i*ind_pitch]   = l+1;
                max_dist             = p_dist[max_row];
            }
        }
    }
}
'''


@cupy.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, options=()):
    assert isinstance(options, tuple)
    kernel_code = cupy.cuda.compile_with_cache(code, options=options)
    return kernel_code.get_function(kernel_name)


def insertion_sort(dist, k):
    H, W = dist.shape
    dist = dist.astype(np.float32)
    # dist = cupy.asfortranarray(dist)
    dist = cupy.ascontiguousarray(dist)

    ind = cupy.zeros((W, k), dtype=np.int32, order='F')
    args = (dist, W, ind, W, W, H, k)

    kernel = load_kernel('cuInsertionSort', code)
    grid = (W / 32, 1, 1)
    block = (32, 1, 1)
    kernel(grid=grid, block=block, args=args)
    ind -= 1
    return ind


def test(dist, k):
    ind = insertion_sort(dist, k)

    dist_cpu = dist.get()
    keys = np.argsort(dist_cpu, axis=0)
    np.testing.assert_equal(keys[:k].T, ind.get())


if __name__ == '__main__':
    k = 5
    # (query)
    shape = (64, 32)
    dist = cupy.arange(np.prod(shape)).reshape(*shape)
    dist = cupy.asfortranarray(dist)
    test(dist, k)

    dist = cupy.arange(np.prod(shape)).reshape(*shape)
    dist = dist[::-1]
    test(dist, k)
