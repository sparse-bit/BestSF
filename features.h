#include <cuda_profiler_api.h>
#include <iostream>
#include <cusp\csr_matrix.h>
#include <cusp\ell_matrix.h>
#include <cusp\hyb_matrix.h> 
#include <cusp\dia_matrix.h>
#include <cusp\array1d.h>
#include <cusp\array2d.h>
#include <cusp\multiply.h>
#include <cusp\detail\device\spmv\csr_scalar.h>
#include <cusp\detail\host\conversion_utils.h>
#include "timer.h"


/* ___________________________________________________________________________________________ */
// Just some Typedefs to make the code easy to read //
using h_coo = cusp::coo_matrix<int, double, cusp::host_memory>;
using d_coo = cusp::coo_matrix<int, double, cusp::device_memory>;
using h_csr = cusp::csr_matrix<int, double, cusp::host_memory>;
using d_csr = cusp::csr_matrix<int, double, cusp::device_memory>;
using h_ell = cusp::ell_matrix<int, double, cusp::host_memory>;
using d_ell = cusp::ell_matrix<int, double, cusp::device_memory>;
using h_hyb = cusp::hyb_matrix<int, double, cusp::host_memory>;
using d_hyb = cusp::hyb_matrix<int, double, cusp::device_memory>;
using h_dia = cusp::dia_matrix<int, double, cusp::host_memory>;
using d_dia = cusp::dia_matrix<int, double, cusp::device_memory>;
using h_array1D = cusp::array1d <double, cusp::host_memory>;
using d_array1D = cusp::array1d <double, cusp::device_memory>;

unsigned int NB_ITERATIONS = 1000;
unsigned int NO_GPU_FOUND = 999;
unsigned int GPU_ID = 0;

int SelectDevice(std::string device_name){
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		if (prop.name == device_name) {
			std::cout << i << "\t" << device_name << "\n";
			cudaSetDevice(i);
			return i;
		}
	}
	return NO_GPU_FOUND;    // just an impossible value.
}


/******************** Compute the sparsity features ****************/
void compute_n_nnz_max_sd(h_csr &input_matrix, int &n, int &m, int &nnz, float &sd, int &max){
	nnz = input_matrix.num_entries;
	n = input_matrix.num_rows;
	m = input_matrix.num_cols;
	float mu = ((float)nnz / (float)n);

	// Computing max and sd:
	int temp = 0;
	max = 0;
	sd = 0;
	for (int i = 0; i < n; i++){
		temp = input_matrix.row_offsets[i + 1] - input_matrix.row_offsets[i];
		float delta = (float)temp - mu;
		sd = sd + (delta * delta);
		if (temp > max) max = temp;
	}
	sd = sqrt((float)(sd / (float)n));
}

void compute_dis(h_csr &input_matrix, float &dis){
	float avg_ds = 0.0; //if we have a single element in a row
	int i = 0;
	int n = input_matrix.num_rows;
	while (i < n){
		int d_row = 0;
		int j = input_matrix.row_offsets[i];
		int stop_row = input_matrix.row_offsets[i + 1]; int row_element_count = 0;
		while (j < stop_row - 1){
			d_row = d_row + input_matrix.column_indices[j + 1] - input_matrix.column_indices[j] - 1;
			row_element_count++;
			j++;
		}
		if (row_element_count == 0) row_element_count++;
		avg_ds = avg_ds + ((float)d_row / (float)row_element_count);
		i++;
	}
	dis = avg_ds / (float)n;
}

void compute_features(h_csr &input_matrix, int &n, int &m, int &nnz, float &dis, float &sd, int &max, int &dia){

	compute_n_nnz_max_sd(input_matrix, n, m, nnz, sd, max);
	compute_dis(input_matrix, dis);
	dia = cusp::detail::host::count_diagonals(input_matrix);
}


