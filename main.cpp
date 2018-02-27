#include <cuda_profiler_api.h>
#include <iostream>
#include <stdio.h>
#include <cusp\io\matrix_market.h>
#include <Windows.h>
#include "BestSF.h"


int main(int argc, char* argv[])
{ 
	if (argc < 3) return EXIT_FAILURE;  

	// Read the input matrix (.mtx file)
	std::string matrix_name = (std::string) argv[1];
	h_csr matrix;
	cusp::io::read_matrix_market_file(matrix, matrix_name);
	if (matrix.num_entries == 0) return 0;
	std::cout << matrix_name << "\t";

	// Select the GPU (Should be changed according to the number of GPUs available in the system)
	std::string gpu_name = (std::string)(argv[2]);
	std::string  gpu_full_name;
	if (gpu_name == "M") gpu_full_name = "GeForce GTX TITAN X";
	if (gpu_name == "P") gpu_full_name = "GeForce GTX 1080";
	GPU_ID = SelectDevice(gpu_full_name);


	// compute the sparsity features
	int n = 0, m = 0, nnz = 0, max = 0, dia = 0; float dis = 0.0, sd = 0.0;  
	compute_features(matrix, n, m, nnz, dis, sd, max, dia);
	float mu = float(nnz) / float(n); 
	float d = float(nnz) / float(n * m);
	float cv = sd / mu;
	std::cout << n << "\t" << m << "\t" << nnz << "\t" << mu << "\t" << d << "\t" << sd  << "\t" << cv << "\t" << max << "\t" << max-mu << "\t" << dis << "\t" << dia << "\t";


	// Collect the training data. Here we report the execution time. The performance (FLOPS) can be calculated as (2*nnz/time_exec).
	double time_exec = 0;

	//std::cout << "\n---------------COO--------------\n";
		try{
			d_array1D d_x_coo(matrix.num_cols, 1.0);
			d_array1D d_y_coo(matrix.num_rows, 0.0);
			d_coo d_matrix_coo(matrix);
			time_exec = coo_kernel(d_matrix_coo, d_x_coo, d_y_coo);
			std::cout << time_exec << "\t";
		}
		catch (std::exception &e){
			std::cout << "coo err" << " \t";
		}
	
		cudaDeviceReset();
	
	
		//std::cout << "\n---------------CSR--------------\n";
	
		try{
			d_array1D d_x_csr(matrix.num_cols, 1.0);
			d_array1D d_y_csr(matrix.num_rows, 0.0);
			d_csr d_matrix_csr(matrix);
			time_exec = csr_kernel(d_matrix_csr, d_x_csr, d_y_csr);
			std::cout << time_exec << "\t";
		}
		catch (std::exception &e){
			std::cout << "csr err" << " \t";
		}
		cudaDeviceReset();
	
		//std::cout << "\n---------------DIA--------------\n";
		try{
			d_array1D d_x_dia(matrix.num_cols, 1.0);
			d_array1D d_y_dia(matrix.num_rows, 0.0); 
			d_dia d_matrix_dia(matrix);
			time_exec = dia_kernel(d_matrix_dia, d_x_dia, d_y_dia);
			std::cout << time_exec << "\t";
		}
		catch (std::exception &e){
			std::cout << "DIA err" << " \t";   // If the matrix is not convertible to DIA format.
		}
		cudaDeviceReset();
		 
	
		//std::cout << "\n---------------ELL--------------\n";
		try{
			d_array1D d_x_ell(matrix.num_cols, 1.0);
			d_array1D d_y_ell(matrix.num_rows, 0.0);
			d_ell d_matrix_ell(matrix);
			time_exec = ell_kernel(d_matrix_ell, d_x_ell, d_y_ell); 
			std::cout << time_exec << "\t";
		}
		catch (std::exception &e){
			std::cout << "ell err" << " \t"; // If the matrix is not convertible to ELL format.
		}
		cudaDeviceReset();
	
	
		//std::cout << "\n---------------HYB--------------\n";
		try{
			d_array1D d_x_hyb(matrix.num_cols, 1.0);
			d_array1D d_y_hyb(matrix.num_rows, 0.0);
			d_hyb d_matrix_hyb(matrix);
			if (d_matrix_hyb.coo.num_entries == 0) std::cout << "ELL" << "\t";
			else if (d_matrix_hyb.ell.num_entries == 0) std::cout << "COO" << "\t";
			else {
				time_exec = hyb_kernel(d_matrix_hyb, d_x_hyb, d_y_hyb);
				std::cout << time_exec << "\t";
			}
		}
		catch (std::exception &e){
			std::cout << "HYB err" << "\t";
		}
		cudaDeviceReset();
	
		std::cout << "\n";
	
	return 0;
}
