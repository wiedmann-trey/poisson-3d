#include <mpi.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <time.h>
#include <stdlib.h>
#include <iostream>

// 3D test function: u(x,y,z) = sin(2πx)*sin(2πy)*sin(2πz)
#define U(x,y,z) (cos(2*M_PI*x)*cos(2*M_PI*y)*cos(2*M_PI*z))
#define RHS(x,y,z) (-3*(2*M_PI)*(2*M_PI)*cos(2*M_PI*x)*cos(2*M_PI*y)*cos(2*M_PI*z))
// Laplacian: ∇²u = -3*(2π)²*sin(2πx)*sin(2πy)*sin(2πz)

struct timespec start, end, total_start, total_end;

double elapsed_time(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

struct SolverState {
    int nx, ny, nz;  // Grid sizes in each dimension
    int max_iter;
    int check_convergence_every_n;
    double convergence_bound;
    
    // Boundary box
    double boundary_start_x, boundary_start_y, boundary_start_z;
    double boundary_end_x, boundary_end_y, boundary_end_z;
    double domain_size_x, domain_size_y, domain_size_z;
    
    // MPI topology
    MPI_Comm cart_comm;
    int rank, size;
    int dims[3], coords[3];
    int north, south, east, west, top, bottom;
    
    // Grid dimensions
    int local_nx, local_ny, local_nz;
    int global_x_start, global_y_start, global_z_start;
    double delta_x, delta_y, delta_z;
    double a_x, a_y, a_z, a_f;
    
    // Host arrays
    double *h_rhs, *h_u;
    double *h_north_data, *h_south_data, *h_east_data, *h_west_data, *h_top_data, *h_bottom_data;
    double *h_out_east, *h_out_west, *h_out_north, *h_out_south, *h_out_top, *h_out_bottom;
    
    // Device arrays
    double *d_rhs, *d_u;
    double *d_north_data, *d_south_data, *d_east_data, *d_west_data, *d_top_data, *d_bottom_data;
    double *d_out_east, *d_out_west, *d_out_north, *d_out_south, *d_out_top, *d_out_bottom;
    double *d_residual_sum, *d_mse_total;
    
    // Kernel launch parameters
    dim3 blockDim, gridDim;
    dim3 extractBlockDim, extractGridDim;
    
    // Statistics and logging
    double avg_update_time, avg_reduction_time;
    int n_iters, n_reductions;
    bool write_solution;
    int write_solution_every_n;
    std::string output_dir;
};

// Red-Black Gauss-Seidel kernel
__global__
void red_black_kernel(double *u, double *rhs, 
                      double *north_data, double *south_data,
                      double *east_data, double *west_data,
                      double *top_data, double *bottom_data,
                      int local_nx, int local_ny, int local_nz, 
                      double a_x, double a_y, double a_z, double a_f,
                      int red_pass) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    
    if(i < local_nz && j < local_ny && k < local_nx) {
        int sum = i + j + k;
        if((sum % 2) != red_pass) return;
        
        int idx = i*local_ny*local_nx + j*local_nx + k;
        
        double west_val = (k == 0) ? west_data[i*local_ny + j] : u[i*local_ny*local_nx + j*local_nx + k-1];
        double east_val = (k == local_nx-1) ? east_data[i*local_ny + j] : u[i*local_ny*local_nx + j*local_nx + k+1];
        double south_val = (j == 0) ? south_data[i*local_nx + k] : u[i*local_ny*local_nx + (j-1)*local_nx + k];
        double north_val = (j == local_ny-1) ? north_data[i*local_nx + k] : u[i*local_ny*local_nx + (j+1)*local_nx + k];
        double bottom_val = (i == 0) ? bottom_data[j*local_nx + k] : u[(i-1)*local_ny*local_nx + j*local_nx + k];
        double top_val = (i == local_nz-1) ? top_data[j*local_nx + k] : u[(i+1)*local_ny*local_nx + j*local_nx + k];
        
        // u^(n+1) = a_x(u_i-1 + u_i+1) + a_y(u_j-1 + u_j+1) + a_z(u_k-1 + u_k+1) - a_f*f
        u[idx] = a_x * (west_val + east_val) + 
                 a_y * (south_val + north_val) + 
                 a_z * (bottom_val + top_val) - 
                 a_f * rhs[idx];
    }
}

__global__
void extract_boundaries(double *u, double *east_out, double *west_out,
                        double *north_out, double *south_out,
                        double *top_out, double *bottom_out,
                        int local_nx, int local_ny, int local_nz) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid < local_ny * local_nz) {
        int i = tid / local_ny;
        int j = tid % local_ny;
        west_out[i*local_ny + j] = u[i*local_ny*local_nx + j*local_nx + 0];
        east_out[i*local_ny + j] = u[i*local_ny*local_nx + j*local_nx + (local_nx-1)];
    }
    
    if(tid < local_nx * local_nz) {
        int i = tid / local_nx;
        int k = tid % local_nx;
        south_out[i*local_nx + k] = u[i*local_ny*local_nx + 0*local_nx + k];
        north_out[i*local_nx + k] = u[i*local_ny*local_nx + (local_ny-1)*local_nx + k];
    }
    
    if(tid < local_nx * local_ny) {
        int j = tid / local_nx;
        int k = tid % local_nx;
        bottom_out[j*local_nx + k] = u[0*local_ny*local_nx + j*local_nx + k];
        top_out[j*local_nx + k] = u[(local_nz-1)*local_ny*local_nx + j*local_nx + k];
    }
}

__global__
void residual_kernel(double *u, double *rhs, double *residual_sum,
                     double *north_data, double *south_data,
                     double *east_data, double *west_data,
                     double *top_data, double *bottom_data,
                     int local_nx, int local_ny, int local_nz, 
                     double delta_x, double delta_y, double delta_z) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    extern __shared__ double shared_res[];
    int tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

    double local_res = 0.0;
    if(i < local_nz && j < local_ny && k < local_nx) {
        int idx = i*local_ny*local_nx + j*local_nx + k;
        
        double west_val = (k == 0) ? west_data[i*local_ny + j] : u[i*local_ny*local_nx + j*local_nx + k-1];
        double east_val = (k == local_nx-1) ? east_data[i*local_ny + j] : u[i*local_ny*local_nx + j*local_nx + k+1];
        double south_val = (j == 0) ? south_data[i*local_nx + k] : u[i*local_ny*local_nx + (j-1)*local_nx + k];
        double north_val = (j == local_ny-1) ? north_data[i*local_nx + k] : u[i*local_ny*local_nx + (j+1)*local_nx + k];
        double bottom_val = (i == 0) ? bottom_data[j*local_nx + k] : u[(i-1)*local_ny*local_nx + j*local_nx + k];
        double top_val = (i == local_nz-1) ? top_data[j*local_nx + k] : u[(i+1)*local_ny*local_nx + j*local_nx + k];
        
        // Compute Laplacian: ∇²u = (u_i-1 - 2u + u_i+1)/Δx² + (u_j-1 - 2u + u_j+1)/Δy² + (u_k-1 - 2u + u_k+1)/Δz²
        double laplacian = (west_val - 2.0*u[idx] + east_val) / (delta_x * delta_x) +
                          (south_val - 2.0*u[idx] + north_val) / (delta_y * delta_y) +
                          (bottom_val - 2.0*u[idx] + top_val) / (delta_z * delta_z);
        double res = rhs[idx] - laplacian;
        local_res = res * res;
    }

    shared_res[tid] = local_res;
    __syncthreads();

    if(tid == 0) {
        double block_res = 0.0;
        for(int i = 0; i < (blockDim.x*blockDim.y*blockDim.z); i++) {
            block_res += shared_res[i];
        }
        atomicAdd(residual_sum, block_res);
    }
}

__global__
void mse_kernel(double *u, double *mse_total, int local_nx, int local_ny, int local_nz,
                int global_x_start, int global_y_start, int global_z_start,
                int nx, int ny, int nz,
                double boundary_start_x, double boundary_start_y, double boundary_start_z,
                double domain_size_x, double domain_size_y, double domain_size_z) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    extern __shared__ double shared_mse[];
    int tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

    double local_error = 0.0;
    if(i < local_nz && j < local_ny && k < local_nx) {
        double x = boundary_start_x + (k + global_x_start + 1) * domain_size_x / (nx + 1);
        double y = boundary_start_y + (j + global_y_start + 1) * domain_size_y / (ny + 1);
        double z = boundary_start_z + (i + global_z_start + 1) * domain_size_z / (nz + 1);
        
        double u_exact = U(x, y, z);
        double diff = u[i*local_ny*local_nx + j*local_nx + k] - u_exact;
        local_error = diff * diff;
    }

    shared_mse[tid] = local_error;
    __syncthreads();

    if(tid == 0) {
        double block_error = 0.0;
        for(int i = 0; i < (blockDim.x*blockDim.y*blockDim.z); i++) {
            block_error += shared_mse[i];
        }
        atomicAdd(mse_total, block_error);
    }
}

void initialize_grid(SolverState &state) {
    int local_size = state.local_nx * state.local_ny * state.local_nz;
    
    state.h_rhs = (double*)malloc(local_size * sizeof(double));
    state.h_u = (double*)malloc(local_size * sizeof(double));
    state.h_north_data = (double*)malloc(state.local_nx * state.local_nz * sizeof(double));
    state.h_south_data = (double*)malloc(state.local_nx * state.local_nz * sizeof(double));
    state.h_east_data = (double*)malloc(state.local_ny * state.local_nz * sizeof(double));
    state.h_west_data = (double*)malloc(state.local_ny * state.local_nz * sizeof(double));
    state.h_top_data = (double*)malloc(state.local_nx * state.local_ny * sizeof(double));
    state.h_bottom_data = (double*)malloc(state.local_nx * state.local_ny * sizeof(double));
    state.h_out_east = (double*)malloc(state.local_ny * state.local_nz * sizeof(double));
    state.h_out_west = (double*)malloc(state.local_ny * state.local_nz * sizeof(double));
    state.h_out_north = (double*)malloc(state.local_nx * state.local_nz * sizeof(double));
    state.h_out_south = (double*)malloc(state.local_nx * state.local_nz * sizeof(double));
    state.h_out_top = (double*)malloc(state.local_nx * state.local_ny * sizeof(double));
    state.h_out_bottom = (double*)malloc(state.local_nx * state.local_ny * sizeof(double));

    hipMalloc(&state.d_rhs, local_size * sizeof(double));
    hipMalloc(&state.d_u, local_size * sizeof(double));
    hipMalloc(&state.d_north_data, state.local_nx * state.local_nz * sizeof(double));
    hipMalloc(&state.d_south_data, state.local_nx * state.local_nz * sizeof(double));
    hipMalloc(&state.d_east_data, state.local_ny * state.local_nz * sizeof(double));
    hipMalloc(&state.d_west_data, state.local_ny * state.local_nz * sizeof(double));
    hipMalloc(&state.d_top_data, state.local_nx * state.local_ny * sizeof(double));
    hipMalloc(&state.d_bottom_data, state.local_nx * state.local_ny * sizeof(double));
    hipMalloc(&state.d_out_east, state.local_ny * state.local_nz * sizeof(double));
    hipMalloc(&state.d_out_west, state.local_ny * state.local_nz * sizeof(double));
    hipMalloc(&state.d_out_north, state.local_nx * state.local_nz * sizeof(double));
    hipMalloc(&state.d_out_south, state.local_nx * state.local_nz * sizeof(double));
    hipMalloc(&state.d_out_top, state.local_nx * state.local_ny * sizeof(double));
    hipMalloc(&state.d_out_bottom, state.local_nx * state.local_ny * sizeof(double));
    hipMalloc(&state.d_residual_sum, sizeof(double));
    hipMalloc(&state.d_mse_total, sizeof(double));

    // Initialize RHS and solution
    for(int i = 0; i < state.local_nz; i++) {
        for(int j = 0; j < state.local_ny; j++) {
            for(int k = 0; k < state.local_nx; k++) {
                double x = state.boundary_start_x + (k + state.global_x_start + 1) * state.domain_size_x / (state.nx + 1);
                double y = state.boundary_start_y + (j + state.global_y_start + 1) * state.domain_size_y / (state.ny + 1);
                double z = state.boundary_start_z + (i + state.global_z_start + 1) * state.domain_size_z / (state.nz + 1);
                
                state.h_rhs[i*state.local_ny*state.local_nx + j*state.local_nx + k] = RHS(x,y,z);
                state.h_u[i*state.local_ny*state.local_nx + j*state.local_nx + k] = 0;
            }
        }
    }

    hipMemcpy(state.d_rhs, state.h_rhs, local_size * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(state.d_u, state.h_u, local_size * sizeof(double), hipMemcpyHostToDevice);
}

void initialize_boundaries(SolverState &state) {
    // Initialize x-direction boundaries (west/east)
    for(int i = 0; i < state.local_nz; i++) {
        for(int j = 0; j < state.local_ny; j++) {
            if(state.coords[0] == 0) {
                double x = state.boundary_start_x;
                double y = state.boundary_start_y + (j + state.global_y_start + 1) * state.domain_size_y / (state.ny + 1);
                double z = state.boundary_start_z + (i + state.global_z_start + 1) * state.domain_size_z / (state.nz + 1);
                state.h_west_data[i*state.local_ny + j] = U(x,y,z);
            } else state.h_west_data[i*state.local_ny + j] = 0;
            
            if(state.coords[0] == state.dims[0]-1) {
                double x = state.boundary_end_x;
                double y = state.boundary_start_y + (j + state.global_y_start + 1) * state.domain_size_y / (state.ny + 1);
                double z = state.boundary_start_z + (i + state.global_z_start + 1) * state.domain_size_z / (state.nz + 1);
                state.h_east_data[i*state.local_ny + j] = U(x,y,z);
            } else state.h_east_data[i*state.local_ny + j] = 0;
        }
    }

    // Initialize y-direction boundaries (south/north)
    for(int i = 0; i < state.local_nz; i++) {
        for(int k = 0; k < state.local_nx; k++) {
            if(state.coords[1] == 0) {
                double x = state.boundary_start_x + (k + state.global_x_start + 1) * state.domain_size_x / (state.nx + 1);
                double y = state.boundary_start_y;
                double z = state.boundary_start_z + (i + state.global_z_start + 1) * state.domain_size_z / (state.nz + 1);
                state.h_south_data[i*state.local_nx + k] = U(x,y,z);
            } else state.h_south_data[i*state.local_nx + k] = 0;
            
            if(state.coords[1] == state.dims[1]-1) {
                double x = state.boundary_start_x + (k + state.global_x_start + 1) * state.domain_size_x / (state.nx + 1);
                double y = state.boundary_end_y;
                double z = state.boundary_start_z + (i + state.global_z_start + 1) * state.domain_size_z / (state.nz + 1);
                state.h_north_data[i*state.local_nx + k] = U(x,y,z);
            } else state.h_north_data[i*state.local_nx + k] = 0;
        }
    }

    // Initialize z-direction boundaries (bottom/top)
    for(int j = 0; j < state.local_ny; j++) {
        for(int k = 0; k < state.local_nx; k++) {
            if(state.coords[2] == 0) {
                double x = state.boundary_start_x + (k + state.global_x_start + 1) * state.domain_size_x / (state.nx + 1);
                double y = state.boundary_start_y + (j + state.global_y_start + 1) * state.domain_size_y / (state.ny + 1);
                double z = state.boundary_start_z;
                state.h_bottom_data[j*state.local_nx + k] = U(x,y,z);
            } else state.h_bottom_data[j*state.local_nx + k] = 0;
            
            if(state.coords[2] == state.dims[2]-1) {
                double x = state.boundary_start_x + (k + state.global_x_start + 1) * state.domain_size_x / (state.nx + 1);
                double y = state.boundary_start_y + (j + state.global_y_start + 1) * state.domain_size_y / (state.ny + 1);
                double z = state.boundary_end_z;
                state.h_top_data[j*state.local_nx + k] = U(x,y,z);
            } else state.h_top_data[j*state.local_nx + k] = 0;
        }
    }

    hipMemcpy(state.d_north_data, state.h_north_data, state.local_nx * state.local_nz * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(state.d_south_data, state.h_south_data, state.local_nx * state.local_nz * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(state.d_east_data, state.h_east_data, state.local_ny * state.local_nz * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(state.d_west_data, state.h_west_data, state.local_ny * state.local_nz * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(state.d_top_data, state.h_top_data, state.local_nx * state.local_ny * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(state.d_bottom_data, state.h_bottom_data, state.local_nx * state.local_ny * sizeof(double), hipMemcpyHostToDevice);
}

void red_black_update(SolverState &state) {
    // Red pass
    red_black_kernel<<<state.gridDim, state.blockDim>>>(
        state.d_u, state.d_rhs, 
        state.d_north_data, state.d_south_data,
        state.d_east_data, state.d_west_data, 
        state.d_top_data, state.d_bottom_data,
        state.local_nx, state.local_ny, state.local_nz, 
        state.a_x, state.a_y, state.a_z, state.a_f, 0);
    hipDeviceSynchronize();
    
    // Black pass
    red_black_kernel<<<state.gridDim, state.blockDim>>>(
        state.d_u, state.d_rhs, 
        state.d_north_data, state.d_south_data,
        state.d_east_data, state.d_west_data, 
        state.d_top_data, state.d_bottom_data,
        state.local_nx, state.local_ny, state.local_nz, 
        state.a_x, state.a_y, state.a_z, state.a_f, 1);
    hipDeviceSynchronize();
}

void halo_exchange(SolverState &state) {
    extract_boundaries<<<state.extractGridDim, state.extractBlockDim>>>(
        state.d_u, state.d_out_east, state.d_out_west,
        state.d_out_north, state.d_out_south,
        state.d_out_top, state.d_out_bottom,
        state.local_nx, state.local_ny, state.local_nz);
    
    hipMemcpy(state.h_out_east, state.d_out_east, state.local_ny * state.local_nz * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(state.h_out_west, state.d_out_west, state.local_ny * state.local_nz * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(state.h_out_north, state.d_out_north, state.local_nx * state.local_nz * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(state.h_out_south, state.d_out_south, state.local_nx * state.local_nz * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(state.h_out_top, state.d_out_top, state.local_nx * state.local_ny * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(state.h_out_bottom, state.d_out_bottom, state.local_nx * state.local_ny * sizeof(double), hipMemcpyDeviceToHost);
    
    MPI_Request reqs[12];
    int count = 0;

    if(state.coords[0] < state.dims[0]-1) MPI_Irecv(state.h_east_data, state.local_ny*state.local_nz, MPI_DOUBLE, state.east, 1, state.cart_comm, &reqs[count++]);
    if(state.coords[0] > 0) MPI_Irecv(state.h_west_data, state.local_ny*state.local_nz, MPI_DOUBLE, state.west, 2, state.cart_comm, &reqs[count++]);
    if(state.coords[1] < state.dims[1]-1) MPI_Irecv(state.h_north_data, state.local_nx*state.local_nz, MPI_DOUBLE, state.north, 3, state.cart_comm, &reqs[count++]);
    if(state.coords[1] > 0) MPI_Irecv(state.h_south_data, state.local_nx*state.local_nz, MPI_DOUBLE, state.south, 4, state.cart_comm, &reqs[count++]);
    if(state.coords[2] < state.dims[2]-1) MPI_Irecv(state.h_top_data, state.local_nx*state.local_ny, MPI_DOUBLE, state.top, 5, state.cart_comm, &reqs[count++]);
    if(state.coords[2] > 0) MPI_Irecv(state.h_bottom_data, state.local_nx*state.local_ny, MPI_DOUBLE, state.bottom, 6, state.cart_comm, &reqs[count++]);

    if(state.coords[0] > 0) MPI_Isend(state.h_out_west, state.local_ny*state.local_nz, MPI_DOUBLE, state.west, 1, state.cart_comm, &reqs[count++]);
    if(state.coords[0] < state.dims[0]-1) MPI_Isend(state.h_out_east, state.local_ny*state.local_nz, MPI_DOUBLE, state.east, 2, state.cart_comm, &reqs[count++]);
    if(state.coords[1] > 0) MPI_Isend(state.h_out_south, state.local_nx*state.local_nz, MPI_DOUBLE, state.south, 3, state.cart_comm, &reqs[count++]);
    if(state.coords[1] < state.dims[1]-1) MPI_Isend(state.h_out_north, state.local_nx*state.local_nz, MPI_DOUBLE, state.north, 4, state.cart_comm, &reqs[count++]);
    if(state.coords[2] > 0) MPI_Isend(state.h_out_bottom, state.local_nx*state.local_ny, MPI_DOUBLE, state.bottom, 5, state.cart_comm, &reqs[count++]);
    if(state.coords[2] < state.dims[2]-1) MPI_Isend(state.h_out_top, state.local_nx*state.local_ny, MPI_DOUBLE, state.top, 6, state.cart_comm, &reqs[count++]);

    MPI_Waitall(count, reqs, MPI_STATUSES_IGNORE);
    
    hipMemcpy(state.d_north_data, state.h_north_data, state.local_nx * state.local_nz * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(state.d_south_data, state.h_south_data, state.local_nx * state.local_nz * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(state.d_east_data, state.h_east_data, state.local_ny * state.local_nz * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(state.d_west_data, state.h_west_data, state.local_ny * state.local_nz * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(state.d_top_data, state.h_top_data, state.local_nx * state.local_ny * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(state.d_bottom_data, state.h_bottom_data, state.local_nx * state.local_ny * sizeof(double), hipMemcpyHostToDevice);
}

bool check_convergence(SolverState &state, int iter) {
    state.n_reductions++;
    
    if(state.rank == 0) clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Compute local residual
    double zero = 0.0;
    hipMemcpy(state.d_residual_sum, &zero, sizeof(double), hipMemcpyHostToDevice);
    
    size_t shared_mem_size = state.blockDim.x * state.blockDim.y * state.blockDim.z * sizeof(double);
    residual_kernel<<<state.gridDim, state.blockDim, shared_mem_size>>>(
        state.d_u, state.d_rhs, state.d_residual_sum,
        state.d_north_data, state.d_south_data, 
        state.d_east_data, state.d_west_data,
        state.d_top_data, state.d_bottom_data,
        state.local_nx, state.local_ny, state.local_nz, 
        state.delta_x, state.delta_y, state.delta_z);
    
    double local_residual;
    hipMemcpy(&local_residual, state.d_residual_sum, sizeof(double), hipMemcpyDeviceToHost);

    // Global reduction
    double global_residual = 0.0;
    MPI_Reduce(&local_residual, &global_residual, 1, MPI_DOUBLE, MPI_SUM, 0, state.cart_comm);

    int should_stop = 0;
    if (state.rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &end);
        state.avg_reduction_time += elapsed_time(start, end);
        
        double residual_norm = sqrt(global_residual / (state.nx * state.ny * state.nz));
        should_stop = residual_norm < state.convergence_bound;
        
        if(iter % (state.check_convergence_every_n * 10) == 0 || should_stop) {
            printf("Iteration %d: Residual norm = %.10e\n", iter, residual_norm);
        }
    }

    MPI_Bcast(&should_stop, 1, MPI_INT, 0, state.cart_comm);
    
    return should_stop;
}

void compare_solution(SolverState &state) {
    double zero = 0.0;
    hipMemcpy(state.d_mse_total, &zero, sizeof(double), hipMemcpyHostToDevice);
    
    size_t shared_mem_size = state.blockDim.x * state.blockDim.y * state.blockDim.z * sizeof(double);
    mse_kernel<<<state.gridDim, state.blockDim, shared_mem_size>>>(
        state.d_u, state.d_mse_total, 
        state.local_nx, state.local_ny, state.local_nz,
        state.global_x_start, state.global_y_start, state.global_z_start,
        state.nx, state.ny, state.nz,
        state.boundary_start_x, state.boundary_start_y, state.boundary_start_z,
        state.domain_size_x, state.domain_size_y, state.domain_size_z);
    
    double local_sq_error;
    hipMemcpy(&local_sq_error, state.d_mse_total, sizeof(double), hipMemcpyDeviceToHost);

    double global_mse = 0.0;
    MPI_Reduce(&local_sq_error, &global_mse, 1, MPI_DOUBLE, MPI_SUM, 0, state.cart_comm);

    if (state.rank == 0) {
        std::cout << "\n=== Final Results ===" << std::endl;
        std::cout << "Avg time per update: " << state.avg_update_time / state.n_iters << " s" << std::endl; 
        std::cout << "Avg time per reduction: " << state.avg_reduction_time / state.n_reductions << " s" << std::endl; 
        std::cout << "Total time to converge: " << elapsed_time(total_start, total_end) << " s" << std::endl; 
        std::cout << "Final MSE: " << global_mse/(state.nx * state.ny * state.nz) << std::endl;
        std::cout << "Iterations to converge: " << state.n_iters << std::endl;
    }
}

void write_halo_to_file(SolverState &state, const char* filename) {
    MPI_File fh;
    MPI_File_open(state.cart_comm, filename, 
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, 
                  MPI_INFO_NULL, &fh);
    
    if(state.rank == 0) {
        int header[3] = {state.nx, state.ny, state.nz};
        MPI_File_write(fh, header, 3, MPI_INT, MPI_STATUS_IGNORE);
        
        double bounds[6] = {state.boundary_start_x, state.boundary_start_y, state.boundary_start_z,
                           state.boundary_end_x, state.boundary_end_y, state.boundary_end_z};
        MPI_File_write(fh, bounds, 6, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
    
    MPI_Offset header_size = 3 * sizeof(int) + 6 * sizeof(double);

    double *arr = (double*) malloc(state.local_nx * sizeof(double));
    
    for(int z = 0; z < state.local_nz; z++) {
      for(int y = 0; y < state.local_ny; y++) {
        int global_z = z + state.global_z_start;
        int global_y = y + state.global_y_start;

        long long global_offset = global_z * state.nx * state.ny + global_y * state.nx + state.global_x_start;
        long long local_offset = z * state.local_nx * state.local_ny + y * state.local_nx;

        for(int i = 0; i < state.local_nx; i++) {
          if (z == 0 && state.coords[2] > 0) {
            arr[i] = 1.0;
            continue;
          }
          if (z == state.local_nz-1 && state.coords[2] < state.dims[2]-1) {
            arr[i] = 1.0;
            continue;
          }
          if (y == 0 && state.coords[1] > 0) {
            arr[i] = 1.0;
            continue;
          }
          if (y == state.local_ny-1 && state.coords[1] < state.dims[1]-1) {
            arr[i] = 1.0;
            continue;
          }
          if (i == 0 && state.coords[0] > 0) {
            arr[i] = 1.0;
            continue;
          }
          if (i == state.local_nx-1 && state.coords[0] < state.dims[0]-1) {
            arr[i] = 1.0;
            continue;
          }
          arr[i] = 0.0;
        }

        MPI_Offset file_offset = header_size + global_offset * sizeof(double);
        
        MPI_File_write_at(fh, file_offset, arr, state.local_nx, 
                          MPI_DOUBLE, MPI_STATUS_IGNORE);
      }
    }
    
    MPI_File_close(&fh);
}

void write_rank_to_file(SolverState &state, const char* filename) {
    MPI_File fh;
    MPI_File_open(state.cart_comm, filename, 
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, 
                  MPI_INFO_NULL, &fh);
    
    if(state.rank == 0) {
        int header[3] = {state.nx, state.ny, state.nz};
        MPI_File_write(fh, header, 3, MPI_INT, MPI_STATUS_IGNORE);
        
        double bounds[6] = {state.boundary_start_x, state.boundary_start_y, state.boundary_start_z,
                           state.boundary_end_x, state.boundary_end_y, state.boundary_end_z};
        MPI_File_write(fh, bounds, 6, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
    
    MPI_Offset header_size = 3 * sizeof(int) + 6 * sizeof(double);

    double *arr = (double*) malloc(state.local_nx * sizeof(double));
    for(int i = 0; i < state.local_nx; i++) arr[i] = (double) state.rank;
    
    for(int z = 0; z < state.local_nz; z++) {
      for(int y = 0; y < state.local_ny; y++) {
        int global_z = z + state.global_z_start;
        int global_y = y + state.global_y_start;

        long long global_offset = global_z * state.nx * state.ny + global_y * state.nx + state.global_x_start;
        long long local_offset = z * state.local_nx * state.local_ny + y * state.local_nx;

        MPI_Offset file_offset = header_size + global_offset * sizeof(double);
        
        MPI_File_write_at(fh, file_offset, arr, state.local_nx, 
                          MPI_DOUBLE, MPI_STATUS_IGNORE);
      }
    }
    free(arr);
    
    MPI_File_close(&fh);
}

void write_solution_to_file(SolverState &state, const char* filename) {
    int local_size = state.local_nx * state.local_ny * state.local_nz;

    hipMemcpy(state.h_u, state.d_u, local_size * sizeof(double), hipMemcpyDeviceToHost);

    
    MPI_File fh;
    MPI_File_open(state.cart_comm, filename, 
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, 
                  MPI_INFO_NULL, &fh);
    
    if(state.rank == 0) {
        int header[3] = {state.nx, state.ny, state.nz};
        MPI_File_write(fh, header, 3, MPI_INT, MPI_STATUS_IGNORE);
        
        double bounds[6] = {state.boundary_start_x, state.boundary_start_y, state.boundary_start_z,
                           state.boundary_end_x, state.boundary_end_y, state.boundary_end_z};
        MPI_File_write(fh, bounds, 6, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
    
    MPI_Offset header_size = 3 * sizeof(int) + 6 * sizeof(double);

    
    for(int z = 0; z < state.local_nz; z++) {
      for(int y = 0; y < state.local_ny; y++) {
        int global_z = z + state.global_z_start;
        int global_y = y + state.global_y_start;

        long long global_offset = global_z * state.nx * state.ny + global_y * state.nx + state.global_x_start;
        long long local_offset = z * state.local_nx * state.local_ny + y * state.local_nx;

        MPI_Offset file_offset = header_size + global_offset * sizeof(double);
        
        MPI_File_write_at(fh, file_offset, state.h_u + local_offset, state.local_nx, 
                          MPI_DOUBLE, MPI_STATUS_IGNORE);

      }
    }
    
    MPI_File_close(&fh);
}


std::string make_filename(const std::string& folder,
                          const std::string& basename,
                          int iter,
                          int max_iters)
{
    namespace fs = std::filesystem;

    fs::create_directories(folder);

    int digits = std::to_string(max_iters).size();

    std::string f = folder;
    if (!f.empty() && f.back() != '/' && f.back() != '\\')
        f += '/';

    std::ostringstream ss;
    ss << f << basename << "_" 
       << std::setw(digits) << std::setfill('0') << iter 
       << ".bin";

    return ss.str();
}

void solver(SolverState &state) {
    
    initialize_grid(state);
    initialize_boundaries(state);
    
    if(state.rank == 0) {
      std::cout << "\n=== Solver Initialized ===" << std::endl;
      std::cout << "Grid size: " << state.nx << "x" << state.ny << "x" << state.nz << std::endl;
      std::cout << "Domain: [" << state.boundary_start_x << ", " << state.boundary_end_x << "] x ["
                << state.boundary_start_y << ", " << state.boundary_end_y << "] x ["
                << state.boundary_start_z << ", " << state.boundary_end_z << "]" << std::endl;
      std::cout << "MPI processes: " << state.size << " (" << state.dims[0] << "x" 
                << state.dims[1] << "x" << state.dims[2] << ")" << std::endl;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &total_start);
    
    int iter = 0;
    for(; iter < state.max_iter; iter++) {
        state.n_iters++;
        
        if(state.rank == 0) clock_gettime(CLOCK_MONOTONIC, &start);
        
        red_black_update(state);
        halo_exchange(state);
        
        if(state.rank == 0) {
            clock_gettime(CLOCK_MONOTONIC, &end);
            state.avg_update_time += elapsed_time(start, end);
        }
        
        if(iter % state.check_convergence_every_n == 0 && iter != 0) {
            bool converged = check_convergence(state, iter);
            if(converged) {
                break;
            }
        }

        if(state.write_solution_every_n > 0 && iter % state.write_solution_every_n == 0 && state.write_solution) {
          write_solution_to_file(state, make_filename(state.output_dir, "solution", iter, state.max_iter).c_str());
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &total_end);
    
    compare_solution(state);

    if(state.write_solution) write_solution_to_file(state, make_filename(state.output_dir, "solution", iter, state.max_iter).c_str());
    //write_rank_to_file(state, "ranks.bin");
    //write_halo_to_file(state, "halo.bin");
}

void setup_mpi(SolverState &state) {
    MPI_Comm_size(MPI_COMM_WORLD, &state.size);
    MPI_Comm_rank(MPI_COMM_WORLD, &state.rank);

    state.dims[0] = 0; state.dims[1] = 0; state.dims[2] = 0;
    MPI_Dims_create(state.size, 3, state.dims);

    int periods[3] = {0, 0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 3, state.dims, periods, 1, &state.cart_comm);
    MPI_Cart_coords(state.cart_comm, state.rank, 3, state.coords);

    MPI_Cart_shift(state.cart_comm, 0, 1, &state.west, &state.east);
    MPI_Cart_shift(state.cart_comm, 1, 1, &state.south, &state.north);
    MPI_Cart_shift(state.cart_comm, 2, 1, &state.bottom, &state.top);

    state.local_nx = state.nx / state.dims[0];
    state.global_x_start = state.coords[0] * state.local_nx;
    int rem_x = state.nx - state.local_nx * state.dims[0];
    if(state.coords[0] < rem_x) state.local_nx++;
    state.global_x_start += std::min(rem_x, state.coords[0]);
    
    state.local_ny = state.ny / state.dims[1];
    state.global_y_start = state.coords[1] * state.local_ny;
    int rem_y = state.ny - state.local_ny * state.dims[1];
    if(state.coords[1] < rem_y) state.local_ny++;
    state.global_y_start += std::min(rem_y, state.coords[1]);
    
    state.local_nz = state.nz / state.dims[2];
    state.global_z_start = state.coords[2] * state.local_nz;
    int rem_z = state.nz - state.local_nz * state.dims[2];
    if(state.coords[2] < rem_z) state.local_nz++;
    state.global_z_start += std::min(rem_z, state.coords[2]);

    state.domain_size_x = state.boundary_end_x - state.boundary_start_x;
    state.domain_size_y = state.boundary_end_y - state.boundary_start_y;
    state.domain_size_z = state.boundary_end_z - state.boundary_start_z;

    state.delta_x = state.domain_size_x / (state.nx + 1);
    state.delta_y = state.domain_size_y / (state.ny + 1);
    state.delta_z = state.domain_size_z / (state.nz + 1);
    
    double dx2 = state.delta_x * state.delta_x;
    double dy2 = state.delta_y * state.delta_y;
    double dz2 = state.delta_z * state.delta_z;
    double denom = 2.0 * (1/dx2 + 1/dy2 + 1/dz2);
    
    state.a_x = (1/dx2) / denom;
    state.a_y = (1/dy2) / denom;
    state.a_z = (1/dz2) / denom;
    state.a_f = 1 / denom;
}

void setup_hip(SolverState &state) {
    int num_devices;
    hipGetDeviceCount(&num_devices);
    hipSetDevice(state.rank % num_devices);

    state.gridDim = dim3((state.local_nx + state.blockDim.x - 1) / state.blockDim.x,
                         (state.local_ny + state.blockDim.y - 1) / state.blockDim.y,
                         (state.local_nz + state.blockDim.z - 1) / state.blockDim.z);
    
    int max_boundary = std::max({state.local_nx*state.local_ny, 
                                  state.local_nx*state.local_nz, 
                                  state.local_ny*state.local_nz});
    state.extractGridDim = dim3((max_boundary + state.extractBlockDim.x - 1) / state.extractBlockDim.x);
}

void cleanup(SolverState &state) {
    free(state.h_rhs);
    free(state.h_u);
    free(state.h_north_data);
    free(state.h_south_data);
    free(state.h_east_data);
    free(state.h_west_data);
    free(state.h_top_data);
    free(state.h_bottom_data);
    free(state.h_out_east);
    free(state.h_out_west);
    free(state.h_out_north);
    free(state.h_out_south);
    free(state.h_out_top);
    free(state.h_out_bottom);

    hipFree(state.d_rhs);
    hipFree(state.d_u);
    hipFree(state.d_north_data);
    hipFree(state.d_south_data);
    hipFree(state.d_east_data);
    hipFree(state.d_west_data);
    hipFree(state.d_top_data);
    hipFree(state.d_bottom_data);
    hipFree(state.d_out_east);
    hipFree(state.d_out_west);
    hipFree(state.d_out_north);
    hipFree(state.d_out_south);
    hipFree(state.d_out_top);
    hipFree(state.d_out_bottom);
    hipFree(state.d_residual_sum);
    hipFree(state.d_mse_total);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    SolverState state;
    
    state.nx = 64;
    state.nz = 64;
    state.ny = 64;

    state.blockDim = dim3(32, 8, 4);
    state.extractBlockDim = dim3(256);
    
    state.boundary_start_x = 0.0;
    state.boundary_start_y = 0.0;
    state.boundary_start_z = 0.0;
    state.boundary_end_x = 1.0;
    state.boundary_end_y = 1.0;
    state.boundary_end_z = 2.0;
    
    state.max_iter = 500000;
    state.check_convergence_every_n = 1000;
    state.convergence_bound = 1e-6;
    
    state.avg_update_time = 0;
    state.avg_reduction_time = 0;
    state.n_iters = 0;
    state.n_reductions = 0;

    state.write_solution = true;
    state.write_solution_every_n = 100;
    state.output_dir = "output";

    setup_mpi(state);
    setup_hip(state);
    
    solver(state);
    
    cleanup(state);
    MPI_Finalize();

    return 0;
}
