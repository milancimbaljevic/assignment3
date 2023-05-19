#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <CL/cl.h>
#include <math.h>

#define MAX_SOURCE_SIZE (0x100000)

void matrix_multipication(float* A, float* B, float* C, int M, int K, int N) {

	for (int m = 0; m < M; m++) {
		for (int n = 0; n < N; n++) {
			float acc = 0.0f;
			for (int k = 0; k < K; k++) {
				acc += A[m * K + k] * B[k * N + n];
			}
			C[m * N + n] = acc;
		}
	}
}

void compare_cpu_gpu(float* C, float* G, int size) {

	for (int i = 0; i < size; i++) {
		if (fabs(C[i] - G[i]) > 0.001) {
			printf("Buffers don't match: C[%d] = %f, G[%d] = %f!\n", i, C[i], i, G[i]);
			return;
		}
	}


	printf("Success buffers match!\n");
}

void randomInit(float* data, int size)
{
	for (unsigned int i = 0; i < size; ++i)
		data[i] = rand() / (float)RAND_MAX;
}

int main(int argc, int** argv) {
	//int m = stoi(argv[1]);
	//int n = m;
	//int k = m;
	//int num_of_rep = stoi(argv[2]);

	int m = 960;
	int n = m;
	int k = m;
	int num_of_rep = 30;

	cl_mem d_A;
	cl_mem d_B;
	cl_mem d_C;

	srand(time(NULL));

	unsigned int size_A = m * k;
	unsigned int size_B = k * n;
	unsigned int size_C = m * n;

	float* h_A = (float*)malloc(sizeof(float) * size_A);
	float* h_B = (float*)malloc(sizeof(float) * size_B);
	float* h_C = (float*)malloc(sizeof(float) * size_C);
	float* h_C_GPU = (float*)malloc(sizeof(float) * size_C);

	randomInit(h_A, size_A);
	randomInit(h_B, size_B);

	cl_platform_id platform_id;
	cl_device_id device_id;
	cl_uint num_platforms, num_devices;
	char vendor_name[1024], device_name[1024];

	// Get the platform ID
	clGetPlatformIDs(1, &platform_id, &num_platforms);

	// Get the device ID
	clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);

	// Get the device name
	clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), &device_name, NULL);
	printf("Device Name: %s\n", device_name);

	// Get the vendor name
	clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(vendor_name), &vendor_name, NULL);
	printf("Vendor Name: %s\n", vendor_name);

	// Get the number of compute units for the device
	cl_uint num_compute_units;
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(num_compute_units), &num_compute_units, NULL);
	printf("Number of compute units: %u\n", num_compute_units);

	// Get the global memory size for the device
	cl_ulong global_mem_size;
	clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, NULL);
	printf("Global memory size: %llu mb\n", global_mem_size / 1024 / 1204);

	// Get the local memory size for the device
	cl_ulong local_mem_size;
	clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, NULL);
	printf("Local memory size: %llu kb\n", local_mem_size / 1024);

	// Get max wokgroup size
	size_t max_workgroup_size;
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_workgroup_size), &max_workgroup_size, NULL);
	printf("Maximum workgroup size: %lu\n", max_workgroup_size);


	// Create a compute context 
	int err;
	cl_context context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context)
	{
		printf("Error: Failed to create a compute context!\n");
		return EXIT_FAILURE;
	}

	// Create a command queue
	cl_command_queue commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
	if (!commands)
	{
		printf("Error: Failed to create a command commands!\n");
		return EXIT_FAILURE;
	}


	// Load the kernel source code into the array source_str
	FILE* fp;
	char* source_str;
	size_t source_size;

	fp = fopen("kernel.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);
	printf("Kernel loading done\n");

	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &err);
	if (!program)
	{
		printf("Error: Failed to create compute program!\n");
		return EXIT_FAILURE;
	}

	// Build the program executable
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		size_t len;
		char buffer[2048];
		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer),
			buffer, &len);

		printf("%s\n", buffer);
		exit(1);
	}

	// Create the compute kernel in the program we wish to run
	//
	cl_kernel kernel = clCreateKernel(program, "matrix_multiplication", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	// Create the input and output arrays in device memory for our calculation
	d_C = clCreateBuffer(context, CL_MEM_READ_WRITE, size_C * sizeof(float), NULL, &err);
	d_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_A * sizeof(float), h_A,
		&err);
	d_B = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_B * sizeof(float), h_B,
		&err);

	if (!d_A || !d_B || !d_C)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}

	int tile_size = 64;
	int work_group_x = 16;
	int work_group_y = 16;

	//int m_prim = m % tile_size != 0 ? m + (tile_size - m % tile_size) : m;
	//int n_prim = n % tile_size != 0 ? n + (tile_size - n % tile_size) : n;

	printf("m: %d\n", m);
	printf("n: %d\n", n);

	
	// int num_of_work = ceil((m_prim * n_prim) / (double)num_compute_units / (double)(work_group_x * work_group_y));
	int num_of_blocks_per_unit = ceil((m * n) / (double)(tile_size * tile_size) / (double)num_compute_units);
	int number_of_work_per_thread_per_block = (tile_size * tile_size) / (work_group_x * work_group_y);
	int num_of_blocks_x = m / tile_size;
	int num_of_blocks_y = n / tile_size;

	//Launch OpenCL kernel
	size_t globalWorkSize[2] = { work_group_x * num_compute_units, work_group_y };
	size_t localWorkSize[2] = { work_group_x, work_group_y };

	printf("NDRange: { %d, %d } \n", globalWorkSize[0], globalWorkSize[1]);
	printf("Num of blocks per unit: %d \n", num_of_blocks_per_unit);
	printf("Num of work per thread per block: %d \n", number_of_work_per_thread_per_block);
	printf("Num of work: %d \n", number_of_work_per_thread_per_block * num_of_blocks_per_unit);

	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&d_A);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&d_B);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&d_C);
	err = clSetKernelArg(kernel, 3, sizeof(int), (void*)&m);
	err = clSetKernelArg(kernel, 4, sizeof(int), (void*)&n);
	err = clSetKernelArg(kernel, 5, sizeof(int), (void*)&k);
	err = clSetKernelArg(kernel, 6, sizeof(int), (void*)&num_of_blocks_per_unit);
	err = clSetKernelArg(kernel, 7, sizeof(int), (void*)&number_of_work_per_thread_per_block);
	err = clSetKernelArg(kernel, 8, sizeof(int), (void*)&num_of_blocks_x);
	err = clSetKernelArg(kernel, 9, sizeof(int), (void*)&num_of_blocks_y);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	double total_time = 0;
	double val[1000];
	int i = num_of_rep;

	while (i--) {
		cl_event ev;

		err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize,
			0, NULL, &ev);

		if (err != CL_SUCCESS)
		{
			printf("Error: Failed to execute kernel! %d\n", err);
			exit(1);
		}

		clWaitForEvents(1, &ev);

		cl_ulong time_start;
		cl_ulong time_end;

		clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

		double nano_seconds = time_end - time_start;
		total_time += nano_seconds;

		val[i] = nano_seconds / 1000000.0;
	}

	printf("OpenCl average execution time is: %lf milliseconds \n", (total_time / (double)num_of_rep) / 1000000.0);

	// Calculate standard deviation

	double average_nano = total_time / (double)num_of_rep / 1000000.0;

	double dev = 0.0;
	for (int i = 0; i < num_of_rep; i++) {
		dev += pow((val[i] - average_nano), 2);
	}

	dev /= (double)num_of_rep - 1;
	dev = sqrt(dev);

	printf("Standard deviation for %d is %lf \n", m, dev);

	////Retrieve result from device
	err = clEnqueueReadBuffer(commands, d_C, CL_TRUE, 0, size_C * sizeof(float), (void*)h_C_GPU, NULL, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	clock_t begin = clock();
	matrix_multipication(h_A, h_B, h_C, m, k, n);
	clock_t end = clock();
	double time_spent = (double)(end - begin);
	printf("CPU time: %f milliseconds\n", (double)(time_spent) / CLOCKS_PER_SEC * 1000.0);

	compare_cpu_gpu(h_C, h_C_GPU, m * n);


	clReleaseContext(context);
	return 0;
}