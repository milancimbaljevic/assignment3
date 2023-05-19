#define tile_size 64
#define workgroup_x 16
#define workgroup_y 16
#define nowpt ((tile_size * tile_size) / (workgroup_x * workgroup_y))

__kernel void matrix_multiplication(__global const float* A, __global const float* B, __global float* C, int m, int n, int k, int num_of_blocks_per_unit, int number_of_work_per_thread_per_block, int num_of_blocks_x, int num_of_blocks_y) {

	const int lx = get_local_id(0);
	const int ly = get_local_id(1);

	__local float A_sub[tile_size][tile_size];
	__local float B_sub[tile_size][tile_size];


	const int unit_id = get_group_id(0);

	const int thread_number = ly * workgroup_x + lx;
	const int tile_offset = thread_number * number_of_work_per_thread_per_block;


	int block_counter = 0;

	float acc[nowpt];


	while (block_counter < num_of_blocks_per_unit) {
		int target_block_number = unit_id * num_of_blocks_per_unit + block_counter; // all threads within the same workgroup have the sam target_block_number
		int target_block_x = target_block_number % num_of_blocks_x;
		int target_block_y = target_block_number / num_of_blocks_x;
		int total_number_of_blocks = num_of_blocks_x * num_of_blocks_y;

		if (target_block_number < total_number_of_blocks) {

			for (int w = 0; w < number_of_work_per_thread_per_block; w++) {
				acc[w] = 0.0f;
			}

			int thread_work_counter = 0;

			// each thread handles number_of_work_per_thread_per_block threads in the target_block
			// caclulate postion of elements that need to be loaded into sub-matrices
			int element_local_y = tile_offset / tile_size;
			int element_global_y = target_block_y * tile_size + element_local_y;

			for (int current_tile = 0; current_tile < k / tile_size; current_tile++) {
				for (int current_work = 0; current_work < number_of_work_per_thread_per_block; current_work++) {
					int element_local_x = (tile_offset + current_work) % tile_size;
					int element_global_x = target_block_x * tile_size + element_local_x;

					A_sub[element_local_y][element_local_x] = A[element_global_y * k + current_tile * tile_size + element_local_x];
					B_sub[element_local_y][element_local_x] = B[(current_tile * tile_size + element_local_y) * n + element_global_x];

				}

				barrier(CLK_LOCAL_MEM_FENCE);
				
				for (int i = 0; i < tile_size; i++) {
					float x = A_sub[tile_offset / tile_size][i];
					int element_local_y = tile_offset / tile_size;

					for (int t = 0; t < number_of_work_per_thread_per_block; t++) {
						int element_local_x = (tile_offset + t) % tile_size;
						acc[t] += x * B_sub[i][element_local_x];
					}

				}

				barrier(CLK_LOCAL_MEM_FENCE);
			}


			for (int t = 0; t < number_of_work_per_thread_per_block; t++) {
				int element_local_x = (tile_offset + t) % tile_size;
				int element_global_x = target_block_x * tile_size + element_local_x;

				C[element_global_y * n + element_global_x] = acc[t];
			}
		}
		else break;


		block_counter++;
	}
}











