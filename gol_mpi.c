#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <math.h>

void displayFullGrid(int* local_grid, int local_m, int local_n, int m, int n, int rank, int numranks, int t);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); 

    int rank, size;  
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    
    int width = 6;
    int rows = size / width;
    int cols = width;

    int m = 6; 
    int n = 6;

    int local_m = (m / rows) + 2; 
    int local_n = (n / cols) + 2;

    int* local_grid = (int*)calloc(local_m * local_n, sizeof(int));
    int* buffer = (int*)calloc(local_m * local_n, sizeof(int));

    if (rank == 0) {
        local_grid[1 * local_n + 2] = 1; 
        local_grid[1 * local_n + 3] = 1; 
        local_grid[1 * local_n + 4] = 1; 
    }

    MPI_Bcast(local_grid, local_m * local_n, MPI_INT, 0, MPI_COMM_WORLD);

    displayFullGrid(local_grid, local_m, local_n, m, n, rank, size, 0);

    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank / cols, rank, &row_comm);

    MPI_Comm col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank % cols, rank, &col_comm);

    int north = (rank / cols == 0) ? MPI_PROC_NULL : rank - cols;
    int south = (rank / cols == rows - 1) ? MPI_PROC_NULL : rank + cols;
    int west = (rank % cols == 0) ? MPI_PROC_NULL : rank - 1;
    int east = (rank % cols == cols - 1) ? MPI_PROC_NULL : rank + 1;

    MPI_Status status;

    double start_time = MPI_Wtime();

    int gens = 5;
    for (int t = 1; t < gens; t++) {
        double halo_start = MPI_Wtime();

        MPI_Sendrecv(local_grid + local_n, local_n, MPI_INT, north, 0, 
                     local_grid + (local_m - 1) * local_n, local_n, MPI_INT, south, 0, 
                     MPI_COMM_WORLD, &status);
        MPI_Sendrecv(local_grid + (local_m - 2) * local_n, local_n, MPI_INT, south, 0, 
                     local_grid, local_n, MPI_INT, north, 0, 
                     MPI_COMM_WORLD, &status);

        for (int i = 1; i < local_m - 1; i++) {
            MPI_Sendrecv(local_grid + i * local_n + 1, 1, MPI_INT, west, 0, 
                         local_grid + i * local_n + (local_n - 1), 1, MPI_INT, east, 0, 
                         MPI_COMM_WORLD, &status);
            MPI_Sendrecv(local_grid + i * local_n + (local_n - 2), 1, MPI_INT, east, 0, 
                         local_grid + i * local_n, 1, MPI_INT, west, 0, 
                         MPI_COMM_WORLD, &status);
        }

        double halo_end = MPI_Wtime();
        double halo_time = halo_end - halo_start;

        for (int i = 1; i < local_m - 1; i++) {
            for (int j = 1; j < local_n - 1; j++) {
                int index = i * local_n + j;
                int counter = 0;
                int center = local_grid[index];
                
                for (int x = -1; x <= 1; x++) {
                    for (int y = -1; y <= 1; y++) {
                        if (x == 0 && y == 0) continue;
                        int nIndex = (i + x) * local_n + (j + y);
                        counter += local_grid[nIndex];
                    }
                }
                
                int temp = center;
                if (center == 1) {
                    if (counter < 2 || counter > 3) temp = 0;
                } else {
                    if (counter == 3) temp = 1;
                }
                
                buffer[index] = temp;
            }
        }

        int* tempgrid = local_grid;
        local_grid = buffer;
        buffer = tempgrid;

        if (rank == 0) {
            printf("Halo exchange time for generation %d: %f seconds\n", t, halo_time);
        }
    }

    double end_time = MPI_Wtime();
    double total_time = end_time - start_time;

    displayFullGrid(local_grid, local_m, local_n, m, n, rank, size, gens);

    if (rank == 0) {
        printf("Total simulation time: %f seconds\n", total_time);
    }

    free(local_grid);
    free(buffer);

    MPI_Finalize();

    return 0;
}

void displayFullGrid(int* local_grid, int local_m, int local_n, int m, int n, int rank, int numranks, int t) {
    int* full_grid = NULL;
    if (rank == 0) {
        full_grid = (int*)malloc(m * n * sizeof(int));
    }

    int* local_data = (int*)malloc((local_m - 2) * (local_n - 2) * sizeof(int));

    int index = 0;
    for (int i = 1; i < local_m - 1; i++) {
        for (int j = 1; j < local_n - 1; j++) {
            local_data[index++] = local_grid[i * local_n + j];
        }
    }

    int* sendcounts = NULL;
    int* displs = NULL;
    if (rank == 0) {
        sendcounts = (int*)malloc(numranks * sizeof(int));
        displs = (int*)malloc(numranks * sizeof(int));

        int rows_per_proc = (m - 2) / (int)sqrt(numranks);
        int cols_per_proc = (n - 2) / (int)sqrt(numranks);
        int proc_per_row = (int)sqrt(numranks);
        
        int k = 0;
        for (int i = 0; i < proc_per_row; i++) {
            for (int j = 0; j < proc_per_row; j++) {
                sendcounts[k] = rows_per_proc * cols_per_proc;
                displs[k] = (i * rows_per_proc * n) + (j * cols_per_proc);
                k++;
            }
        }
    }

    MPI_Gatherv(local_data, (local_m - 2) * (local_n - 2), MPI_INT,
                full_grid, sendcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Full grid at Iteration %d:\n", t);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                printf("%d ", full_grid[i * n + j]);
            }
            printf("\n");
        }
        printf("\n");
        free(full_grid);
        free(sendcounts);
        free(displs);
    }

    free(local_data);
}
