./compile.sh
mkdir -p results/images
touch results/output.csv
echo "Kernel,file,attempt,algorithm,execution time (ms),throughput (MiB/s),speedup,correct" >> results/output.csv
./convolution "./resources/image21.pgm" "./results/images/image21" | tee -a ./results/output.csv
./convolution "./resources/lena_bw.pgm" "./results/images/lena_bw" | tee -a ./results/output.csv
./convolution "./resources/man.pgm" "./results/images/man" | tee -a ./results/output.csv
./convolution "./resources/mandrill.pgm" "./results/images/mandrill" | tee -a ./results/output.csv
./convolution "./resources/ref_rotated.pgm" "./results/images/ref_rotated" | tee -a ./results/output.csv