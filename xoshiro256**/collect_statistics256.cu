// nvcc -arch=sm_86 collect_statistics256_v3.cu -o collect_statistics256_v3 --ptxas-options=-v
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <iomanip>

// Константы
const size_t SEQUENCE_LENGTH = 65536;   // Длинна последовательности
const size_t MAX_SHIFT = 10;            // Максимальный сдвиг для автокорреляции
const size_t BLOCK_SIZE = 384;          // Количество потоков в блоке
const size_t BLOCK_ROWS = 2048;         // Число генераторов в одном блоке
const int    NUM_BINS = 256;            // Количество интервалов для гистограммы
const size_t NUM_CLUSTERS = 32;          // Число кластеров для кластерного анализа

// Функция вращения влево
__device__ __host__ unsigned long long int rotl(const unsigned long long int x, int k) {
    return (x << k) | (x >> (64 - k));
}

// Генератор xoshiro256**
__device__ unsigned long long int xoshiro256_next(unsigned long long int* state) {
    const unsigned long long int result = rotl(state[1] * 5, 7) * 9;

    const unsigned long long int t = state[1] << 17;

    state[2] ^= state[0];
    state[3] ^= state[1];
    state[1] ^= state[2];
    state[0] ^= state[3];

    state[2] ^= t;

    state[3] = rotl(state[3], 45);

    return result;
}

// Функция для чтения начальных состояний из CSV
std::vector<uint64_t> readCSV(const std::string& filename) {
    std::vector<uint64_t> states;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл: " + filename);
    }

    std::string line;
    getline(file, line); // Пропускаем заголовок
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string s0_str, s1_str, s2_str, s3_str;

        getline(ss, s0_str, ','); // init_s0
        getline(ss, s1_str, ','); // init_s1
        getline(ss, s2_str, ','); // init_s2
        getline(ss, s3_str, ','); // init_s3

        states.push_back(std::stoull(s0_str, nullptr, 16));
        states.push_back(std::stoull(s1_str, nullptr, 16));
        states.push_back(std::stoull(s2_str, nullptr, 16));
        states.push_back(std::stoull(s3_str, nullptr, 16));
    }

    return states;
}

__global__ void calculateBitFrequencies_Entropy(unsigned long long int* states, 
    double* bitFrequencies, double* bitEntropy, size_t rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows) return;

    unsigned long long int local_state[4];
    for (int i = 0; i < 4; ++i) local_state[i] = states[idx * 4 + i];
    
    unsigned long long int local_bitFreq[64] = {0};
    double local_bitEntropy[64] = {0.0};

    for (int i = 0; i < SEQUENCE_LENGTH; ++i) {
        unsigned long long int res = xoshiro256_next(local_state);
        for (int b = 0; b < 64; ++b) {
            local_bitFreq[b] += (res >> b) & 1;
        }
    }

    for (int b = 0; b < 64; ++b) {
        double p1 = (double)local_bitFreq[b] / SEQUENCE_LENGTH;
        double p0 = 1.0 - p1;
        if (p1 > 0) local_bitEntropy[b] -= p1 * log2(p1);
        if (p0 > 0) local_bitEntropy[b] -= p0 * log2(p0);
    }

    for (int b = 0; b < 64; ++b) {
        bitFrequencies[idx * 64 + b] = (double)local_bitFreq[b] / SEQUENCE_LENGTH;
        bitEntropy[idx * 64 + b] = local_bitEntropy[b];
    }
}

__global__ void calculatePairTripleQuadFrequencies(unsigned long long int* states, double* pairFrequencies, double* tripleFrequencies, double* quadFrequencies, size_t rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows) return;

    unsigned long long int local_state[4];
    for (int i = 0; i < 4; ++i) local_state[i] = states[idx * 4 + i];
    unsigned long long int local_pairFreq[4] = {0};
    unsigned long long int local_tripleFreq[8] = {0};
    unsigned long long int local_quadFreq[16] = {0};

    for (int i = 0; i < SEQUENCE_LENGTH; ++i) {
        unsigned long long int res = xoshiro256_next(local_state);

        for (int b = 0; b < 63; ++b) {
            int pair = ((res >> b) & 1) | (((res >> (b + 1)) & 1) << 1);
            local_pairFreq[pair]++;
        }

        for (int b = 0; b < 62; ++b) {
            int triple = ((res >> b) & 1) | (((res >> (b + 1)) & 1) << 1) | (((res >> (b + 2)) & 1) << 2);
            local_tripleFreq[triple]++;
        }

        for (int b = 0; b < 61; ++b) {
            int quad = ((res >> b) & 1) | (((res >> (b + 1)) & 1) << 1) |
                       (((res >> (b + 2)) & 1) << 2) | (((res >> (b + 3)) & 1) << 3);
            local_quadFreq[quad]++;
        }
    }

    for (int p = 0; p < 4; ++p) {
        pairFrequencies[idx * 4 + p] = (double)local_pairFreq[p] / (SEQUENCE_LENGTH * 63);
    }

    for (int t = 0; t < 8; ++t) {
        tripleFrequencies[idx * 8 + t] = (double)local_tripleFreq[t] / (SEQUENCE_LENGTH * 62);
    }

    for (int q = 0; q < 16; ++q) {
        quadFrequencies[idx * 16 + q] = (double)local_quadFreq[q] / (SEQUENCE_LENGTH * 61);
    }
}

__global__ void calculateUniformity(unsigned long long int* states, double* uniformity, unsigned long long int* results, size_t rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows) return;

    unsigned long long int local_state[4];
    for (int i = 0; i < 4; ++i) local_state[i] = states[idx * 4 + i];
    double local_uniformity = 0;
    unsigned long long int result = 0;

    for (int i = 0; i < SEQUENCE_LENGTH; ++i) {
        unsigned long long int res = xoshiro256_next(local_state);
        local_uniformity += (double)(res % 256) / 256.0;
        result ^= res;
    }

    uniformity[idx] = local_uniformity / SEQUENCE_LENGTH;
    results[idx] = result;
}

__global__ void calculateCorrelation(unsigned long long int* states, double* correlations, size_t rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows) return;

    unsigned long long int local_state[4];
    for (int i = 0; i < 4; ++i) local_state[i] = states[idx * 4 + i];

    double local_corr[4] = {0.0}, mean_s[4] = {0.0}, mean_res = 0.0;
    double cov_s_res[4] = {0.0}, var_s[4] = {0.0}, var_res = 0.0;

    for (int i = 0; i < SEQUENCE_LENGTH; ++i){
        unsigned long long int res = xoshiro256_next(local_state);

        mean_res += (double)res;
        for (int j = 0; j < 4; ++j) {
            mean_s[j] += (double)local_state[j];
        }
    }

    for (int i = 0; i < 4; ++i) mean_s[i] /= SEQUENCE_LENGTH;
    mean_res /= SEQUENCE_LENGTH;

    for (int i = 0; i < 4; ++i) local_state[i] = states[idx * 4 + i];
    for (int i = 0; i < SEQUENCE_LENGTH; ++i){
        double diff_s[4];
        for (int j = 0; j < 4; ++j) diff_s[j] = (double)local_state[j] - mean_s[j];

        unsigned long long int res = xoshiro256_next(local_state);
        double diff_res = (double)res - mean_res;

        for (int j = 0; j < 4; ++j) {
            cov_s_res[j] += diff_s[j] * diff_res;
            var_s[j] += diff_s[j] * diff_s[j];
        }
        var_res += diff_res * diff_res;
    }

    for (int i = 0; i < 4; ++i) {
        cov_s_res[i] /= SEQUENCE_LENGTH;
        var_s[i] /= SEQUENCE_LENGTH;
    }
    var_res /= SEQUENCE_LENGTH;

    for (int i = 0; i < 4; ++i){
        local_corr[i] = cov_s_res[i] / (sqrt(var_s[i]) * sqrt(var_res));
    }

    for (int i = 0; i < 4; ++i){
        correlations[idx * 4 + i] = local_corr[i];
    }
}

__global__ void calculateNIST(unsigned long long int* states, 
    double* globalFrequency, double* runsTest, double* blockFrequency, 
    size_t numBlocksPerSequence, size_t rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows) return;

    unsigned long long int local_state[4];
    for (int i = 0; i < 4; ++i) local_state[i] = states[idx * 4 + i];

    // Локальные переменные для тестов NIST
    unsigned long long int global_one_count = 0;
    unsigned long long int block_one_count = 0;
    unsigned long long int run_count = 0;
    unsigned long long int current_run_length = 0;

    // --- Block Frequency Test
    for (size_t blockIdx = 0; blockIdx < numBlocksPerSequence; ++blockIdx) {
        block_one_count = 0;
        for (size_t i = 0; i < SEQUENCE_LENGTH / numBlocksPerSequence; ++i) {
            unsigned long long int res = xoshiro256_next(local_state);
            
            block_one_count += __popcll(res); // Подсчет единичных битов в блоке
        }
        blockFrequency[idx * numBlocksPerSequence + blockIdx] = (double)block_one_count / (64 * (SEQUENCE_LENGTH / numBlocksPerSequence));
    }

    for (int i = 0; i < 4; ++i) local_state[i] = states[idx * 4 + i];
    for (int i = 0; i < SEQUENCE_LENGTH; ++i){
        unsigned long long int res = xoshiro256_next(local_state);
        // --- Global Frequency Test
        global_one_count += __popcll(res); // Подсчет единиц на глобальном уровне

        // --- Runs Test
        for (int b = 0; b < 64; ++b) {
            unsigned long long int current_bit = (res >> b) & 1;
            if (b == 0 || current_bit == ((res >> (b - 1)) & 1)) {
                ++current_run_length;
            } else {
                ++run_count;
                current_run_length = 1;
            }
        }
        if (current_run_length > 0) ++run_count;
    }
    globalFrequency[idx] = (double)global_one_count / (64 * SEQUENCE_LENGTH);
    runsTest[idx] = (double)run_count / (64 * SEQUENCE_LENGTH);
}

__global__ void calculateMutualInformation(unsigned long long int* states, double* mutualInformation_s_s, double* mutualInformation_res_s, size_t rows){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows) return;

    unsigned long long int local_state[4];
    for (int i = 0; i < 4; ++i) local_state[i] = states[idx * 4 + i];

    // Локальные массивы для подсчёта частот (Взаимная информация)
    unsigned long long int bitCount_s[4][2] = {{0}};
    unsigned long long int bitCount_res[2] = {0};
    unsigned long long int jointCount_s_res[4][2][2] = {{{0}}};
    unsigned long long int jointCount_s_s[4][4][2][2] = {{{{0}}}};


    for (int i = 0; i < SEQUENCE_LENGTH; ++i) {
        unsigned long long int res = xoshiro256_next(local_state);
        // Подсчёт взаимной информации
        for (int b = 0; b < 64; ++b) {
            int bit_res = (res >> b) & 1;

            for (int j1 = 0; j1 < 4; ++j1) {
                int bit_s1 = (local_state[j1] >> b) & 1;
                bitCount_s[j1][bit_s1]++;
                jointCount_s_res[j1][bit_s1][bit_res]++;

                for (int j2 = j1 + 1; j2 < 4; ++j2) {
                    int bit_s2 = (local_state[j2] >> b) & 1;
                    jointCount_s_s[j1][j2][bit_s1][bit_s2]++;
                }
            }
            bitCount_res[bit_res]++;
        }
    }

    // Расчёт взаимной информации между состояниями и результатом
    for (int j1 = 0; j1 < 4; ++j1) { // Цикл по первому состоянию
        for (int j2 = j1 + 1; j2 < 4; ++j2) { // Цикл по второму состоянию (j2 > j1)
            double local_MI_states = 0.0;

            for (int x = 0; x < 2; ++x) {
                for (int y = 0; y < 2; ++y) {
                    double p_joint_states = (double)jointCount_s_s[j1][j2][x][y] / (SEQUENCE_LENGTH * 64);
                    double p_s1 = (double)bitCount_s[j1][x] / (SEQUENCE_LENGTH * 64);
                    double p_s2 = (double)bitCount_s[j2][y] / (SEQUENCE_LENGTH * 64);
                    
                    if (p_joint_states > 0) {
                        local_MI_states += p_joint_states * log2(p_joint_states / (p_s1 * p_s2));
                    }
                }
            }

            // Сохранение результата
            mutualInformation_s_s[idx * 6 + (j1 * 3 + j2 - 1)] = local_MI_states;//max(local_MI_states, 0.0); // Индексирование массива MI
        }

        // Расчёт взаимной информации между состоянием и результатом
        double local_MI_res_state = 0.0;
        for (int x = 0; x < 2; ++x) {
            for (int y = 0; y < 2; ++y) {
                double p_joint_res = (double)jointCount_s_res[j1][x][y] / (SEQUENCE_LENGTH * 64);
                double p_state = (double)bitCount_s[j1][x] / (SEQUENCE_LENGTH * 64);
                double p_res = (double)bitCount_res[y] / (SEQUENCE_LENGTH * 64);

                if (p_joint_res > 0) {
                    local_MI_res_state += p_joint_res * log2(p_joint_res / (p_state * p_res));
                }
            }
        }

        // Сохранение результата
        mutualInformation_res_s[idx * 4 + j1] = max(local_MI_res_state, 0.0);
    }
}

__global__ void calculateAutocorr(unsigned long long int* states, double* autocorrelations, size_t rows){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows) return;

    unsigned long long int local_state[4];
    for (int i = 0; i < 4; ++i) local_state[i] = states[idx * 4 + i];

    // --- Автокорреляция ---
    double mean = 0.0;
    double variance = 0.0;
    double autocorr[MAX_SHIFT] = {0.0};

    // Подсчёт среднего и дисперсии
    for (int i = 0; i < SEQUENCE_LENGTH; ++i) {
        unsigned long long int current_value = xoshiro256_next(local_state);
        mean += (double)current_value;
        variance += (double)current_value * (double)current_value;
    }
    mean /= SEQUENCE_LENGTH;
    variance = variance / SEQUENCE_LENGTH - mean * mean;

    // Подсчёт автокорреляции
    unsigned long long int local_state_j[4];
    for (int shift = 0; shift < MAX_SHIFT; ++shift) {
        // Сброс начального состояния
        for (int i = 0; i < 4; ++i) {
            local_state[i] = states[idx * 4 + i];
            local_state_j[i] = states[idx * 4 + i];
        }

        // Прокручиваем генератор для состояния j на `shift` шагов
        for (int s = 0; s < shift; ++s) {
            xoshiro256_next(local_state_j);
        }

        double autocorr_sum = 0.0;
        for (int i = 0; i < SEQUENCE_LENGTH - shift; ++i) {
            // Генерация значений
            unsigned long long int value_i = xoshiro256_next(local_state);
            unsigned long long int value_j = xoshiro256_next(local_state_j);

            // Вычисление автокорреляции
            autocorr_sum += ((double)value_i - mean) * ((double)value_j - mean);
        }
        autocorr[shift] = autocorr_sum / ((SEQUENCE_LENGTH - shift) * variance);
    }

    // Сохранение автокорреляции
    for (int shift = 0; shift < MAX_SHIFT; ++shift) {
        autocorrelations[idx * MAX_SHIFT + shift] = autocorr[shift];
    }
}

__global__ void calculateDiff(unsigned long long int* states, double* diffHistogram, size_t rows){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows) return;

    unsigned long long int local_state[4];
    for (int i = 0; i < 4; ++i) local_state[i] = states[idx * 4 + i];

    // --- Гистограмма разностей ---
    const double MAX_DIFF = pow(2, 64) - 1;         // Максимальное значение разности 2^64-1
    const double MIN_DIFF = MAX_DIFF * (-1);        // Минимальное значение разности -2^64+1
    const double BIN_WIDTH = (MAX_DIFF - MIN_DIFF) / NUM_BINS;

    unsigned long long int prev_value = rotl(local_state[1] * 5, 7) * 9; // Первый результат генератора

    unsigned int localHistogram[NUM_BINS] = {0};
    for (int i = 1; i < SEQUENCE_LENGTH; ++i) {
        unsigned long long int current_value = xoshiro256_next(local_state);
        double diff = (double)current_value - (double)prev_value;
        //printf("MIN_DIFF diff MAX_DIFF\t\t%lf %lf %lf\n", MIN_DIFF, diff, MAX_DIFF);
        // Индексация интервала
        if (diff >= MIN_DIFF && diff <= MAX_DIFF) {
            int bin_index = (int)((diff - MIN_DIFF) / BIN_WIDTH);
            bin_index = min(NUM_BINS - 1, max(0, bin_index)); // Защита от выхода за пределы
            localHistogram[bin_index]++;
        }

        prev_value = current_value;
    }

    // Нормализация гистограммы
    for (int bin = 0; bin < NUM_BINS; ++bin) {
        diffHistogram[idx * NUM_BINS + bin] = (double)localHistogram[bin] / (SEQUENCE_LENGTH - 1);
    }

}

__global__ void calculateCluster(unsigned long long int* states, double* clusters, double* clusterCounts_save, size_t rows){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows) return;

    unsigned long long int local_state[4];
    const double MAX_DIFF = pow(2, 64) - 1;         // Максимальное значение result 2^64-1
    const double MIN_DIFF = 0;                      // Минимальное значение result  0

    double centroids[NUM_CLUSTERS];
    for (int k = 0; k < NUM_CLUSTERS; ++k) {
        centroids[k] = MIN_DIFF + k * (MAX_DIFF - MIN_DIFF) / (NUM_CLUSTERS - 1);
    }

    int clusterCounts[NUM_CLUSTERS] = {0};
    double clusterSums[NUM_CLUSTERS] = {0.0};

    for (int iter = 0; iter < 10; ++iter) {
        // Сброс кластеров
        for (int k = 0; k < NUM_CLUSTERS; ++k) {
            clusterCounts[k] = 0;
            clusterSums[k] = 0.0;
        }

        // Назначение кластеров
        for (int i = 0; i < 4; ++i) local_state[i] = states[idx * 4 + i];
        for (int i = 0; i < SEQUENCE_LENGTH; ++i) {
            unsigned long long int current_value = xoshiro256_next(local_state);
            double minDist = fabs((double)current_value - centroids[0]);
            int bestCluster = 0;

            for (int k = 1; k < NUM_CLUSTERS; ++k) {
                double dist = fabs((double)current_value - centroids[k]);
                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = k;
                }
            }

            clusterCounts[bestCluster]++;
            clusterSums[bestCluster] += (double)current_value;
        }

        // Обновление центроидов
        for (int k = 0; k < NUM_CLUSTERS; ++k) {
            if (clusterCounts[k] > 0) {
                centroids[k] = clusterSums[k] / clusterCounts[k];
            }
        }
    }

    // Сохранение результатов кластеров
    for (int k = 0; k < NUM_CLUSTERS; ++k) {
        clusters[idx * NUM_CLUSTERS + k] = centroids[k];
        clusterCounts_save[idx * NUM_CLUSTERS + k] = clusterCounts[k];
    }
}


int main() {
    try {
        // Чтение начальных состояний из CSV
        std::vector<uint64_t> initialStates = readCSV("dependency_table_xoshiro256_0.csv");
        size_t totalRows = initialStates.size() / 4;

        // Открытие файла для записи результатов
        std::ofstream resultFile("./data/analysis_results256.csv");
        resultFile << "Block,Row,Result";
        for (int c = 0; c < 4; ++c) resultFile << ",Correlation_s_" << c;
        resultFile << ",Uniformity";
        for (int b = 0; b < 64; ++b) resultFile << ",Bit_" << b;
        for (int p = 0; p < 4; ++p) resultFile << ",Pair_" << p;
        for (int t = 0; t < 8; ++t) resultFile << ",Triple_" << t;
        for (int q = 0; q < 16; ++q) resultFile << ",Quad_" << q;
        for (int e = 0; e < 64; ++e) resultFile << ",Entropy_" << e;
        resultFile << ",GlobalFrequency,RunsTest";
        size_t numBlocksPerSequence = 16; // Количество блоков для Block Frequency Test
        for (size_t i = 0; i < numBlocksPerSequence; ++i) resultFile << ",BlockFrequency_" << i;
        for (size_t i = 0; i < 4; ++i){
            for (size_t j = i + 1; j < 4; ++j){
                resultFile << ",MutualInformation_s" << i << "_s" << j;
            }
        }
        for (size_t i = 0; i < 4; ++i) resultFile << ",MutualInformation_res_s" << i;
        for (int shift = 0; shift < MAX_SHIFT; ++shift) resultFile << ",Autocorr_" << shift;
        for (int i = 0; i < NUM_BINS; ++i) resultFile << ",Difference_" << i;
        for (int k = 0; k < NUM_CLUSTERS; ++k) resultFile << ",Cluster_" << k;
        for (int k = 0; k < NUM_CLUSTERS; ++k) resultFile << ",ClusterCount_" << k;
        resultFile << "\n";

        // Блоками обрабатываем данные
        for (size_t offset = 0; offset < totalRows; offset += BLOCK_ROWS) {
            size_t currentRows = std::min(BLOCK_ROWS, totalRows - offset);

            unsigned long long int* d_states;
            unsigned long long int* d_results;
            double* d_correlations;
            double* d_uniformity;
            double* d_bitFrequencies;
            double* d_pairFrequencies;
            double* d_tripleFrequencies;
            double* d_quadFrequencies;
            double* d_bitEntropy;
            double* d_globalFrequency;
            double* d_runsTest;
            double* d_blockFrequency;
            double *d_mutualInformation_s_s, *d_mutualInformation_res_s;
            double* d_autocorrelations;
            double* d_diffHistogram;
            double* d_clusters;
            double* d_clusterCount;

            cudaMalloc(&d_states, currentRows * 4 * sizeof(unsigned long long int*));
            cudaMalloc(&d_results, currentRows * sizeof(unsigned long long int*));
            cudaMalloc(&d_correlations, currentRows * 4 * sizeof(double));
            cudaMalloc(&d_uniformity, currentRows * sizeof(double));
            cudaMalloc(&d_bitFrequencies, currentRows * 64 * sizeof(double));
            cudaMalloc(&d_pairFrequencies, currentRows * 4 * sizeof(double));
            cudaMalloc(&d_tripleFrequencies, currentRows * 8 * sizeof(double));
            cudaMalloc(&d_quadFrequencies, currentRows * 16 * sizeof(double));
            cudaMalloc(&d_bitEntropy, currentRows * 64 * sizeof(double));
            cudaMalloc(&d_globalFrequency, currentRows * sizeof(double));
            cudaMalloc(&d_runsTest, currentRows * sizeof(double));
            cudaMalloc(&d_blockFrequency, currentRows * numBlocksPerSequence * sizeof(double));
            cudaMalloc(&d_mutualInformation_s_s, 6 * currentRows * sizeof(double));
            cudaMalloc(&d_mutualInformation_res_s, 4 * currentRows * sizeof(double));
            cudaMalloc(&d_autocorrelations, currentRows * MAX_SHIFT * sizeof(double));
            cudaMalloc(&d_diffHistogram, currentRows * NUM_BINS * sizeof(double));
            cudaMalloc(&d_clusters, currentRows * NUM_CLUSTERS * sizeof(double));
            cudaMalloc(&d_clusterCount, currentRows * NUM_CLUSTERS * sizeof(double));

            cudaMemcpy(d_states, &initialStates[offset * 4], currentRows * 4 * sizeof(unsigned long long int*), cudaMemcpyHostToDevice);

            /*int minGridSize, blockSize;
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculateBitFrequencies_Entropy, 0, 0);
            std::cout << "Рекомендуемый calculateBitFrequencies_Entropy BLOCK_SIZE: " << blockSize << std::endl;
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculatePairTripleQuadFrequencies, 0, 0);
            std::cout << "Рекомендуемый calculatePairTripleQuadFrequencies BLOCK_SIZE: " << blockSize << std::endl;
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculateUniformity, 0, 0);
            std::cout << "Рекомендуемый calculateUniformity BLOCK_SIZE: " << blockSize << std::endl;
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculateCorrelation, 0, 0);
            std::cout << "Рекомендуемый calculateCorrelation BLOCK_SIZE: " << blockSize << std::endl;
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculateNIST, 0, 0);
            std::cout << "Рекомендуемый calculateNIST BLOCK_SIZE: " << blockSize << std::endl;
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculateMutualInformation, 0, 0);
            std::cout << "Рекомендуемый calculateMutualInformation BLOCK_SIZE: " << blockSize << std::endl;
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculateAutocorr, 0, 0);
            std::cout << "Рекомендуемый calculateAutocorr BLOCK_SIZE: " << blockSize << std::endl;
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculateDiff, 0, 0);
            std::cout << "Рекомендуемый calculateDiff BLOCK_SIZE: " << blockSize << std::endl;
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculateCluster, 0, 0);
            std::cout << "Рекомендуемый calculateCluster BLOCK_SIZE: " << blockSize << std::endl;*/

            // Запуск CUDA ядер
            size_t numBlocks = (currentRows + BLOCK_SIZE - 1) / BLOCK_SIZE;
            calculateBitFrequencies_Entropy<<<numBlocks, BLOCK_SIZE>>>(d_states, d_bitFrequencies, d_bitEntropy, currentRows);
            calculatePairTripleQuadFrequencies<<<numBlocks, BLOCK_SIZE>>>(d_states, d_pairFrequencies, d_tripleFrequencies, d_quadFrequencies, currentRows);
            calculateUniformity<<<numBlocks, BLOCK_SIZE>>>(d_states, d_uniformity, d_results, currentRows);
            calculateCorrelation<<<numBlocks, BLOCK_SIZE>>>(d_states, d_correlations, currentRows);
            calculateNIST<<<numBlocks, BLOCK_SIZE>>>(d_states, d_globalFrequency, d_runsTest, d_blockFrequency, numBlocksPerSequence, currentRows);
            calculateMutualInformation<<<numBlocks, BLOCK_SIZE>>>(d_states, d_mutualInformation_s_s, d_mutualInformation_res_s, currentRows);
            calculateAutocorr<<<numBlocks, BLOCK_SIZE>>>(d_states, d_autocorrelations, currentRows);
            calculateDiff<<<numBlocks, BLOCK_SIZE>>>(d_states, d_diffHistogram, currentRows);
            calculateCluster<<<numBlocks, BLOCK_SIZE>>>(d_states, d_clusters, d_clusterCount, currentRows);

            // Копирование результатов на хост
            std::vector<unsigned long long int*> results(currentRows);
            std::vector<double> correlations(currentRows * 4);
            std::vector<double> uniformity(currentRows);
            std::vector<double> bitFrequencies(currentRows * 64);
            std::vector<double> pairFrequencies(currentRows * 4);
            std::vector<double> tripleFrequencies(currentRows * 8);
            std::vector<double> quadFrequencies(currentRows * 16);
            std::vector<double> bitEntropy(currentRows * 64);
            std::vector<double> globalFrequency(currentRows);
            std::vector<double> runsTest(currentRows);
            std::vector<double> blockFrequency(currentRows * numBlocksPerSequence);
            std::vector<double> mutualInformation_s_s(currentRows * 6);
            std::vector<double> mutualInformation_res_s(currentRows * 4);
            std::vector<double> autocorrelations(currentRows * MAX_SHIFT);
            std::vector<double> diffHistogram(currentRows * NUM_BINS);
            std::vector<double> clusters(currentRows * NUM_CLUSTERS);
            std::vector<double> clusterCounts(currentRows * NUM_CLUSTERS);

            cudaMemcpy(results.data(), d_results, currentRows * sizeof(unsigned long long int*), cudaMemcpyDeviceToHost);
            cudaMemcpy(correlations.data(), d_correlations, currentRows * 4 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(uniformity.data(), d_uniformity, currentRows * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(bitFrequencies.data(), d_bitFrequencies, currentRows * 64 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(pairFrequencies.data(), d_pairFrequencies, currentRows * 4 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(tripleFrequencies.data(), d_tripleFrequencies, currentRows * 8 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(quadFrequencies.data(), d_quadFrequencies, currentRows * 16 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(bitEntropy.data(), d_bitEntropy, currentRows * 64 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(globalFrequency.data(), d_globalFrequency, currentRows * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(runsTest.data(), d_runsTest, currentRows * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(blockFrequency.data(), d_blockFrequency, currentRows * numBlocksPerSequence * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(mutualInformation_s_s.data(), d_mutualInformation_s_s, currentRows * 6 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(mutualInformation_res_s.data(), d_mutualInformation_res_s, currentRows * 4 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(autocorrelations.data(), d_autocorrelations, currentRows * MAX_SHIFT * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(diffHistogram.data(), d_diffHistogram, currentRows * NUM_BINS * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(clusters.data(), d_clusters, currentRows * NUM_CLUSTERS * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(clusterCounts.data(), d_clusterCount, currentRows * NUM_CLUSTERS * sizeof(double), cudaMemcpyDeviceToHost);


            // Запись результатов текущего блока в CSV
            for (size_t i = 0; i < currentRows; ++i) {
                resultFile << (offset / BLOCK_ROWS) << "," << i << "," << std::hex << results[i] << std::dec << std::fixed << std::setprecision(6);
                for (int c = 0; c < 4; ++c) resultFile << "," << correlations[i * 4 + c];
                resultFile << "," << uniformity[i];
                for (int b = 0; b < 64; ++b) resultFile << "," << bitFrequencies[i * 64 + b];
                for (int p = 0; p < 4; ++p) resultFile << "," << pairFrequencies[i * 4 + p];
                for (int t = 0; t < 8; ++t) resultFile << "," << tripleFrequencies[i * 8 + t];
                for (int q = 0; q < 16; ++q) resultFile << "," << quadFrequencies[i * 16 + q];
                for (int e = 0; e < 64; ++e) resultFile << "," << bitEntropy[i * 64 + e];
                resultFile << "," << globalFrequency[i]
                           << "," << runsTest[i];
                for (int b = 0; b < numBlocksPerSequence; ++b) resultFile << "," << blockFrequency[i * numBlocksPerSequence + b];
                for (int mi_ss = 0; mi_ss < 6; ++mi_ss) resultFile << "," << mutualInformation_s_s[i * 6 + mi_ss];
                for (int mi_res_s = 0; mi_res_s < 4; ++mi_res_s) resultFile << "," << mutualInformation_res_s[i * 4 + mi_res_s];
                for (int shift = 0; shift < MAX_SHIFT; ++shift) resultFile << "," << autocorrelations[i * MAX_SHIFT + shift];
                resultFile << std::fixed << std::setprecision(10);
                for (int n = 0; n < NUM_BINS; ++n) resultFile << "," << diffHistogram[i * NUM_BINS + n];
                resultFile << std::fixed << std::setprecision(1);
                for (int c = 0; c < NUM_CLUSTERS; ++c) resultFile << "," << clusters[i * NUM_CLUSTERS + c];
                for (int c = 0; c < NUM_CLUSTERS; ++c) resultFile << "," << clusterCounts[i * NUM_CLUSTERS + c];
                resultFile << "\n";
            }

            cudaFree(d_states);
            cudaFree(d_results);
            cudaFree(d_correlations);
            cudaFree(d_uniformity);
            cudaFree(d_bitFrequencies);
            cudaFree(d_pairFrequencies);
            cudaFree(d_tripleFrequencies);
            cudaFree(d_quadFrequencies);
            cudaFree(d_bitEntropy);
            cudaFree(d_globalFrequency);
            cudaFree(d_runsTest);
            cudaFree(d_blockFrequency);
            cudaFree(d_mutualInformation_s_s);
            cudaFree(d_mutualInformation_res_s);
            cudaFree(d_autocorrelations);
            cudaFree(d_diffHistogram);
            cudaFree(d_clusters);
            cudaFree(d_clusterCount);
        }

        resultFile.close();
        std::cout << "Анализ завершён, результаты сохранены в './data/analysis_results256.csv'." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}