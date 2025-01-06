//компиляция: nvcc -arch=sm_86 collect_statistics128.cu -o collect_statistics128 --ptxas-options=-v
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
#include "xoroshiro128+.h"

// Константы
const size_t SEQUENCE_LENGTH = 65536; // Длина последовательности
const size_t MAX_SHIFT = 10;         // Максимальный сдвиг для автокорреляции
const size_t BLOCK_SIZE = 384;       // Количество потоков в блоке
const size_t BLOCK_ROWS = 2048;      // Число генераторов в одном блоке обработки
const size_t NUM_CLUSTERS = 10;       // Число кластеров для кластерного анализа
const int NUM_BINS = 256; // Количество интервалов для гистограммы


// Функция для чтения CSV-файла
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
        std::string s0_str, s1_str;

        getline(ss, s0_str, ','); // init_s0
        getline(ss, s1_str, ','); // init_s1

        uint64_t s0 = std::stoull(s0_str, nullptr, 16);
        uint64_t s1 = std::stoull(s1_str, nullptr, 16);

        states.push_back(s0);
        states.push_back(s1);
    }

    return states;
}

__device__ void calc_corr(
    unsigned long long int s0, unsigned long long int s1,
    double mean_s0, double mean_s1, double mean_res,
    double* local_corr
) {
    mean_s0 /= SEQUENCE_LENGTH;
    mean_s1 /= SEQUENCE_LENGTH;
    mean_res /= SEQUENCE_LENGTH;

    // Вычисление ковариаций и дисперсий
    double cov_s0_res = 0.0, cov_s1_res = 0.0;
    double var_s0 = 0.0, var_s1 = 0.0, var_res = 0.0;

    for (int i = 0; i < SEQUENCE_LENGTH; ++i) {
        unsigned long long int res = s0 + s1;
        double diff_s0 = (double)s0 - mean_s0;
        double diff_s1 = (double)s1 - mean_s1;
        double diff_res = (double)res - mean_res;

        cov_s0_res += diff_s0 * diff_res;
        cov_s1_res += diff_s1 * diff_res;

        var_s0 += diff_s0 * diff_s0;
        var_s1 += diff_s1 * diff_s1;
        var_res += diff_res * diff_res;

        s1 ^= s0;
        s0 = ((s0 << 24) | (s0 >> (64 - 24))) ^ s1 ^ (s1 << 16);
        s1 = (s1 << 37) | (s1 >> (64 - 37));
    }

    cov_s0_res /= SEQUENCE_LENGTH;
    cov_s1_res /= SEQUENCE_LENGTH;
    var_s0 /= SEQUENCE_LENGTH;
    var_s1 /= SEQUENCE_LENGTH;
    var_res /= SEQUENCE_LENGTH;

    // Вычисление корреляции
    local_corr[0] = cov_s0_res / (sqrt(var_s0) * sqrt(var_res));
    local_corr[1] = cov_s1_res / (sqrt(var_s1) * sqrt(var_res));
}



// CUDA ядро: Генерация последовательностей и полный анализ
__global__ void processGenerator1(
    unsigned long long int* states, unsigned long long int* results, size_t rows,
    double* bitFrequencies, double* pairFrequencies,
    double* tripleFrequencies, double* quadFrequencies,
    double* correlations, double* uniformity, double* bitEntropy,
    double* globalFrequency, double* blockFrequency, double* runsTest,
    size_t numBlocksPerSequence,
    double* mutualInformation_s0_s1, double* mutualInformation_s0_res, double* mutualInformation_s1_res
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < rows) {
        unsigned long long int s0 = states[idx * 2];
        unsigned long long int s1 = states[idx * 2 + 1];
        unsigned long long int local_result = 0;

        // Локальные массивы для частот
        unsigned long long int local_bitFreq[64] = {0};
        unsigned long long int local_pairFreq[4] = {0};
        unsigned long long int local_tripleFreq[8] = {0};
        unsigned long long int local_quadFreq[16] = {0};
        double local_bitEntropy[64] = {0.0};
        double local_corr[2] = {0.0}, mean_s0 = 0.0, mean_s1 = 0.0, mean_res = 0.0; // для корреляции
        double local_uniformity = 0;

        // Локальные массивы для подсчёта частот (Взаимная информация)
        unsigned long long int bitCount_s0[2] = {0, 0}; // Частоты битов в s0
        unsigned long long int bitCount_s1[2] = {0, 0}; // Частоты битов в s1
        unsigned long long int bitCount_res[2] = {0, 0}; // Частоты битов в res
        unsigned long long int jointCount_s0_s1[2][2] = {{0, 0}, {0, 0}}; // Совместные частоты s0 и s1
        unsigned long long int jointCount_s0_res[2][2] = {{0, 0}, {0, 0}}; // Совместные частоты s0 и res
        unsigned long long int jointCount_s1_res[2][2] = {{0, 0}, {0, 0}}; // Совместные частоты s1 и res


        // Локальные переменные для тестов NIST
        unsigned long long int global_one_count = 0;
        unsigned long long int block_one_count = 0;
        unsigned long long int run_count = 0;
        unsigned long long int current_run_length = 0;

        // Block Frequency Test
        for (size_t blockIdx = 0; blockIdx < numBlocksPerSequence; ++blockIdx) {
            block_one_count = 0;
            for (size_t i = 0; i < SEQUENCE_LENGTH / numBlocksPerSequence; ++i) {
                unsigned long long int res = s0 + s1;
                s1 ^= s0;
                s0 = ((s0 << 24) | (s0 >> (64 - 24))) ^ s1 ^ (s1 << 16);
                s1 = (s1 << 37) | (s1 >> (64 - 37));
                
                block_one_count += __popcll(res); // Подсчет единичных битов в блоке
            }
            blockFrequency[idx * numBlocksPerSequence + blockIdx] = (double)block_one_count / (64 * (SEQUENCE_LENGTH / numBlocksPerSequence));
        }

        s0 = states[idx * 2];
        s1 = states[idx * 2 + 1];
        for (int i = 0; i < SEQUENCE_LENGTH; ++i) {
            unsigned long long int res = s0 + s1;
            local_result ^= res;

            // Подсчет частот
            for (int b = 0; b < 64; ++b) {
                local_bitFreq[b] += (res >> b) & 1;
            }
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

            mean_s0 += (double)s0;
            mean_s1 += (double)s1;
            mean_res += (double)res;

            local_uniformity += (double)(res % 256) / 256.0;

            // Global Frequency Test
            global_one_count += __popcll(res); // Подсчет единиц на глобальном уровне
            // Runs Test
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

            //Подсчёт частот для Взаимной информации
            for (int b = 0; b < 64; ++b) {
                int bit_s0 = (s0 >> b) & 1;
                int bit_s1 = (s1 >> b) & 1;
                int bit_res = (res >> b) & 1;

                // Частоты битов
                bitCount_s0[bit_s0]++;
                bitCount_s1[bit_s1]++;
                bitCount_res[bit_res]++;

                // Совместные частоты
                jointCount_s0_s1[bit_s0][bit_s1]++;
                jointCount_s0_res[bit_s0][bit_res]++;
                jointCount_s1_res[bit_s1][bit_res]++;
            }

            s1 ^= s0;
            s0 = ((s0 << 24) | (s0 >> (64 - 24))) ^ s1 ^ (s1 << 16);
            s1 = (s1 << 37) | (s1 >> (64 - 37));
        }

        // Вычисление энтропии бит
        for (int b = 0; b < 64; ++b) {
            double p1 = (double)local_bitFreq[b] / SEQUENCE_LENGTH;
            double p0 = 1.0 - p1;
            if (p1 > 0) local_bitEntropy[b] -= p1 * log2(p1);
            if (p0 > 0) local_bitEntropy[b] -= p0 * log2(p0);
        }

        for (int b = 0; b < 64; ++b) {
            bitEntropy[idx * 64 + b] = local_bitEntropy[b];
        }

        // Запись вероятностей
        for (int b = 0; b < 64; ++b) {
            double freq = (double)local_bitFreq[b] / SEQUENCE_LENGTH;
            bitFrequencies[idx * 64 + b] = freq;
        } 

        for (int p = 0; p < 4; ++p) {
            double freq = (double)local_pairFreq[p] / (SEQUENCE_LENGTH * 63);
            pairFrequencies[idx * 4 + p] = freq;
        }

        for (int t = 0; t < 8; ++t) {
            double freq = (double)local_tripleFreq[t] / (SEQUENCE_LENGTH * 62);
            tripleFrequencies[idx * 8 + t] = freq;
        }

        for (int q = 0; q < 16; ++q) {
            double freq = (double)local_quadFreq[q] / (SEQUENCE_LENGTH * 61);
            quadFrequencies[idx * 16 + q] = freq;
        }

        // Вычисление корреляции
        s0 = states[idx * 2];
        s1 = states[idx * 2 + 1];
        calc_corr(s0, s1, mean_s0, mean_s1, mean_res, local_corr);

        // Рассчёт взаимной информации
        double local_MI_s0_s1 = 0.0, local_MI_s0_res = 0.0, local_MI_s1_res = 0.0;
        for (int x = 0; x < 2; ++x) {
            for (int y = 0; y < 2; ++y) {
                double p_s0_s1 = (double)jointCount_s0_s1[x][y] / (SEQUENCE_LENGTH * 64);
                double p_s0 = (double)bitCount_s0[x] / (SEQUENCE_LENGTH * 64);
                double p_s1 = (double)bitCount_s1[y] / (SEQUENCE_LENGTH * 64);

                if (p_s0_s1 > 0) {
                    local_MI_s0_s1 += p_s0_s1 * log2(p_s0_s1 / (p_s0 * p_s1));
                }

                double p_s0_res = (double)jointCount_s0_res[x][y] / (SEQUENCE_LENGTH * 64);
                double p_res = (double)bitCount_res[y] / (SEQUENCE_LENGTH * 64);

                if (p_s0_res > 0) {
                    local_MI_s0_res += p_s0_res * log2(p_s0_res / (p_s0 * p_res));
                }

                double p_s1_res = (double)jointCount_s1_res[x][y] / (SEQUENCE_LENGTH * 64);

                if (p_s1_res > 0) {
                    local_MI_s1_res += p_s1_res * log2(p_s1_res / (p_s1 * p_res));
                }
            }
        }
        mutualInformation_s0_s1[idx] = max(local_MI_s0_s1, 0.0);
        mutualInformation_s0_res[idx] = max(local_MI_s0_res, 0.0);
        mutualInformation_s1_res[idx] = max(local_MI_s1_res, 0.0);

        results[idx] = local_result;
        correlations[idx * 2] = local_corr[0];
        correlations[idx * 2 + 1] = local_corr[1];
        uniformity[idx] = local_uniformity / SEQUENCE_LENGTH;

        // Запись результатов NIST
        globalFrequency[idx] = (double)global_one_count / (64 * SEQUENCE_LENGTH);
        runsTest[idx] = (double)run_count / (64 * SEQUENCE_LENGTH);
    }
}

__global__ void processGenerator2(
    unsigned long long int* states, size_t rows,
    double* autocorrelations, double* diffHistogram, double* clusters, double* clusterCounts_save
) {
    const double MAX_DIFF = pow(2, 64) - 1;         // Максимальное значение разности 2^64-1
    const double MIN_DIFF = MAX_DIFF * (-1);        // Минимальное значение разности -2^64+1
    const double BIN_WIDTH = (MAX_DIFF - MIN_DIFF) / NUM_BINS;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < rows) {
        unsigned long long int s0 = states[idx * 2];
        unsigned long long int s1 = states[idx * 2 + 1];

        // --- Автокорреляция ---
        double mean = 0.0;
        double variance = 0.0;
        double autocorr[MAX_SHIFT] = {0.0};
        unsigned long long int prev_value = 0;
        unsigned long long int first_value = 0;

        // Подсчёт среднего и дисперсии
        for (int i = 0; i < SEQUENCE_LENGTH; ++i) {
            unsigned long long int current_value = s0 + s1;
            if (i == 0) first_value = current_value; // Для кластерного анализа
            mean += (double)current_value;
            variance += (double)current_value * (double)current_value;

            // Обновление генератора
            s1 ^= s0;
            s0 = ((s0 << 24) | (s0 >> (64 - 24))) ^ s1 ^ (s1 << 16);
            s1 = (s1 << 37) | (s1 >> (64 - 37));
        }
        mean /= SEQUENCE_LENGTH;
        variance = variance / SEQUENCE_LENGTH - mean * mean;

        // Подсчёт автокорреляции
        for (int shift = 0; shift < MAX_SHIFT; ++shift) {
            s0 = states[idx * 2];
            s1 = states[idx * 2 + 1];
            unsigned long long int s0_j = states[idx * 2];
            unsigned long long int s1_j = states[idx * 2 + 1];

            for (int s = 0; s < shift; ++s){
                // Генерация следующего значения
                s1_j ^= s0_j;
                s0_j = ((s0_j << 24) | (s0_j >> (64 - 24))) ^ s1_j ^ (s1_j << 16);
                s1_j = (s1_j << 37) | (s1_j >> (64 - 37));
            }

            double autocorr_sum = 0.0;
            for (int i = 0; i < SEQUENCE_LENGTH - shift; ++i) {
                unsigned long long int value_i = s0 + s1;
                unsigned long long int value_j = s0_j + s1_j;
                if (i >= shift) {
                    autocorr_sum += ((double)value_i - mean) * ((double)value_j - mean);
                }

                // Генерация следующего значения
                s1 ^= s0;
                s0 = ((s0 << 24) | (s0 >> (64 - 24))) ^ s1 ^ (s1 << 16);
                s1 = (s1 << 37) | (s1 >> (64 - 37));

                s1_j ^= s0_j;
                s0_j = ((s0_j << 24) | (s0_j >> (64 - 24))) ^ s1_j ^ (s1_j << 16);
                s1_j = (s1_j << 37) | (s1_j >> (64 - 37));
            }
            autocorr[shift] = autocorr_sum / ((SEQUENCE_LENGTH - shift) * variance);
        }

        // --- Гистограмма разностей ---
        s0 = states[idx * 2];
        s1 = states[idx * 2 + 1];
        prev_value = first_value;

        unsigned int localHistogram[NUM_BINS] = {0};
        for (int i = 1; i < SEQUENCE_LENGTH; ++i) {
            unsigned long long int current_value = s0 + s1;
            double diff = (double)current_value - (double)prev_value;
            //printf("MIN_DIFF diff MAX_DIFF\t\t%lf %lf %lf\n", MIN_DIFF, diff, MAX_DIFF);
            // Индексация интервала
            if (diff >= MIN_DIFF && diff <= MAX_DIFF) {
                int bin_index = (int)((diff - MIN_DIFF) / BIN_WIDTH);
                bin_index = min(NUM_BINS - 1, max(0, bin_index)); // Защита от выхода за пределы
                localHistogram[bin_index]++;
            }

            prev_value = current_value;

            // Обновление генератора
            s1 ^= s0;
            s0 = ((s0 << 24) | (s0 >> (64 - 24))) ^ s1 ^ (s1 << 16);
            s1 = (s1 << 37) | (s1 >> (64 - 37));
        }

        // Нормализация гистограммы
        for (int bin = 0; bin < NUM_BINS; ++bin) {
            diffHistogram[idx * NUM_BINS + bin] = (double)localHistogram[bin] / (SEQUENCE_LENGTH - 1);
        }

        // --- Кластерный анализ ---
        double centroids[NUM_CLUSTERS];
        for (int k = 0; k < NUM_CLUSTERS; ++k) {
            centroids[k] = 0 + k * (MAX_DIFF - 0) / (NUM_CLUSTERS - 1);
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
            s0 = states[idx * 2];
            s1 = states[idx * 2 + 1];
            for (int i = 0; i < SEQUENCE_LENGTH; ++i) {
                unsigned long long int current_value = s0 + s1;
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

                // Обновление генератора
                s1 ^= s0;
                s0 = ((s0 << 24) | (s0 >> (64 - 24))) ^ s1 ^ (s1 << 16);
                s1 = (s1 << 37) | (s1 >> (64 - 37));
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

        // Сохранение автокорреляции
        for (int shift = 0; shift < MAX_SHIFT; ++shift) {
            autocorrelations[idx * MAX_SHIFT + shift] = autocorr[shift];
        }
    }
}



int main() {
    try {
        // Чтение начальных состояний из CSV
        std::vector<uint64_t> initialStates = readCSV("dependency_table_xoroshiro128_0.csv");
        size_t totalRows = initialStates.size() / 2;

        // Открытие файла для записи результатов
        std::ofstream resultFile("./data/analysis_results128.csv");
        resultFile << "Block,Row,Result,Correlation_s0,Correlation_s1,Uniformity";
        for (int b = 0; b < 64; ++b) resultFile << ",Bit_" << b;
        for (int p = 0; p < 4; ++p) resultFile << ",Pair_" << p;
        for (int t = 0; t < 8; ++t) resultFile << ",Triple_" << t;
        for (int q = 0; q < 16; ++q) resultFile << ",Quad_" << q;
        for (int b = 0; b < 64; ++b) resultFile << ",Entropy_" << b;
        resultFile << ",GlobalFrequency,RunsTest";
        size_t numBlocksPerSequence = 16; // Количество блоков для Block Frequency Test
        for (size_t i = 0; i < numBlocksPerSequence; ++i) resultFile << ",BlockFrequency_" << i;
        resultFile << ",MutualInformation_s0_s1,MutualInformation_s0_res,MutualInformation_s1_res";
        for (int shift = 0; shift < MAX_SHIFT; ++shift) resultFile << ",Autocorr_" << shift;
        for (int i = 0; i < NUM_BINS; ++i) resultFile << ",Difference_" << i;
        for (int k = 0; k < NUM_CLUSTERS; ++k) resultFile << ",Cluster_" << k;
        for (int k = 0; k < NUM_CLUSTERS; ++k) resultFile << ",ClusterCount_" << k;
        resultFile << "\n";

        // Блоками обрабатываем данные
        for (size_t offset = 0; offset < totalRows; offset += BLOCK_ROWS) {
            size_t currentRows = std::min(BLOCK_ROWS, totalRows - offset);
        
            // Заменяем uint64_t на unsigned long long int
            unsigned long long int* states = reinterpret_cast<unsigned long long int*>(&initialStates[offset * 2]);
        
            // Выделение памяти для текущего блока
            unsigned long long int *d_states, *d_results;
            double *d_bitFrequencies, *d_pairFrequencies, *d_tripleFrequencies, *d_quadFrequencies;
            double *d_correlations, *d_uniformity, *d_bitEntropy;
            double *d_globalFrequency, *d_blockFrequency, *d_runsTest;
            double *d_mutualInformation_s0_s1, *d_mutualInformation_s0_res, *d_mutualInformation_s1_res;
            double *d_autocorrelations, *d_diffHistogram, *d_clusters, *d_clusterCount;
        
            cudaMalloc(&d_states, currentRows * 2 * sizeof(unsigned long long int));
            cudaMalloc(&d_results, currentRows * sizeof(unsigned long long int));
            cudaMalloc(&d_bitFrequencies, currentRows * 64 * sizeof(double));
            cudaMalloc(&d_pairFrequencies, currentRows * 4 * sizeof(double));
            cudaMalloc(&d_tripleFrequencies, currentRows * 8 * sizeof(double));
            cudaMalloc(&d_quadFrequencies, currentRows * 16 * sizeof(double));
            cudaMalloc(&d_correlations, currentRows * 2 * sizeof(double));
            cudaMalloc(&d_uniformity, currentRows * sizeof(double));
            cudaMalloc(&d_bitEntropy, currentRows * 64 * sizeof(double));
            cudaMalloc(&d_globalFrequency, currentRows * sizeof(double));
            cudaMalloc(&d_blockFrequency, currentRows * numBlocksPerSequence * sizeof(double));
            cudaMalloc(&d_runsTest, currentRows * sizeof(double));
            cudaMalloc(&d_mutualInformation_s0_s1, currentRows * sizeof(double));
            cudaMalloc(&d_mutualInformation_s0_res, currentRows * sizeof(double));
            cudaMalloc(&d_mutualInformation_s1_res, currentRows * sizeof(double));
            cudaMalloc(&d_autocorrelations, currentRows * MAX_SHIFT * sizeof(double));
            cudaMalloc(&d_diffHistogram, currentRows * NUM_BINS * sizeof(double));
            cudaMalloc(&d_clusters, currentRows * NUM_CLUSTERS * sizeof(double));
            cudaMalloc(&d_clusterCount, currentRows * NUM_CLUSTERS * sizeof(double));
        
            cudaMemset(d_bitFrequencies, 0, currentRows * 64 * sizeof(double));
            cudaMemset(d_pairFrequencies, 0, currentRows * 4 * sizeof(double));
            cudaMemset(d_tripleFrequencies, 0, currentRows * 8 * sizeof(double));
            cudaMemset(d_quadFrequencies, 0, currentRows * 16 * sizeof(double));
            cudaMemset(d_bitEntropy, 0, currentRows * 64 * sizeof(double));
            cudaMemset(d_blockFrequency, 0, currentRows * numBlocksPerSequence * sizeof(double));
            cudaMemset(d_diffHistogram, 0, currentRows * NUM_BINS * sizeof(double));
        
            cudaMemcpy(d_states, states, currentRows * 2 * sizeof(unsigned long long int), cudaMemcpyHostToDevice);

            // Запуск CUDA ядра
            size_t numBlocks = (currentRows + BLOCK_SIZE - 1) / BLOCK_SIZE;
            processGenerator1<<<numBlocks, BLOCK_SIZE>>>(
                d_states, d_results, currentRows,
                d_bitFrequencies, d_pairFrequencies, d_tripleFrequencies, d_quadFrequencies,
                d_correlations, d_uniformity, d_bitEntropy,
                d_globalFrequency, d_blockFrequency, d_runsTest,
                numBlocksPerSequence,
                d_mutualInformation_s0_s1, d_mutualInformation_s0_res, d_mutualInformation_s1_res
            );
            processGenerator2<<<numBlocks, BLOCK_SIZE>>>(
                d_states, currentRows,
                d_autocorrelations, d_diffHistogram, d_clusters, d_clusterCount
            );

            
            // Копирование результатов на CPU
            unsigned long long int results[BLOCK_ROWS];
            double bitFrequencies[BLOCK_ROWS * 64];
            double pairFrequencies[BLOCK_ROWS * 4];
            double tripleFrequencies[BLOCK_ROWS * 8];
            double quadFrequencies[BLOCK_ROWS * 16];

            double correlations[BLOCK_ROWS * 2];
            double uniformity[BLOCK_ROWS];
            double bitEntropy[BLOCK_ROWS * 64];

            double globalFrequency[BLOCK_ROWS];
            double blockFrequency[BLOCK_ROWS * numBlocksPerSequence];
            double runsTest[BLOCK_ROWS];

            double mutualInformation_s0_s1[BLOCK_ROWS], 
                   mutualInformation_s0_res[BLOCK_ROWS], 
                   mutualInformation_s1_res[BLOCK_ROWS];
                   
            double autocorrelations[BLOCK_ROWS * MAX_SHIFT];
            double diffHistogram[BLOCK_ROWS * NUM_BINS];
            double clusters[BLOCK_ROWS * NUM_CLUSTERS];
            double clusterCounts[BLOCK_ROWS * NUM_CLUSTERS];

            cudaMemcpy(results, d_results, currentRows * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(bitFrequencies, d_bitFrequencies, currentRows * 64 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(pairFrequencies, d_pairFrequencies, currentRows * 4 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(tripleFrequencies, d_tripleFrequencies, currentRows * 8 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(quadFrequencies, d_quadFrequencies, currentRows * 16 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(correlations, d_correlations, currentRows * 2 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(uniformity, d_uniformity, currentRows * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(bitEntropy, d_bitEntropy, currentRows * 64 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(globalFrequency, d_globalFrequency, currentRows * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(blockFrequency, d_blockFrequency, currentRows * numBlocksPerSequence * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(runsTest, d_runsTest, currentRows * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(mutualInformation_s0_s1, d_mutualInformation_s0_s1, currentRows * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(mutualInformation_s0_res, d_mutualInformation_s0_res, currentRows * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(mutualInformation_s1_res, d_mutualInformation_s1_res, currentRows * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(autocorrelations, d_autocorrelations, currentRows * MAX_SHIFT * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(diffHistogram, d_diffHistogram, currentRows * NUM_BINS * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(clusters, d_clusters, currentRows * NUM_CLUSTERS * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(clusterCounts, d_clusterCount, currentRows * NUM_CLUSTERS * sizeof(double), cudaMemcpyDeviceToHost);

            // Запись результатов текущего блока в CSV
            for (size_t i = 0; i < currentRows; ++i) {
                resultFile << (offset / BLOCK_ROWS) << "," << i << "," << std::hex << results[i] << std::dec << std::fixed << std::setprecision(6)
                           << "," << correlations[i * 2]
                           << "," << correlations[i * 2 + 1]
                           << "," << uniformity[i];
                for (int b = 0; b < 64; ++b) resultFile << "," << bitFrequencies[i * 64 + b];
                for (int p = 0; p < 4; ++p) resultFile << "," << pairFrequencies[i * 4 + p];
                for (int t = 0; t < 8; ++t) resultFile << "," << tripleFrequencies[i * 8 + t];
                for (int q = 0; q < 16; ++q) resultFile << "," << quadFrequencies[i * 16 + q];
                for (int b = 0; b < 64; ++b) resultFile << "," << bitEntropy[i * 64 + b];
                resultFile << "," << globalFrequency[i]
                           << "," << runsTest[i];
                for (size_t j = 0; j < numBlocksPerSequence; ++j) resultFile << "," << blockFrequency[i * numBlocksPerSequence + j];
                resultFile << ","  << mutualInformation_s0_s1[i] 
                           << "," << mutualInformation_s0_res[i] 
                           << "," << mutualInformation_s1_res[i];
                resultFile << std::fixed << std::setprecision(10);
                for (int shift = 0; shift < MAX_SHIFT; ++shift) resultFile << "," << autocorrelations[i * MAX_SHIFT + shift];
                for (int bin = 0; bin < NUM_BINS; ++bin) resultFile << "," << diffHistogram[i * NUM_BINS + bin];
                resultFile << std::fixed << std::setprecision(1);
                for (int k = 0; k < NUM_CLUSTERS; ++k) resultFile << "," << clusters[i * NUM_CLUSTERS + k];
                for (int k = 0; k < NUM_CLUSTERS; ++k) resultFile << "," << clusterCounts[i * NUM_CLUSTERS + k];
                resultFile << "\n";
            }

            // Очистка памяти
            cudaFree(d_states);
            cudaFree(d_results);
            cudaFree(d_bitFrequencies);
            cudaFree(d_pairFrequencies);
            cudaFree(d_tripleFrequencies);
            cudaFree(d_quadFrequencies);
            cudaFree(d_correlations);
            cudaFree(d_uniformity);
            cudaFree(d_bitEntropy);
            cudaFree(d_globalFrequency);
            cudaFree(d_blockFrequency);
            cudaFree(d_runsTest);
            cudaFree(d_mutualInformation_s0_s1);
            cudaFree(d_mutualInformation_s0_res);
            cudaFree(d_mutualInformation_s1_res);
            cudaFree(d_autocorrelations);
            cudaFree(d_diffHistogram);
            cudaFree(d_clusters);
            cudaFree(d_clusterCount);
        }

        resultFile.close();
        std::cout << "Анализ завершён, результаты сохранены в './data/analysis_results128.csv'." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
