//компиляция: nvcc -arch=sm_86 pre_visualization_pocessing256.cu -o pre_visualization_pocessing256 --ptxas-options=-v
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cuda_runtime.h>
#include <iomanip>          //outputFile << std::fixed << std::setprecision(6);
#include <map>

#define BLOCK_SIZE 640
#define NUM_COLUMNS 522
#define NUM_ROWS_PER_BLOCK 4096

// Группы столбцов для обработки
enum ColumnGroup {
    CORRELATION = 0,
    UNIFORMITY,
    BIT_FREQ,
    PAIRS,
    TRIPLES,
    QUADS,
    ENTROPY,
    GLOBAL_STATS,
    RUN_TESTS,
    BLOCK_FREQUENCY,
    MUTUAL_INFORMATION,
    AUTOCORR,
    DIFFERENCES,
    CLUSTERS,
    CLUSTER_COUNT,
    NUM_GROUPS  // Всего групп
};

// Смещения групп столбцов в исходном массиве
__constant__ size_t d_groupOffsets[NUM_GROUPS];

const size_t h_groupOffsets[NUM_GROUPS] = {
    3,   // CORRELATION (столбцы 3-4)
    7,   // UNIFORMITY (столбец 5)
    8,   // BIT_FREQ (столбцы 6-69)
    72,  // PAIRS (столбцы 70-73)
    76,  // TRIPLES (столбцы 74-81)
    84,  // QUADS (столбцы 82-97)
    100, // ENTROPY (столбцы 98-161)
    164, // GLOBAL_STATS (столбец 162)
    165, // RUN_TESTS (столбец 163)
    166, // BLOCK_FREQUENCY (столбцы 164-179)
    182, // MUTUAL_INFORMATION (столбцы 180-182)
    192, // AUTOCORR (столбцы 183-192)
    202, // DIFFERENCES (столбцы 193-448)
    458, // CLUSTERS (столбцы 449-452)
    490  // CLUSTER_COUNT
};

// Размеры групп столбцов
__constant__ size_t d_groupSizes[NUM_GROUPS];

const size_t h_groupSizes[NUM_GROUPS] = {
    4,  // CORRELATION
    1,  // UNIFORMITY
    64, // BIT_FREQ
    4,  // PAIRS
    8,  // TRIPLES
    16, // QUADS
    64, // ENTROPY
    1,  // GLOBAL_STATS
    1,  // RUN_TESTS
    16, // BLOCK_FREQUENCY
    10, // MUTUAL_INFORMATION
    10, // AUTOCORR
    256,// DIFFERENCES
    32, // CLUSTERS
    32  // CLUSTER_COUNT
};

// Копирование значений в `__constant__` память
void initializeConstantMemory() {
    cudaMemcpyToSymbol(d_groupSizes, h_groupSizes, sizeof(size_t) * NUM_GROUPS);
    cudaMemcpyToSymbol(d_groupOffsets, h_groupOffsets, sizeof(size_t) * NUM_GROUPS);
}


void writeHeader(std::ofstream& outputFile) {
    // Массив названий групп
    const std::string groupNames[NUM_GROUPS] = {
        "Correlation",       // CORRELATION
        "Uniformity",        // UNIFORMITY
        "BitFrequency",      // BIT_FREQ
        "Pairs",             // PAIRS
        "Triples",           // TRIPLES
        "Quads",             // QUADS
        "Entropy",           // ENTROPY
        "GlobalStats",       // GLOBAL_STATS
        "RunsTest",          // RUN_TESTS
        "BlockFrequency",    // BLOCK_FREQUENCY
        "MutualInformation", // MUTUAL_INFORMATION
        "Autocorrelation",   // AUTOCORR
        "Differences",       // DIFFERENCES
        "Clusters",          // CLUSTERS
        "ClusterCount"       // CLUSTER_COUNT
    };
    
    const std::string calculationsNames[5] = {
        "_mean_", 
        "_dispersion_",
        "_stDev_",
        "_entropy_",
        "_distance_"
    };

    enum numCalc {mean = 0, dispersion, stDev, entropy, distance};
    const std::vector<int> CalculationsGroups[NUM_GROUPS] = {
        {mean, dispersion},             // CORRELATION
        {mean},                         // UNIFORMITY
        {mean, entropy},                // BIT_FREQ
        {mean, dispersion},             // PAIRS
        {mean, dispersion},             // TRIPLES
        {mean, dispersion},             // QUADS
        {mean, stDev},                  // ENTROPY
        {mean, dispersion},             // GLOBAL_STATS
        {mean, dispersion},             // RUN_TESTS
        {mean, dispersion},             // BLOCK_FREQUENCY
        {mean, dispersion},             // MUTUAL_INFORMATION
        {mean, stDev},                  // AUTOCORR
        {mean},                         // DIFFERENCES
        {mean, distance},               // CLUSTERS
        {mean}                          // CLUSTER_COUNT
    };

    // Генерация заголовка
    for (size_t group = 0; group < NUM_GROUPS; ++group) {
        if(group == CLUSTERS){
            for (size_t i = 0; i < h_groupSizes[group]; ++i) {
                outputFile << groupNames[group] << calculationsNames[CalculationsGroups[group][0]] << i;
                outputFile << ",";
                outputFile << groupNames[group] << calculationsNames[CalculationsGroups[group][1]] << i << "_" << (i+1)%h_groupSizes[CLUSTERS];
                if (!(group == NUM_GROUPS - 1 && i == h_groupSizes[group] - 1)) {
                    outputFile << ",";
                }
            }
        }
        else{
            for (size_t i = 0; i < h_groupSizes[group]; ++i) {
                for (size_t j = 0; j < CalculationsGroups[group].size(); j++){
                    outputFile << groupNames[group] << calculationsNames[CalculationsGroups[group][j]] << i;
                    if (!(group == NUM_GROUPS - 1 && i == h_groupSizes[group] - 1 && j == CalculationsGroups[group].size() - 1)) {
                        outputFile << ",";
                    }
                }
            }
        }
        
    }
    outputFile << "\n"; // Закончить строку заголовка
}

// Функция для считывания блока строк из файла
std::vector<std::vector<double>> readBlock(std::ifstream& file, size_t numRows) {
    
    std::vector<std::vector<double>> block(numRows, std::vector<double>(NUM_COLUMNS));
    std::string line;
    size_t row = 0;
    
    // Пропуск заголовка
    if (file.tellg() == 0) std::getline(file, line);

    while (row < numRows && std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        size_t col = 0;
        
        while (std::getline(ss, value, ',')) {
            //printf("test \t\t\t\t readBlock getline row = %lu col = %lu value = {%s} - ok\n", row, col, value.c_str());
            block[row][col++] = col == 2 ? (double)std::stoull(value, nullptr, 16) : std::stod(value);
            //printf("test \t\t\t\t readBlock stod(value) %lf - ok\n", block[row][col-1]);
        }
        ++row;
    }

    block.resize(row); // Уменьшаем размер, если строки закончились
    return block;
}

 // функция для усреднения значений и расчёта дисперсии
__device__ void processCorrelation(double* input, double* sharedSums, double* sharedSquaredSums,
                                   double* output, size_t offset, size_t groupSize, size_t numRows) {
    for (size_t col = 0; col < groupSize; ++col) {
        double mean = sharedSums[offset + col] / numRows;
        double variance = (sharedSquaredSums[offset + col] / numRows) - (mean * mean);

        // Сохраняем среднее и дисперсию
        output[offset + col] = mean;           // Среднее значение
        output[offset + groupSize + col] = variance; // Дисперсия
    }
}

// CUDA Kernel для обработки данных
__global__ void processData(
    double* input, double* means, double* variances, double* entropies, double* distanceClusters, int numRows) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= NUM_COLUMNS) return;

    // Определяем группу текущего столбца
    int group = -1;
    for (int g = 0; g < NUM_GROUPS; ++g) {
        if (tid >= d_groupOffsets[g] && tid < d_groupOffsets[g] + d_groupSizes[g]) {
            group = g;
            break;
        }
    }

    // Если группа не определена, выходим
    if (group == -1) return;

    // Вычисление среднего значения
    double mean = 0.0;
    for (int i = 0; i < numRows; ++i) {
        mean += input[i * NUM_COLUMNS + tid];
    }
    mean /= numRows;
    means[tid] = mean;

    // Дополнительные вычисления в зависимости от группы
    switch (group) {
        case CORRELATION: {
            // Вычисление дисперсии
            double variance = 0.0;
            for (int i = 0; i < numRows; ++i) {
                double diff = input[i * NUM_COLUMNS + tid] - mean;
                variance += diff * diff;
            }
            variance /= numRows;
            variances[tid] = variance;
            break;
        }

        case BIT_FREQ: {
            // Вычисление энтропии
            double entropy = 0.0;
            for (int i = 0; i < numRows; ++i) {
                double p = input[i * NUM_COLUMNS + tid] / numRows;
                if (p > 0.0) {
                    entropy -= p * log2(p);
                }
            }
            entropies[tid] = entropy;
            break;
        }

        case PAIRS: 
        case TRIPLES: 
        case QUADS: {
            // Вычисление среднего значения и дисперсии
            double variance = 0.0;
            for (int i = 0; i < numRows; ++i) {
                double diff = input[i * NUM_COLUMNS + tid] - mean;
                variance += diff * diff;
            }
            variance /= numRows;
            variances[tid] = variance;
            break;
        }

        case ENTROPY: {
            // Обработка значений энтропии
            double stdDev = 0.0; // Стандартное отклонение
            for (int i = 0; i < numRows; ++i) {
                double diff = input[i * NUM_COLUMNS + tid] - mean;
                stdDev += diff * diff;
            }
            stdDev = sqrt(stdDev / numRows);
            variances[tid] = stdDev; // Стандартное отклонение
            break;
        }

        case GLOBAL_STATS: {
            // Вычисление дисперсии
            double variance = 0.0;
            for (int i = 0; i < numRows; ++i) {
                double diff = input[i * NUM_COLUMNS + tid] - mean;
                variance += diff * diff;
            }
            variances[tid] = variance / numRows;
            break;
        }

        case RUN_TESTS: {
            // Вычисление среднего значения и дисперсии RunsTest
            double variance = 0.0;
            for (int i = 0; i < numRows; ++i) {
                double diff = input[i * NUM_COLUMNS + tid] - mean;
                variance += diff * diff;
            }
            variances[tid] = variance / numRows;
            break;
        }

        case BLOCK_FREQUENCY: {
            // Вычисление среднего значения и дисперсии для каждого блока
            double variance = 0.0;
            for (int i = 0; i < numRows; ++i) {
                double diff = input[i * NUM_COLUMNS + tid] - mean;
                variance += diff * diff;
            }
            variances[tid] = variance / numRows;
            break;
        }

        case MUTUAL_INFORMATION: {
            // Вычисление средней взаимной информации и дисперсии
            double variance = 0.0;
            for (int i = 0; i < numRows; ++i) {
                double diff = input[i * NUM_COLUMNS + tid] - mean;
                variance += diff * diff;
            }
            variances[tid] = variance / numRows;
            break;
        }

        case AUTOCORR: {
            // Вычисление средней автокорреляции и стандартного отклонения
            double stdDev = 0.0;
            for (int i = 0; i < numRows; ++i) {
                double diff = input[i * NUM_COLUMNS + tid] - mean;
                stdDev += diff * diff;
            }
            stdDev = sqrt(stdDev / numRows);
            variances[tid] = stdDev; // Записываем стандартное отклонение
            break;
        }

        case CLUSTERS: {
            // Средние координаты кластеров
            int next_cluster_num = (tid - d_groupOffsets[CLUSTERS] + 1) % 32;

            double mean_next_cluster = 0;
            for (int i = 0; i < numRows; ++i) {
                mean_next_cluster += input[i * NUM_COLUMNS + d_groupOffsets[CLUSTERS] + next_cluster_num];
            }
            mean_next_cluster /= numRows;

            double distance = fabs(mean - mean_next_cluster);
            distanceClusters[tid] = distance;
            break;
        }


        default:
            break;
    }
}

// Основная программа
int main() {
    std::string inputFilename = "./data/analysis_results256.csv";
    std::string outputFilename = "./data/processed_data256.csv";

    size_t numRowsPerBlock = NUM_ROWS_PER_BLOCK;

    std::ifstream inputFile(inputFilename);
    if (!inputFile.is_open()) {
        std::cerr << "Ошибка открытия файла: " << inputFilename << std::endl;
        return -1;
    }

    std::ofstream outputFile(outputFilename);
    if (!outputFile.is_open()) {
        std::cerr << "Ошибка открытия файла для записи: " << outputFilename << std::endl;
        return -1;
    }

    // Инициализация константной памяти
    initializeConstantMemory();

    // Запись заголовка в файл
    writeHeader(outputFile);

    // Выделение памяти на GPU
    double* d_input;
    double* d_means;
    double* d_variances;
    double* d_entropies;
    double* d_distanceClusters;

    size_t inputSize = numRowsPerBlock * NUM_COLUMNS * sizeof(double);
    size_t outputSizeMeans = NUM_COLUMNS * sizeof(double);
    size_t outputSizeVariances = NUM_COLUMNS * sizeof(double);
    size_t outputSizeEntropies = NUM_COLUMNS * sizeof(double);
    size_t outputSizeFrequencies = NUM_COLUMNS * sizeof(double);

    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_means, outputSizeMeans);
    cudaMalloc(&d_variances, outputSizeVariances);
    cudaMalloc(&d_entropies, outputSizeEntropies);
    cudaMalloc(&d_distanceClusters, outputSizeFrequencies);

    /*int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, processData, 0, 0);
    std::cout << "Рекомендуемый BLOCK_SIZE: " << blockSize << std::endl;*/

    while (!inputFile.eof()) {
        // Считываем блок строк
        auto block = readBlock(inputFile, numRowsPerBlock);

        if (block.empty()) break;

        size_t rowsInBlock = block.size();
        std::vector<double> flattenedBlock(rowsInBlock * NUM_COLUMNS);

        for (size_t i = 0; i < rowsInBlock; ++i) {
            for (size_t j = 0; j < NUM_COLUMNS; ++j) {
                flattenedBlock[i * NUM_COLUMNS + j] = block[i][j];
            }
        }

        // Копируем данные на GPU
        cudaMemcpy(d_input, flattenedBlock.data(), rowsInBlock * NUM_COLUMNS * sizeof(double), cudaMemcpyHostToDevice);

        // Запуск CUDA Kernel
        int numBlocks = (rowsInBlock + BLOCK_SIZE - 1) / BLOCK_SIZE;
        processData<<<numBlocks, BLOCK_SIZE>>>(d_input, d_means, d_variances, d_entropies, d_distanceClusters, rowsInBlock);

        // Копируем результаты обратно на CPU
        std::vector<double> means(NUM_COLUMNS);
        std::vector<double> variances(NUM_COLUMNS, 0.0); // Для тех столбцов, где дисперсия не считается
        std::vector<double> entropies(NUM_COLUMNS, 0.0); // Для тех столбцов, где энтропия не считается
        std::vector<double> distanceClusters(NUM_COLUMNS, 0.0); // Для DIFFERENCES

        cudaMemcpy(means.data(), d_means, outputSizeMeans, cudaMemcpyDeviceToHost);
        cudaMemcpy(variances.data(), d_variances, outputSizeVariances, cudaMemcpyDeviceToHost);
        cudaMemcpy(entropies.data(), d_entropies, outputSizeEntropies, cudaMemcpyDeviceToHost);
        cudaMemcpy(distanceClusters.data(), d_distanceClusters, outputSizeFrequencies, cudaMemcpyDeviceToHost);

        // Запись результатов в файл
        outputFile << std::fixed << std::setprecision(10);
        for (size_t j = h_groupOffsets[0]; j < NUM_COLUMNS; ++j) {
            // Записываем среднее значение
            outputFile << means[j];

            // Если требуется, добавляем дополнительные значения
            if (j >= h_groupOffsets[CORRELATION] && j < h_groupOffsets[CORRELATION] + h_groupSizes[CORRELATION]) {
                outputFile << "," << variances[j];
            } else if (j >= h_groupOffsets[BIT_FREQ] && j < h_groupOffsets[BIT_FREQ] + h_groupSizes[BIT_FREQ]) {
                outputFile << "," << entropies[j];
            } else if (j >= h_groupOffsets[PAIRS] && j < h_groupOffsets[ENTROPY]) { // Добавление дисперсий для пар, троек и четверок
                outputFile << "," << variances[j];
            } else if (j >= h_groupOffsets[ENTROPY] && j < h_groupOffsets[GLOBAL_STATS]) { // Добавление энтропий для битов
                outputFile << "," << variances[j];
            } else if (j >= h_groupOffsets[GLOBAL_STATS] && j < h_groupOffsets[RUN_TESTS]) {
                outputFile << "," << variances[j];
            }else if (j >= h_groupOffsets[RUN_TESTS] && j < h_groupOffsets[BLOCK_FREQUENCY]) {
                outputFile << "," << variances[j];
            }else if (j >= h_groupOffsets[BLOCK_FREQUENCY] && j < h_groupOffsets[MUTUAL_INFORMATION]) {
                outputFile << "," << variances[j];
            }else if (j >= h_groupOffsets[MUTUAL_INFORMATION] && j < h_groupOffsets[AUTOCORR]) {
                outputFile << "," << variances[j];
            }else if (j >= h_groupOffsets[AUTOCORR] && j < h_groupOffsets[DIFFERENCES]) {
                outputFile << "," << variances[j];
            }else if (j >= h_groupOffsets[CLUSTERS] && j < h_groupOffsets[CLUSTER_COUNT]) {
                outputFile << "," << distanceClusters[j];
            }

            // Добавляем запятую между столбцами
            if (j < NUM_COLUMNS - 1) outputFile << ",";
        }
        outputFile << "\n"; // Переход на новую строку
    }

    // Освобождение памяти
    cudaFree(d_input);
    cudaFree(d_means);
    cudaFree(d_variances);
    cudaFree(d_entropies);
    cudaFree(d_distanceClusters);

    inputFile.close();
    outputFile.close();

    std::cout << "Предобработка завершена. Результаты сохранены в " << outputFilename << std::endl;
    return 0;
}