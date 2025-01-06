//компиляция: nvcc -arch=sm_86 collect_statistics128_spectral_analysis.cu -o collect_statistics128_spectral_analysis --ptxas-options=-v -lcufft
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iomanip>

// Константы
const size_t SEQUENCE_LENGTH = 1024; // Длина последовательности
const size_t BLOCK_ROWS = 32;       // Число генераторов в одном блоке (было 32)

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
        uint64_t s0, s1;
        sscanf(line.c_str(), "%lx,%lx", &s0, &s1);
        states.push_back(s0);
        states.push_back(s1);
    }

    return states;
}

// CUDA ядро: Заполнение данных
__global__ void generateSequence(uint64_t* states, cufftComplex* d_fftBuffer, size_t rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < rows) {
        uint64_t s0 = states[idx * 2];
        uint64_t s1 = states[idx * 2 + 1];

        for (int i = 0; i < SEQUENCE_LENGTH; ++i) {
            uint64_t res = s0 + s1;
            d_fftBuffer[idx * SEQUENCE_LENGTH + i].x = (float)res / (float)UINT64_MAX; // Нормализация данных
            d_fftBuffer[idx * SEQUENCE_LENGTH + i].y = 0.0f;

            s1 ^= s0;
            s0 = ((s0 << 24) | (s0 >> (64 - 24))) ^ s1 ^ (s1 << 16);
            s1 = (s1 << 37) | (s1 >> (64 - 37));
        }
    }
}

int main() {
    try {
        // Чтение начальных состояний из CSV
        std::vector<uint64_t> initialStates = readCSV("dependency_table_xoroshiro128_0.csv");
        size_t totalRows = initialStates.size() / 2;

        // Открытие файла для записи результатов
        std::ofstream resultFile("./data/analysis_results128_spectral.csv");
        resultFile << "Block,Row";
        for (int i = 0; i < SEQUENCE_LENGTH / 2; ++i) resultFile << ",Spectral_" << i;
        resultFile << "\n";

        // Блоками обрабатываем данные
        for (size_t offset = 0; offset < totalRows; offset += BLOCK_ROWS) {
            size_t currentRows = std::min(BLOCK_ROWS, totalRows - offset);

            // Указатели на устройстве
            uint64_t* d_states;
            cufftComplex* d_fftBuffer;

            // Выделение памяти на устройстве
            if (cudaMalloc(&d_states, currentRows * 2 * sizeof(uint64_t)) != cudaSuccess) {
                throw std::runtime_error("Ошибка выделения памяти для d_states");
            }
            if (cudaMalloc(&d_fftBuffer, currentRows * SEQUENCE_LENGTH * sizeof(cufftComplex)) != cudaSuccess) {
                throw std::runtime_error("Ошибка выделения памяти для d_fftBuffer");
            }

            // Копирование начальных состояний на устройство
            cudaMemcpy(d_states, &initialStates[offset * 2], currentRows * 2 * sizeof(uint64_t), cudaMemcpyHostToDevice);

            // Запуск CUDA ядра для генерации последовательностей
            int threadsPerBlock = 768;
            int numBlocks = (currentRows + threadsPerBlock - 1) / threadsPerBlock;
            generateSequence<<<numBlocks, threadsPerBlock>>>(d_states, d_fftBuffer, currentRows);

            // План FFT
            cufftHandle plan;
            if (cufftPlan1d(&plan, SEQUENCE_LENGTH, CUFFT_C2C, currentRows) != CUFFT_SUCCESS) {
                throw std::runtime_error("Ошибка планирования FFT");
            }

            // Выполнение FFT
            cufftExecC2C(plan, d_fftBuffer, d_fftBuffer, CUFFT_FORWARD);

            // Копирование результатов обратно на хост
            std::vector<cufftComplex> fftResults(currentRows * SEQUENCE_LENGTH);
            cudaMemcpy(fftResults.data(), d_fftBuffer, currentRows * SEQUENCE_LENGTH * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

            // Освобождение памяти и уничтожение плана
            cudaFree(d_states);
            cudaFree(d_fftBuffer);
            cufftDestroy(plan);

            // Запись результатов текущего блока в CSV
            for (size_t i = 0; i < currentRows; ++i) {
                resultFile << (offset / BLOCK_ROWS) << "," << i;
                for (int j = 0; j < SEQUENCE_LENGTH / 2; ++j) {
                    float amplitude = sqrt(
                        fftResults[i * SEQUENCE_LENGTH + j].x * fftResults[i * SEQUENCE_LENGTH + j].x +
                        fftResults[i * SEQUENCE_LENGTH + j].y * fftResults[i * SEQUENCE_LENGTH + j].y
                    ) / SEQUENCE_LENGTH;
                    resultFile << "," << amplitude;
                }
                resultFile << "\n";
            }
        }

        resultFile.close();
        std::cout << "Анализ завершён, результаты сохранены в './data/spectral_analysis_results128.csv'." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
