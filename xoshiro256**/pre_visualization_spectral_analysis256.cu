#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>

// Константы
const int BLOCK_ROWS = 256;     // Число строк в одном блоке 256/4096 -> 65536/4096 строк в outputFile
const int NUM_COLUMNS = 512;    // Число столбцов (Spectral_0...Spectral_511)

// CUDA ядро для усреднения столбцов
__global__ void computeBlockAverages(float* d_input, float* d_output, int rows) {
    int col = threadIdx.x; // Поток отвечает за один столбец
    if (col < NUM_COLUMNS) {
        float sum = 0.0f;
        for (int i = 0; i < rows; ++i) {
            sum += d_input[i * NUM_COLUMNS + col];
        }
        d_output[col] = sum / rows; // Усреднение
    }
}

int main() {
    try {
        std::ifstream inputFile("./data/analysis_results256_spectral.csv");
        std::ofstream outputFile("./data/processed_spectral_analysis256.csv");
        if (!inputFile.is_open() || !outputFile.is_open()) {
            throw std::runtime_error("Не удалось открыть файлы для чтения или записи");
        }

        // Пропускаем заголовок
        std::string header;
        std::getline(inputFile, header);

        // Записываем заголовок нового файла
        outputFile << "Block";
        for (int i = 0; i < NUM_COLUMNS; ++i) {
            outputFile << ",Spectral_" << i;
        }
        outputFile << "\n";

        // Память на хосте
        std::vector<float> hostBlock(BLOCK_ROWS * NUM_COLUMNS);
        std::vector<float> hostAverages(NUM_COLUMNS);

        // Память на устройстве
        float *d_input, *d_output;
        cudaMalloc(&d_input, BLOCK_ROWS * NUM_COLUMNS * sizeof(float));
        cudaMalloc(&d_output, NUM_COLUMNS * sizeof(float));

        /*int minGridSize, blockSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeBlockAverages, 0, 0);
        std::cout << "Рекомендуемый BLOCK_SIZE: " << blockSize << std::endl;*/


        int blockIndex = 0;
        while (!inputFile.eof()) {
            // Считываем блок данных
            int rowCount = 0;
            for (; rowCount < BLOCK_ROWS && !inputFile.eof(); ++rowCount) {
                std::string line;
                std::getline(inputFile, line);
                if (line.empty()) break;

                // Парсим строку
                size_t pos = 0;
                // Пропускаем первые две колонки
                for (int skip = 0; skip < 2; ++skip) {
                    pos = line.find(',', pos) + 1; // Найти следующую запятую и продвинуть позицию
                }

                // Читаем значения для столбцов Spectral
                for (int col = 0; col < NUM_COLUMNS; ++col) {
                    size_t nextPos = line.find(',', pos);
                    hostBlock[rowCount * NUM_COLUMNS + col] = std::stof(line.substr(pos, nextPos - pos));
                    pos = nextPos + 1;
                }
            }

            if (rowCount == 0) break; // Конец файла

            // Копируем данные на устройство
            cudaMemcpy(d_input, hostBlock.data(), rowCount * NUM_COLUMNS * sizeof(float), cudaMemcpyHostToDevice);

            // Запуск CUDA ядра
            computeBlockAverages<<<1, NUM_COLUMNS>>>(d_input, d_output, rowCount);

            // Копируем результат обратно на хост
            cudaMemcpy(hostAverages.data(), d_output, NUM_COLUMNS * sizeof(float), cudaMemcpyDeviceToHost);

            // Записываем усреднённые данные в файл
            outputFile << blockIndex++;
            for (int i = 0; i < NUM_COLUMNS; ++i) {
                outputFile << "," << hostAverages[i];
            }
            outputFile << "\n";
        }

        // Освобождение ресурсов
        cudaFree(d_input);
        cudaFree(d_output);

        inputFile.close();
        outputFile.close();

        std::cout << "Предобработка завершена, результат сохранён в ./data/processed_spectral_analysis256.csv" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
