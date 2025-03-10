#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>
#include <chrono>
#include "xoroshiro128+.h" // Включаем реализацию генератора

void generateDependencyTableXoroshiro(const std::string& filename, uint64_t numEntries) {
    // Открываем файл для записи
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        std::cerr << "Не удалось открыть файл для записи: " << filename << std::endl;
        return;
    }

    // Заголовок таблицы
    outputFile << "init_s0,init_s1,result,next_s0,next_s1\n";

    // Генератор случайных чисел для начального состояния
    std::random_device rd;
    std::mt19937_64 rng(static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);

    // Генерация данных
    for (uint64_t i = 0; i < numEntries; ++i) {
        // Случайные начальные состояния
        uint64_t init_s0 = dist(rng);
        uint64_t init_s1 = dist(rng);

        // Инициализация генератора
        Xoroshiro128plus generator(init_s0, init_s1);

        // Генерация первого результата
        uint64_t result = generator.next();

        // Получение следующего состояния
        const uint64_t* nextState = generator.getState();

        // Запись строки в файл
        outputFile << std::hex << std::setfill('0') << std::setw(16) << init_s0 << ","
                   << std::hex << std::setfill('0') << std::setw(16) << init_s1 << ","
                   << std::hex << std::setfill('0') << std::setw(16) << result << ","
                   << std::hex << std::setfill('0') << std::setw(16) << nextState[0] << ","
                   << std::hex << std::setfill('0') << std::setw(16) << nextState[1] << "\n";

        // Отображение прогресса в консоли
        if (i % 100000 == 0) {
            std::cout << "\rСгенерировано строк: " 
                    << std::fixed << std::setprecision(2) << (static_cast<double>(i) / numEntries) * 100.0 
                    << "%    " << i << "/" << numEntries << std::flush;
        }
    }

    std::cout << "\nГенерация завершена. Данные сохранены в файл: " << filename << std::endl;
    outputFile.close();
}

int main(int argc, char const *argv[]) {
    system("chcp 65001 >nul 2>&1");
    // Имя выходного файла и количество записей
    std::string filename;           // Укажите название файла
    uint64_t numEntries;            // Укажите количество строк

    if (argc == 2 && std::isdigit(argv[1][0])) {
        // Аргумент — цифра, формируем имя файла
        filename = "dependency_table_xoroshiro128_" + std::string(1, argv[1][0]) + ".csv";
        numEntries = 0x1000000; //2^24 = 16777216

        generateDependencyTableXoroshiro(filename, numEntries); // Генерация таблицы
        return 0;
    }
    else if(argc == 3){
        // Два аргумента: имя файла и количество строк
        filename = std::string(argv[1]) + ".csv";
        numEntries = std::stoi(std::string(argv[2]));

        generateDependencyTableXoroshiro(filename, numEntries);
        return 0;
    }
    else if(argc == 1){
        // Без аргументов: стандартные параметры
        filename = "dependency_table_xoroshiro128_268kk.csv";
        numEntries = 0x10000000;         // 268kk
        generateDependencyTableXoroshiro(filename, numEntries);

        filename = "dependency_table_xoroshiro128_4kkk.csv";
        numEntries = 0x100000000;        // 4kkk
        generateDependencyTableXoroshiro(filename, numEntries);

        return 0;
    }
    
    std::cout << "Упс, что-то введено некорректно..." << std::endl;
    return 0;
}