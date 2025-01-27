#include <iostream>
#include <iomanip>
#include <cstdint>
#include <random>
#include <vector>

// Функция для циклического сдвига влево
uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

uint64_t xoroshiro128_next(uint64_t &s0, uint64_t &s1){
    // Обновление состояния
    s1 ^= s0;
    s0 = rotl(s0, 24) ^ s1 ^ (s1 << 16);
    s1 = rotl(s1, 37);

    return s0 ^ s1;
}


// Генерация случайных начальных состояний
void generate_random_states(uint64_t &s0, uint64_t &s1) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    s0 = gen();
    s1 = gen();
}

// Основная функция генерации последовательности
std::vector<uint64_t> generate_sequence(uint64_t s0_0, uint64_t s1_0) {
    // Начальное состояние
    uint64_t s0 = s0_0;
    uint64_t s1 = s1_0;

    // Генерация r_0
    uint64_t r0 = s0 ^ s1;

    // Сохранение r_1
    uint64_t r1 = xoroshiro128_next(s0, s1); 

    // Вывод значений в 16-ричном формате
    std::cout << "s0_0 = 0x" << std::hex << s0_0 << "\n";
    std::cout << "s1_0 = 0x" << std::hex << s1_0 << "\n";
    std::cout << "r0   = 0x" << std::hex << r0 << "\n";
    std::cout << "s0_1 = 0x" << std::hex << s0 << "\n";
    std::cout << "s1_1 = 0x" << std::hex << s1 << "\n";
    std::cout << "r1   = 0x" << std::hex << r1 << "\n";

    return {r0, r1};
}

// Функция для циклического сдвига вправо
uint64_t rotr(uint64_t x, int k) {
    return (x >> k) | (x << (64 - k));
}

// Функция восстановления состояния
void crack_xoroshiro128(uint64_t r0, uint64_t r1) {
    // Вычисляем s0_0 и s1_0
    uint64_t s0_0 = rotr(r1 ^ rotl(r0, 37) ^ r0 ^ (r0 << 16), 24);
    uint64_t s1_0 = r0 ^ s0_0;

    // Вывод результатов
    std::cout << "\nRecovered states:\n";
    std::cout << "s0_0 = 0x" << std::hex << s0_0 << "\n";
    std::cout << "s1_0 = 0x" << std::hex << s1_0 << "\n";
}

// Точка входа
int main() {
    uint64_t s0, s1;

    // Сгенерировать случайные состояния
    generate_random_states(s0, s1);

    // Сгенерировать последовательность
    std::vector<uint64_t> r = generate_sequence(s0, s1);

    // Восстановить начальные состояния
    crack_xoroshiro128(r[0], r[1]);

    return 0;
}
