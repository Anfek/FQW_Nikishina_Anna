#include <iostream>
#include <iomanip>
#include <cstdint>
#include <random>
#include <vector>

// Функция для циклического сдвига влево
uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

uint64_t xoroshiro256_next(uint64_t &s0, uint64_t &s1, uint64_t &s2, uint64_t &s3){
    // Обновление состояния
    uint64_t t = s1 << 17;
    s2 ^= s0;
    s3 ^= s1;
    s1 ^= s2;
    s0 ^= s3;

    s2 ^= t;
    s3 = rotl(s3, 45);

    return rotl(s1 * 5, 7) * 9;
}


// Генерация случайных начальных состояний
void generate_random_states(uint64_t &s0, uint64_t &s1, uint64_t &s2, uint64_t &s3) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    s0 = gen();
    s1 = gen();
    s2 = gen();
    s3 = gen();
}

// Основная функция генерации последовательности
std::vector<uint64_t> generate_sequence(uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3) {
    uint64_t s[4][4];
    uint64_t r[4];

    // Начальное состояние
    s[0][0] = s0;
    s[1][0] = s1;
    s[2][0] = s2;
    s[3][0] = s3;

    r[0] = rotl(s[1][0] * 5, 7) * 9;

    // Генерация r_1
    for (int i = 1; i < 4; i++){
        r[i] = xoroshiro256_next(s0, s1, s2, s3);
        s[0][i] = s0;
        s[1][i] = s1;
        s[2][i] = s2;
        s[3][i] = s3;
    }
    
    // Вывод значений в 16-ричном формате
    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++){
            std::cout << "s"<< j << "_" << i <<" = 0x" << std::hex << s[j][i] << " \t ";
        }
        std::cout << "\nr" << i << "   = 0x" << std::hex << r[i] << "\n";
    }

    return {r[0], r[1], r[2], r[3]};
}

// Функция для циклического сдвига вправо
uint64_t rotr(uint64_t x, int k) {
    return (x >> k) | (x << (64 - k));
}

// Функция восстановления состояния
void crack_xoshiro256(uint64_t r0, uint64_t r1, uint64_t r2, uint64_t r3) {
    // Вычисляем 5^(-1) и 9^(-1)
    uint64_t rev_5 = 0xCCCCCCCCCCCCCCCD;
    uint64_t rev_9 = 0x8E38E38E38E38E39;

    // Вычисляем s1
    uint64_t s1_0 = rotr(r0 * rev_9, 7) * rev_5;
    uint64_t s1_1 = rotr(r1 * rev_9, 7) * rev_5;
    uint64_t s1_2 = rotr(r2 * rev_9, 7) * rev_5;
    uint64_t s1_3 = rotr(r3 * rev_9, 7) * rev_5;

    // Вычисляем s0_1 s0_2
    uint64_t s0_1 = s1_0 ^ (uint64_t)(s1_0 << 17) ^ s1_2;
    uint64_t s0_2 = s1_1 ^ (s1_1 << 17) ^ s1_3;

    // Вычисляем s2_1
    uint64_t s2_1 = s1_2 ^ s1_1 ^ s0_1;

    // Вычисляем s3_1
    uint64_t s3_1 = s0_2 ^ s0_1 ^ s1_1;


    // Вычисляем s3_0
    uint64_t s3_0 = rotr(s3_1, 45) ^ s1_0;

    // Вычисляем s0_0
    uint64_t s0_0 = s1_0 ^ s0_1 ^ s3_0;

    // Вычисляем s2_0
    uint64_t s2_0 = (s1_0 << 17) ^ s2_1 ^ s0_0;
    

    // Вывод результатов
    std::cout << "\nRecovered states:\n";
    std::cout << "s0_0 = 0x" << std::hex << s0_0 << "\n";
    std::cout << "s1_0 = 0x" << std::hex << s1_0 << "\n";
    std::cout << "s2_0 = 0x" << std::hex << s2_0 << "\n";
    std::cout << "s3_0 = 0x" << std::hex << s3_0 << "\n";
}


// Точка входа
int main() {
    uint64_t s0, s1, s2, s3;

    // Сгенерировать случайные состояния
    generate_random_states(s0, s1, s2, s3);

    // Сгенерировать последовательность
    std::vector<uint64_t> r = generate_sequence(s0, s1, s2, s3);

    // Восстановить начальные состояния
    crack_xoshiro256(r[0], r[1], r[2], r[3]);

    return 0;
}
