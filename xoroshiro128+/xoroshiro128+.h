#include <stdint.h>
#include <iostream>

class Xoroshiro128plus {
private:
    uint64_t s[2];

    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

public:
    Xoroshiro128plus(uint64_t init_s0, uint64_t init_s1) : s{init_s0, init_s1} {}

    uint64_t next(void) {
        const uint64_t s0 = s[0];
        uint64_t s1 = s[1];
        const uint64_t result = s0 + s1;

        s1 ^= s0;
        s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16); // a, b
        s[1] = rotl(s1, 37); // c

        return result;
    }

    const uint64_t* getState() const {
        return s; // Возвращаем указатель на массив состояния
    }
};
