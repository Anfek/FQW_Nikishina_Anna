#include <stdint.h>

class Xoshiro256xx{
private:
    uint64_t s[4];

    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

public:
    Xoshiro256xx(uint64_t init_s0, uint64_t init_s1, uint64_t init_s2, uint64_t init_s3) : s{init_s0, init_s1, init_s2, init_s3} {}

    uint64_t next(void) {
        const uint64_t result = rotl(s[1] * 5, 7) * 9;

        const uint64_t t = s[1] << 17;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;

        s[3] = rotl(s[3], 45);

        return result;
    }

    const uint64_t* getState() const {
        return s; // Возвращаем указатель на массив состояния
    }
};
