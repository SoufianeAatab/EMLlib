#ifndef N_HH
#define N_HH
#include <iostream>
#include <assert.h>
#include <chrono>
#include <stdint.h>

#define USEMOMENTUM 1

#if ONCOMPUTER
#include <random>
#include <fstream>

std::random_device rd;
std::mt19937 generator(0);
std::uniform_real_distribution<double> distribution;

#else
#include <Arduino.h>
#endif
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef float f32;
typedef double f64;

typedef uint32_t bool32;

struct memory_arena
{
    size_t Used;
    size_t Size;
    u8 *Base;
};
static memory_arena MemoryArena = {};

inline void *PushSize(memory_arena *Arena, size_t SizeToReserve)
{
    assert(Arena->Used + SizeToReserve <= Arena->Size);
    void *Result = Arena->Base + Arena->Used;
    Arena->Used += SizeToReserve;
    return (void *)Result;
}

void InitMemory(u32 Size)
{
    MemoryArena.Base = (u8 *)malloc(Size);
    MemoryArena.Size = Size;
    MemoryArena.Used = 0;
}

static f32 randLowVal = 0;
static f32 randHighVal = 0;
static f32 scale = 0.0f;

#if ONCOMPUTER
f32 uniform_rand(){
    static std::mt19937 gen(1);
    //std::printf("between %f, %f\n", randLowVal, randHighVal);
    std::uniform_real_distribution<f32> dst(randLowVal, randHighVal);
    return dst(gen);
}
#else
f32 uniform_rand(){
    float num = (float)random(1001) / 1000.0 - randLowVal;
    return num;
}
#endif

struct M
{
    f32 *data;
    u32 rows;
    u32 cols;

    M() = default;
    M(f32 *data, u32 rows, u32 cols) : data(data), rows(rows), cols(cols) {}

    static M ones(u32 rows, u32 cols)
    {
        f32 *data = (f32 *)PushSize(&MemoryArena, rows * cols * sizeof(f32));
        M o(data, rows, cols);
        for (u32 i = 0; i < rows * cols; ++i)
        {
            o.data[i] = 1.0f;
        }
        return o;
    }

    static M zeros(u32 rows, u32 cols)
    {
        f32 *data = (f32 *)PushSize(&MemoryArena, rows * cols * sizeof(f32));
        M o(data, rows, cols);
        for (u32 i = 0; i < rows * cols; ++i)
        {
            o.data[i] = 0.0f;
        }
        return o;
    }

    static M rand(u32 rows, u32 cols)
    {
        f32 *data = (f32 *)PushSize(&MemoryArena, rows * cols * sizeof(f32));
        M o(data, rows, cols);
        for (u32 i = 0; i < rows * cols; ++i)
        {
            #if ONCOMPUTER
            o.data[i] = uniform_rand();
            #else
            o.data[i] = uniform_rand();
            #endif
        }
        return o;
    }

    void shape()
    {
        std::printf("Shape: (%u, %u)\n", rows, cols);
    }

    void print()
    {
        this->shape();
        for (u32 i = 0; i < rows; ++i)
        {
            for (u32 j = 0; j < cols; ++j)
            {
                std::printf("%f ", data[i * cols + j]);
            }
            std::printf("\n");
        }
    }

    void transpose()
    {
        if (cols != 1)
        {
            u32 aux = cols;
            cols = rows;
            rows = aux;
        }
    }

    f32 &operator[](u32 idx)
    {
        return data[idx];
    }

    M operator-(M B)
    {
        M out = M::zeros(rows, cols);
        for (u32 i = 0; i < rows * cols; ++i)
        {
            out.data[i] = data[i] - B.data[i];
        }
        return out;
    }

    M &operator*=(f32 v)
    {
        for (u32 i = 0; i < rows * cols; ++i)
        {
            data[i] = data[i] * v;
        }
        return *this;
    }

    M &operator+=(M v)
    {
        assert(this->cols == v.cols && this->rows == v.rows);
        for (u32 i = 0; i < rows * cols; ++i)
        {
            data[i] = data[i] + v.data[i];
        }
        return *this;
    }

    M &operator-=(M v)
    {
        assert(this->cols == v.cols && this->rows == v.rows);
        for (u32 i = 0; i < rows * cols; ++i)
        {
            data[i] -= v.data[i];
        }
        return *this;
    }

    // M operator-()
    // {
    //     M out = M::zeros(rows, cols);
    //     for (u32 i = 0; i < rows * cols; ++i)
    //     {
    //         out.data[i] = -data[i];
    //     }
    //     return out;
    // }

    M operator/(M B)
    {
        M out = M::zeros(rows, cols);
        for (u32 i = 0; i < rows * cols; ++i)
        {
            out.data[i] = data[i] * (1.0f / B.data[i]);
        }
        return out;
    }

    M operator*(M B)
    {
        assert(B.rows == rows && B.cols == cols);
        M out = M::zeros(rows, cols);
        for (u32 i = 0; i < rows * cols; ++i)
        {
            out.data[i] = data[i] * B.data[i];
        }
        return out;
    }

    M operator+(M B)
    {
        assert(B.rows == rows && B.cols == cols);
        M out = M::zeros(rows, cols);
        for (u32 i = 0; i < rows * cols; ++i)
        {
            out.data[i] = data[i] + B.data[i];
        }
        return out;
    }

    M operator*(float B)
    {
        M out = M::zeros(rows, cols);
        for (u32 i = 0; i < rows * cols; ++i)
        {
            out.data[i] = data[i] * B;
        }
        return out;
    }

    void add(M b)
    {
        assert(rows == b.rows && cols == b.cols);
        for (u32 i = 0; i < b.rows * b.cols; ++i)
        {
            this->data[i] += b.data[i];
        }
    }

    void sub(M b)
    {
        assert(rows == b.rows && cols == b.cols);
        for (u32 i = 0; i < b.rows * b.cols; ++i)
        {
            this->data[i] -= b.data[i];
        }
    }

    float sum()
    {
        f32 s = 0.0f;
        for (u32 i = 0; i < rows * cols; ++i)
        {
            s += data[i];
        }
        return s;
    }

    float mean()
    {
        return (this->sum() / cols);
    }

    f32 std(){
        f32 sum = 0.0f;
        u32 n = rows * cols;
        for (u32 i = 0; i < n; ++i)
        {
            sum += data[i];
        }

        f32 mean = sum / n;
        f32 sd = 0.0f;
        for(u32 i = 0; i < n; i++) {
            sd += pow(data[i] - mean, 2);
        }

        return sqrt(sd / (f32)n);
    }

    u32 argmax()
    {
        u32 imax = 0;
        for (u32 i = 0; i < rows * cols; ++i)
        {
            if (data[i] > data[imax])
            {
                imax = i;
            }
        }

        return imax;
    }

    M square()
    {
        M out = M::zeros(rows, cols);
        for (u32 i = 0; i < rows * cols; ++i)
        {
            out.data[i] = data[i] * data[i];
        }
        return out;
    }

    static M MatMul(M A, M B)
    {
        assert(A.cols == B.rows);
        M Out = M::zeros(A.rows, B.cols);
        for (u32 i = 0; i < A.rows; i++)
        {
            for (u32 j = 0; j < B.cols; j++)
            {
                Out.data[i * B.cols + j] = 0;
                for (u32 k = 0; k < A.cols; k++)
                {
                    Out.data[i * B.cols + j] += A.data[i * A.cols + k] * B.data[k * B.cols + j];
                }
            }
        }
        return Out;
    }

    static void MatMul_(M A, M B, M &Out)
    {
        for (u32 i = 0; i < A.rows; i++)
        {
            for (u32 j = 0; j < B.cols; j++)
            {
                Out.data[i * B.cols + j] = 0;
                for (u32 k = 0; k < A.cols; k++)
                {
                    Out.data[i * B.cols + j] += A.data[i * A.cols + k] * B.data[k * B.cols + j];
                }
            }
        }
    }

    #if ONCOMPUTER
    void store(const char* filename){
        std::ofstream outfile(filename);

        outfile << this->rows;
        outfile << "\t";
        outfile << this->cols;
        outfile << "\t";
        for(u32 i=0;i<this->rows*this->cols;++i){
            outfile << this->data[i];
            outfile << " ";
        }
        outfile.close();
    }

    void load(const char* filename){
        std::ifstream infile(filename);
        infile >> this->rows;
        infile >> this->cols;
        for(u32 i=0;i<this->rows * this->cols;++i){
            infile >> this->data[i];
        }
        infile.close();
    }
    #endif
};

struct M3{
    f32* data;  
    u32 d1;
    u32 d2;
    u32 d3;

    M3() = default;

    M3(f32* data, u32 d1, u32 d2, u32 d3): data(data), d1(d1), d2(d2), d3(d3){

    }
    static M3 rand(u32 d1, u32 d2, u32 d3)
    {
        f32 *data = (f32 *)PushSize(&MemoryArena, d1 * d2 * d3 * sizeof(f32));
        M3 o(data, d1, d2, d3);
        for (u32 i = 0; i < d1*d2*d3; ++i)
        {
            #if ONCOMPUTER
            o.data[i] = uniform_rand();
            #else
            o.data[i] = uniform_rand();
            #endif       
        }
        return o;
    }

    static M3 zeros(u32 d1, u32 d2, u32 d3)
    {
        f32 *data = (f32 *)PushSize(&MemoryArena, d1 * d2 * d3 * sizeof(f32));
        M3 o(data, d1, d2, d3);
        for (u32 i = 0; i < d1*d2*d3; ++i)
        {
            o.data[i] = 0.0f;
        }
        return o;
    }

    f32 operator()(u32 i, u32 j, u32 k){
        u32 index = i * (d3*d2) + j*d3 + k;
        return data[index];
    }

    void set(u32 i, u32 j, u32 k, f32 val){
        u32 index = i * (d3*d2) + j*d3 + k;
        data[index] = val;
    }

    M3 operator*(M3 B)
    {
        assert(B.d1 == d1 && B.d2 == d2 && B.d3 == d3);
        M3 out = M3::zeros(d1,d2,d3);
        for (u32 i = 0; i < d1 * d2 * d3; ++i){
            out.data[i] = data[i] * B.data[i];
        }
        return out;
    }

    M3 operator+(M3 B)
    {
        assert(B.d1 == d1 && B.d2 == d2 && B.d3 == d3);
        M3 out = M3::zeros(d1,d2,d3);
        for (u32 i = 0; i < d1 * d2 * d3; ++i){
            out.data[i] = data[i] + B.data[i];
        }
        return out;
    }

    M3 operator*(f32 B)
    {
        M3 out = M3::zeros(d1,d2,d3);
        for (u32 i = 0; i < d1 * d2 * d3; ++i){
            out.data[i] = data[i] * B;
        }
        return out;
    }

    M flatten(){
        return M(data, 1, d1*d2*d3);
    }

    void shape(){
        std::printf("SHAPE: %lu, %lu, %lu\n",d1,d2,d3 );
    }

    void print(){
        this->shape();
        for(u32 i=0;i<d1;++i){
            std::printf("[\n");
            for(u32 j=0;j<d2;++j){
                std::printf("[");
                for(u32 k=0;k<d3;++k){
                    u32 index = i * (d3*d2) + j*d3 + k;
                    std::printf("%.9f, ",data[index]);
                }   
                std::printf("],\n");
            }
            std::printf("],\n");
        }
        std::printf("\n");
    }

    M operator()(u32 i){
        u32 index = i * (d2*d3);

        M o = M(data+index, d2,d3);
        return o;
    }
    
    f32 std(){
        f32 sum = 0.0f;
        u32 n = d1 * d2 * d3;
        for (u32 i = 0; i < d1 * d2 * d3; ++i)
        {
            sum += data[i];
        }

        f32 mean = sum / n;
        f32 sd = 0.0f;
        for(u32 i = 0; i < d1 * d2 * d3; i++) {
            sd += pow(data[i] - mean, 2);
        }

        return sqrt(sd / (f32)n);
    }

    #if ONCOMPUTER

    void store(const char* filename){
        std::ofstream outfile(filename);

        outfile << this->d1;
        outfile << "\t";
        outfile << this->d2;
        outfile << "\t";
        outfile << this->d3;
        outfile << "\t";
        for(u32 i=0;i<this->d1*this->d2*this->d3;++i){
            outfile << this->data[i];
            outfile << " ";
        }
        outfile.close();
    }

    void load(const char* filename){
        std::ifstream infile(filename);
        infile >> this->d1;
        infile >> this->d2;
        infile >> this->d3;
        for(u32 i=0;i<this->d1 * this->d2 * this->d3;++i){
            infile >> this->data[i];
        }
        infile.close();
    }
    #endif

};
struct M4{
    f32* data;  
    u32 d1;
    u32 d2;
    u32 d3;
    u32 d4;

    M4() = default;
    M4(f32* data, u32 d1, u32 d2, u32 d3, u32 d4): data(data), d1(d1), d2(d2), d3(d3), d4(d4){}
    static M4 ones(u32 d1, u32 d2, u32 d3, u32 d4)
    {
        f32 *data = (f32 *)PushSize(&MemoryArena, d1*d2*d3*d4 * sizeof(f32));
        M4 o(data, d1, d2, d3, d4);
        for (u32 i = 0; i < d1*d2*d3*d4; ++i)
        {
            o.data[i] = 1.0f;
        }
        return o;
    }

    static M4 zeros(u32 d1, u32 d2, u32 d3, u32 d4)
    {
        f32 *data = (f32 *)PushSize(&MemoryArena, d1*d2*d3*d4 * sizeof(f32));
        M4 o(data, d1, d2, d3, d4);
        for (u32 i = 0; i < d1*d2*d3*d4; ++i)
        {
            o.data[i] = 0.0f;
        }
        return o;
    }

    static M4 rand(u32 d1, u32 d2, u32 d3, u32 d4)
    {
        f32 *data = (f32 *)PushSize(&MemoryArena, d1*d2*d3*d4 * sizeof(f32));
        M4 o(data, d1, d2, d3, d4);
        for (u32 i = 0; i < d1*d2*d3*d4; ++i)
        {
            #if ONCOMPUTER
            o.data[i] = uniform_rand();
            #else
            //o.data[i] = uniform_rand();
            #endif        
        }
        return o;
    }

    M4 &operator-=(M4 v)
    {
        assert(this->d1 == v.d1 && this->d2 == v.d2 && this->d3 == v.d3 && this->d4 == v.d4);
        for (u32 i = 0; i < d1 * d2*d3*d4; ++i)
        {
            data[i] -= v.data[i];
        }
        return *this;
    }

    M4 operator*(f32 lr)
    {
        M4 output = M4::zeros(d1,d2,d3,d4);
        for (u32 i = 0; i < d1 * d2*d3*d4; ++i)
        {
            output.data[i] = data[i] * lr;
        }
        return output;
    }

    f32 operator()(u32 l, u32 k, u32 j, u32 i){
        //assert(i < d1 && j < d2 && k < d3 && l < d4);
        u32 index = l * (d4*d2*d3) + k * (d4*d3) + j*d4 + i;
        return data[index];
    }

    void set(u32 l, u32 k, u32 j, u32 i, f32 val){
        //assert(i < d1 && j < d2 && k < d3 && l < d4);

        u32 index = l * (d4*d2*d3) + k * (d4*d3) + j*d4 + i;
        data[index] = val;
    }

    void add(u32 l, u32 k, u32 j, u32 i, f32 val){
        u32 index = l * (d4*d2*d3) + k * (d4*d3) + j*d4 + i;
        data[index] += val;
    }

    M operator()(u32 i, u32 j){
        u32 index = i * (d2*d3*d4) + j*(d3*d4);
        M o = M(data+index, d3, d4);
        return o;
    }

    M3 operator()(u32 i){
        u32 index = i * (d4*d2*d3);

        M3 o = M3(data+index, d2, d3,d4);
        return o;
    }

    M4 operator+(M4 B)
    {
        M4 out = M4::zeros(d1, d2, d3, d4);
        for (u32 i = 0; i < d1 * d2 * d3 * d4; ++i)
        {
            out.data[i] = data[i] + B.data[i];
        }
        return out;
    }

    f32 std(){
        f32 sum = 0.0f;
        u32 n = d1 * d2 * d3 * d4;
        for (u32 i = 0; i < d1 * d2 * d3 * d4; ++i)
        {
            sum += data[i];
        }

        f32 mean = sum / n;
        f32 sd = 0.0f;
        for(u32 i = 0; i < d1 * d2 * d3 * d4; i++) {
            sd += pow(data[i] - mean, 2);
        }

        return sqrt(sd / (f32)n);
    }

    void print(){
        std::printf("SHAPE: %lu, %lu, %lu, %lu\n",d1,d2,d3,d4 );

        for(u32 i=0;i<d1;++i){
            std::printf("[\n");
            for(u32 j=0;j<d2;++j){
                std::printf("[\n");
                for(u32 k=0;k<d3;++k){
                    std::printf("[");
                    for(u32 l=0;l<d4;++l){
                        u32 index = i * (d4*d2*d3) + j * (d4*d3) + k*d4 + l;
                        //u32 index = i * (d4*d2*d3) + j * (d4*d3) + k*d4 + l;
                        std::printf("%f, ",data[index]);
                    }   
                    std::printf("],\n");
                }   
                std::printf("],\n");
            }
            std::printf("]\n");
        }
    }
    #if ONCOMPUTER

    void store(const char* filename){
        std::ofstream outfile(filename);

        outfile << this->d1;
        outfile << "\t";
        outfile << this->d2;
        outfile << "\t";
        outfile << this->d3;
        outfile << "\t";
        outfile << this->d4;
        outfile << "\t";
        for(u32 i=0;i<this->d1*this->d2*this->d3*this->d4;++i){
            outfile << this->data[i];
            outfile << " ";
        }
        outfile.close();
    }

    void load(const char* filename){
        std::ifstream infile(filename);
        infile >> this->d1;
        infile >> this->d2;
        infile >> this->d3;
        infile >> this->d4;
        for(u32 i=0;i<this->d1 * this->d2 * this->d3*this->d4;++i){
            infile >> this->data[i];
        }
        infile.close();
    }

    #endif
};

M4 operator*(f32 scalar, M4 rhs)
{
    M4 output = M4::zeros(rhs.d1,rhs.d2,rhs.d3,rhs.d4);
    for (u32 i = 0; i < rhs.d1 * rhs.d2*rhs.d3*rhs.d4; ++i)
    {
        output.data[i] = rhs.data[i] * scalar;
    }
    return output;
}

M4 operator-(M4 lhs, M4 rhs)
{
    //assert();
    M4 output = M4::zeros(rhs.d1,rhs.d2,rhs.d3,rhs.d4);
    for (u32 i = 0; i < rhs.d1 * rhs.d2*rhs.d3*rhs.d4; ++i)
    {
        output.data[i] = lhs.data[i] - rhs.data[i];
    }
    return output;
}

M3 operator*(f32 scalar, M3 rhs)
{
    M3 output = M3::zeros(rhs.d1,rhs.d2,rhs.d3);
    for (u32 i = 0; i < rhs.d1 * rhs.d2*rhs.d3; ++i)
    {
        output.data[i] = rhs.data[i] * scalar;
    }
    return output;
}

M3 operator-(M3 lhs, M3 rhs)
{
    //assert();
    M3 output = M3::zeros(rhs.d1,rhs.d2,rhs.d3);
    for (u32 i = 0; i < rhs.d1 * rhs.d2*rhs.d3; ++i)
    {
        output.data[i] = lhs.data[i] - rhs.data[i];
    }
    return output;
}


M operator*(f32 scalar, M rhs)
{
    M output = M::zeros(rhs.rows,rhs.cols);
    for (u32 i = 0; i < rhs.rows * rhs.cols; ++i)
    {
        output.data[i] = rhs.data[i] * scalar;
    }
    return output;
}


#define MT_N 624
#define MT_M 397

typedef struct {
    unsigned int mt[MT_N];
    int index;
} mt_state;
mt_state mystate;
void mt_seed(mt_state *state, unsigned int seed) {
    state->mt[0] = seed;
    for (int i = 1; i < MT_N; i++) {
        state->mt[i] = 1812433253 * (state->mt[i-1] ^ (state->mt[i-1] >> 30)) + i;
    }
    state->index = MT_N;
}

void mt_twist(mt_state *state) {
    static const unsigned int MAG01[2] = {0, 0x9908b0df};
    unsigned int y;

    for (int i = 0; i < MT_N; i++) {
        y = (state->mt[i] & 0x80000000) | (state->mt[(i+1) % MT_N] & 0x7fffffff);
        state->mt[i] = state->mt[(i + MT_M) % MT_N] ^ (y >> 1) ^ MAG01[y & 0x1];
    }
    state->index = 0;
}

unsigned int mt_rand(mt_state *state) {
    if (state->index >= MT_N) {
        mt_twist(state);
    }
    unsigned int y = state->mt[state->index++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= (y >> 18);
    return y;
}

double mt_rand_double(mt_state *state) {
    return (double)mt_rand(state) / (double)0xffffffff;
}

double mt_rand_xavier(mt_state *state, f32 low, f32 high) {
    return low + (mt_rand_double(state) * (high - low));
}

void XavierInitialization(M4 data, f32 fan_in, f32 fan_out, f32 min_val, f32 max_val){
    // Calculate the scaling factor based on the fan-in and fan-out
    //float scale = sqrt(6.0 / (fan_in + fan_out));
    
    // Initialize the weights with random values from a uniform distribution
    // std::random_device rd;
    // std::mt19937 gen(0);
    // std::uniform_real_distribution<float> dist(min_val, max_val);
    f32 scale = (sqrt(6.0) / sqrt(fan_out + fan_in));

    for(u32 i=0;i<data.d1 * data.d2 * data.d3 * data.d4;++i){
        data.data[i] = mt_rand_xavier(&mystate, -scale, scale);
    }
}

void XavierInitialization(M data, f32 fan_in, f32 fan_out, f32 min_val, f32 max_val){
    // Calculate the scaling factor based on the fan-in and fan-out
    //float scale = sqrtf(6.0 / (fan_in + fan_out));
    
    // Initialize the weights with random values from a uniform distribution
    // std::random_device rd;
    // std::mt19937 gen(0);
    // std::uniform_real_distribution<float> dist(min_val, max_val);
    f32 scale = (sqrt(6.0) / sqrt(fan_out + fan_in));
    for(u32 i=0;i<data.rows * data.cols;++i){
        data.data[i] = mt_rand_xavier(&mystate, -scale, scale);
    }
}

f32 clip_by_value(f32 val, f32 min, f32 max){
    if (val > max){
        return max;
    } 
    if(val < min){
        return min;
    }

    return val;
}


// This function computes the Cross Entropy loss between two sets of outputs, y and y_hat.
f32 CrossEntropy(M y, M y_hat)
{
    // Initialize the loss to zero.
    f32 loss = 0;
    
    // Loop through each output in the target y and predicted y_hat.
    for (u32 i = 0; i < y.cols; i++)
    {
        // The 1e-9 is added to avoid log(0) which would result in a NaN (Not a Number) value.
        loss += y[i] * log(clip_by_value(y_hat[i],1e-7, 1.0f - 1e-7 ) );
    }

    // // Apply L2 regularization
    // f32 regularization = 0;
    // f32 lambda = 1e-3;
    // for (u32 i = 0; i < y.cols; i++)
    // {
    //     regularization += lambda * (y_hat[i] * y_hat[i]);  // L2 regularization term
    // }
    
    // // Add the L2 regularization term to the loss
    // loss += 0.5 * regularization;
    
    // Return the negative of the loss.
    return -loss;
}

// This function computes the derivative of the Cross Entropy loss with respect to the predicted outputs, y_hat.
M CrossEntropyPrime(M y, M y_hat)
{  
    // Initialize an empty matrix with the same shape as the inputs, y and y_hat.
    M out = M::zeros(y.rows, y.cols);
    for (u32 i = 0; i < y.cols; i++)
    {
        // The 1e-15 is added to avoid division by zero.
        out.data[i] = -y[i] / (clip_by_value(y_hat[i],1e-7, 1.0f - 1e-7 ));
    }

    // f32 lambda = 1e-3;
    // // Apply L2 regularization to the loss
    // for (u32 i = 0; i < y.cols; i++)
    // {
    //     out.data[i] += lambda * y_hat[i];  // Add the L2 regularization term
    // }
    
    return out;
    // return -(y / y_hat);
}

f32 BinaryCrossEntropyLoss(M y, M y_hat)
{
    f32 loss = 0;
    
    for (u32 i = 0; i < y.cols; i++)
    {
        // Add a small value to avoid taking the log of zero.
        const f32 eps = 1e-7;
        y_hat[i] = clip_by_value(y_hat[i],1e-7, 1.0f - 1e-7 );
        loss += -(y[i] * log(y_hat[i]) + (1 - y[i]) * log(1 - y_hat[i]));
    }
    
    return loss / y.cols;
}

M BinaryCrossEntropyPrime(M y, M y_hat)
{
    // Initialize an empty matrix with the same shape as the inputs, y and y_hat.
    M out = M::zeros(y.rows, y.cols);
    
    for (u32 i = 0; i < y.cols; i++)
    {
        // Add a small value to avoid division by zero.
        const double eps = 1e-7;
        y_hat[i] = clip_by_value(y_hat[i],1e-7, 1.0f - 1e-7 );

        out.data[i] = (y_hat[i] - y[i]) / (y_hat[i] * (1 - y_hat[i]));
    }
    
    return out;
}

// Computes the derivative of the mean squared error (MSE) loss with respect to 'y_hat'
inline M MsePrime(M y, M y_hat)
{
    return y_hat - y;
}

// Computes the mean squared error (MSE) loss between two matrices 'y' and 'y_hat'
inline f32 Mse(M y, M y_hat)
{
    return (y - y_hat).square().mean();
}

// Softmax function calculates the probability of each output for a given input matrix
M Softmax(M X)
{
    f32 sum = 0.0f;
    f32 max = X.data[X.argmax()];
    M out = M::zeros(X.rows, X.cols);
    for (u32 i = 0; i < X.cols; ++i)
    {
        sum += exp(X.data[i] - max);
    }
    for (u32 i = 0; i < X.cols; ++i)
    {
        // subtract max to avoid NaN/+inf errors
        out.data[i] = exp(X.data[i] - max) / sum;
    }
    return out;
}

// SoftmaxPrime function calculates the derivative of the softmax function
M SoftmaxPrime(M X)
{
    M Out = M::zeros(X.cols, X.cols);
    // calculate the derivative of the softmax function
    for (u32 i = 0; i < X.cols; ++i)
    {
        for (u32 j = 0; j < X.cols; ++j)
        {
            if (i == j)
            {
                Out.data[i * X.cols + j] = X.data[i] * (1.0f - X.data[i]);
            }
            else
            {
                Out.data[i * X.cols + j] = -X.data[i] * X.data[j];
            }
        }
    }
    return Out;
}

M Sigmoid(M X)
{
    u32 Cols = X.cols;
    u32 Rows = X.rows;
    for (u32 i = 0; i < Rows * Cols; ++i)
    {
        X.data[i] = 1.0f / (1.0f + exp(-X.data[i]));
    }
    return X;
}

M SigmoidPrime(M X)
{
    M Out = M::zeros(X.rows, X.cols);
    for (u32 i = 0; i < X.rows * X.cols; ++i)
    {
        Out.data[i] = X.data[i] * (1.0f - X.data[i]);
    }
    return Out;
}

M3 Sigmoid(M3 X)
{
    for (u32 i = 0; i < X.d1 * X.d2 * X.d3; ++i)
    {
        X.data[i] = 1.0f / (1.0f + exp(-X.data[i]));
    }
    return X;
}

M3 SigmoidPrime(M3 X)
{
    M3 Out = M3::zeros(X.d1, X.d2, X.d3);
    for (u32 i = 0; i < X.d1 * X.d2 * X.d3; ++i)
    {
        Out.data[i] = X.data[i] * (1.0f - X.data[i]);
    }
    return Out;
}

M Relu(M X)
{
    M Out = M::zeros(X.rows, X.cols);
    for (u32 i = 0; i < X.rows * X.cols; ++i)
    {
        Out.data[i] = X.data[i] > 0.0f ? X.data[i] : 0.0f;
    }
    return Out;
}

M ReluPrime(M X)
{
    M Out = M::zeros(X.rows, X.cols);
    for (u32 i = 0; i < X.rows * X.cols; ++i)
    {
        Out.data[i] = X.data[i] > 0.0f ? 1.0 : 0.0f;
    }
    return Out;
}

M3 Relu(M3 X)
{
    M3 Out = M3::zeros(X.d1, X.d2, X.d3);
    for (u32 i = 0; i < X.d1 * X.d2 * X.d3; ++i)
    {
        Out.data[i] = X.data[i] > 0.0f ? X.data[i] : 0.0f;
    }
    return Out;
}

M3 ReluPrime(M3 X)
{
    M3 Out = M3::zeros(X.d1, X.d2, X.d3);
    for (u32 i = 0; i < X.d1 * X.d2 * X.d3; ++i)
    {
        Out.data[i] = X.data[i] > 0.0f ? 1.0 : 0.0f;
    }
    return Out;
}

M Tanh(M X)
{
    M Out = M::zeros(X.rows, X.cols);
    for (u32 i = 0; i < X.rows * X.cols; ++i)
    {
        Out.data[i] = tanh(X.data[i]);
    }
    return Out;
}

M TanhPrime(M X)
{
    M Out = M::zeros(X.rows, X.cols);
    for (u32 i = 0; i < X.rows * X.cols; ++i)
    {
        Out.data[i] = (1 - X.data[i] * X.data[i]);
    }
    return Out;
}

M3 Tanh(M3 X)
{
    M3 Out = M3::zeros(X.d1, X.d2, X.d3);
    for (u32 i = 0; i <  X.d1*X.d2*X.d3; ++i)
    {
        Out.data[i] = tanh(X.data[i]);
    }
    return Out;
}

M3 TanhPrime(M3 X)
{
    M3 Out = M3::zeros(X.d1, X.d2, X.d3);
    for (u32 i = 0; i < X.d1*X.d2*X.d3; ++i)
    {
        Out.data[i] = (1 - X.data[i] * X.data[i]);
    }
    return Out;
}


void setRandomUniform(double low, double high)
{
    #if ONCOMPUTER
        distribution = std::uniform_real_distribution<double>(low, high);
    #else
        randLowVal = low;
        randHighVal = high;
    #endif
}

// Set Glorot uniform distribution for weights random intialization
void setGlorotUniform(u32 in, u32 out)
{
    double scale = sqrt(6.0f / ((f32)in + (f32)out));
    #if ONCOMPUTER
    randLowVal = -scale;
    randHighVal = scale;  
    #else
    randLowVal = -scale;
    randHighVal = scale;
    #endif
    
}

struct Layer
{
    // Weight and bias matrices
    M w;
    M b;

    // Gradient and biases weights matrices
    M dw;
    M db;

    M vdw;
    M vdb;

    // Factory function to create a new layer object
    static Layer *create(u32 input_size, u32 output_size)
    {
        // Allocate memory for the layer on the memory arena
        Layer *l = (Layer *)PushSize(&MemoryArena, sizeof(Layer));

#if GLOROT_UNIFORM
        // Initialize the weight matrix with Glorot uniform distribution
        //setGlorotUniform(input_size, output_size);
#else
        //setRandomUniform(-0.5, 0.5);
#endif
        // Initialize the weight and bias matrices with random values
        l->w = M::rand(input_size, output_size);
#if ONCOMPUTER
        XavierInitialization(l->w, input_size, output_size, -1.0f, 1.0f);
#endif
        l->b = M::zeros(1, output_size);

        // Initialize the gradient matrices to zero
        l->dw = M::zeros(input_size, output_size);
        l->db = M::zeros(1, output_size);

        // // Initialize the momentum matrices to zero
        l->vdw = M::zeros(input_size, output_size);
        l->vdb = M::zeros(1, output_size);
        return l;
    }

    // Forward propagation function
    M forward(M x)
    {
        // Check that the input matrix has the correct number of columns, this saves us from stupid bugs.
        assert(x.cols == w.rows);
        M Out = M::zeros(x.rows, w.cols);
        for (u32 i = 0; i < w.cols; ++i)
        {
            f32 accum = 0;
            for (u32 j = 0; j < x.cols; ++j)
            {
                accum += x.data[j] * w.data[j * w.cols + i];
                // maybe slightly improve the performance ?
                // accum += x.data[j] * w.data[idx];
                // idx += w.cols;
            }
            Out.data[i] = accum + b.data[i];
        }
        return Out;
    }

    // Backward propagation function
    M backward(M grad)
    {
        // Check that the gradient matrix has the correct number of columns
        assert(grad.cols == w.cols);
        M out = M::zeros(1, w.rows);
        for (u32 k = 0; k < w.rows; ++k)
        {
            //f32 accum = 0;
            for (u32 l = 0; l < grad.cols; ++l)
            {
                //accum += grad.data[l] * w.data[k * w.cols + l];
                out.data[k] += grad.data[l] * w.data[k * w.cols + l];
            }
            //out.data[k] = accum;
        }
        return out;
    }

    // Function to reset the gradient matrices to zero
    void resetGradients()
    {
        this->dw = M::zeros(this->dw.rows, this->dw.cols);
        this->db = M::zeros(1, this->db.cols);
    }

    // Function to calculate the weights matrix gradients, using the input gradient and the activation function of the current layer
    M getDelta(M grads, M a)
    {
        assert(w.rows == a.cols);
        M out = M::zeros(w.rows, w.cols);
        for (u32 i = 0; i < w.rows; ++i)
        {
            for (u32 j = 0; j < w.cols; ++j)
            {
                f32 g = grads.data[j];
                out.data[i * w.cols + j] = g * a.data[i];
            }
        }
        return out;
    }

    void setDelta(M grads, M a)
    {
        assert(w.rows == a.cols);
        for (u32 i = 0; i < w.rows; ++i)
        {
            for (u32 j = 0; j < w.cols; ++j)
            {
                f32 g = grads.data[j];
                dw.data[i * w.cols + j] = g * a.data[i];
            }
            db.data[i] = grads.data[i];
        }

    }

    void UpdateWeights(f32 lr, u32 batchsize = 1)
    {
        // scale the learning rate by the batch size. By default, the batch size is set to 1.
        lr = lr * (1.0f / (f32)batchsize);

        // // MOMENTUM
        f32 momentum = 0.9f;
        vdw = (vdw * momentum) + (dw * 0.1f);
        vdb = (vdb * momentum) + (db * 0.1f);

        for (u32 i = 0; i < w.rows; ++i)
        {
            for (u32 j = 0; j < w.cols; ++j)
            {
                // Update weights
                //w.data[i * w.cols + j] -= lr * this->dw[i * this->dw.cols + j];
                w.data[i * w.cols + j] -= lr * this->vdw[i * this->vdw.cols + j];
            }
            // Update bias
            //b.data[i] -= lr * this->db[i];
            b.data[i] -= lr * this->vdb.data[i];
        }
    }

    void UpdateWeights(M grads, M a, f32 lr)
    {
        // MOMENTUM
        vdw = vdw * 0.9f + dw * (1.0f-0.9f);  
        vdb = vdb * 0.9f + db * (1.0f-0.9f);  

        for (u32 i = 0; i < w.rows; ++i)
        {
            u32 index = i * w.cols;
            for (u32 j = 0; j < w.cols; ++j)
            {
                //w.data[i * w.cols + j] -= lr * this->dw[i * this->dw.cols + j];
                dw.data[index + j] = grads.data[j] * a.data[i];
                // update momentum vel 
                vdw.data[index + j] = vdw.data[index + j] * 0.9f + dw.data[index + j] * (1.0f-0.9f);
                // Update weights
                w.data[index + j] -= lr * this->vdw.data[index + j];
            }
            db.data[i] = grads.data[i];
            // update momentum vel
            vdb.data[i] = vdb.data[i] * 0.9f + db.data[i] * (1.0f-0.9f);  
            // Update bias
            b.data[i] -= lr * this->vdb.data[i];
        }
    }
};

struct Size3D{
    i32 h;
    i32 w;
    i32 c;
};

Size3D get_output_from_kernel(u32 ih, u32 iw, u32 kh, u32 kw, u32 kc){
    Size3D output;
    output.c = kc;
    output.h = ih - kh + 1;
    output.w = iw - kw + 1;
    return output;
}

struct Conv2D{
    Size3D input_size;
    Size3D output_size;
    u32 numKernels;
    u32 stride;

    M4 kernels;
    M4 dkernels;

    M bias;
    M db;
    
    M4 vdkernels;
    M vdb;
    //M kernels[outputChannels][inputChannels];
    //M dkernels[outputChannels][inputChannels];

    // Factory function to create a new layer object
    static Conv2D *Create(u32 h, u32 w, u32 c, u32 kh, u32 kw, u32 kc)
    {
        // Allocate memory for the layer on the memory arena
        Conv2D *l = (Conv2D *)PushSize(&MemoryArena, sizeof(Conv2D));

        l->input_size.c = c;
        l->input_size.h = h;
        l->input_size.w = w;

        l->numKernels = kc;
        
        l->output_size = get_output_from_kernel(h, w, kh, kw, kc);
        //setGlorotUniform(kh * kw * c, kh * kw * kc);

        l->kernels = M4::zeros(kh, kw, c, kc);
        XavierInitialization(l->kernels,kh * kw * c, kh * kw * kc, -1.0f, 1.0f);

        l->dkernels = M4::zeros(kh, kw, c, kc);
        l->vdkernels = M4::zeros(kh, kw, c, kc);
        l->vdb = M::zeros(1, kc);
        
        l->bias = M::zeros(1, kc);
        l->db = M::zeros(1, kc);

        l->stride = 1;
        return l;
    }

    // CONVOLVE2D CORRECT (COMPARED WITH KERAS)
    M3 convolve2D(M3 input){
        //std::printf("%u, %u, %u   =   %u, %u, %u,\n",    input.d1, input.d2, input.d3, input_size.h, input_size.h, input_size.c);
        assert(input_size.c > 0 && input.d3 == input_size.c  && input.d1 == input_size.h && input.d2 == input_size.w && "Input image mismatch layer");

        // TODO: add padding
        const Size3D output_size = this->output_size;
        const u32 stride = 1;

        const u32 input_channels = this->input_size.c;
        const u32 output_channels = this->output_size.c;

        M3 output = M3::zeros(output_size.h, output_size.w, output_size.c);
        for(u32 j=0;j<output_channels;++j){
            for (u32 h=0;h<output_size.h;++h) {
                for( u32 w=0;w<output_size.w;++w){
                    f32 s = 0;
                    for(u32 i=0;i<input_channels;++i){
                        u32 w_start = w * stride;
                        u32 h_start = h * stride;
                        for(u32 l=0;l<kernels.d1;++l) {
                            for(u32 m=0;m<kernels.d2;++m) {
                                // TODO: should check for borders?
                                    // To conv: perform element-wise multiplication of the slices of the prev (input) matrix
                                    // then sum up all the values from all channels, add the bias to the sum of convolutions for each output channel
                                s += input(l+h_start,m+w_start,i) * kernels(l,m,i,j);
                            }
                        }
                        s += bias.data[j]; 
                    }
                    output.set(h,w,j, s);
                    //output[j].data[(output_size.w * h) + w] = s;
                }
            }
        }
        return output;
    }

    Size3D getInputSize() const {
        return this->input_size;
    }

    Size3D getOutputSize() const {
        return this->output_size;
    }

    u32 getLinearFlattenedSize() const {
        return this->output_size.c * this->output_size.h * this->output_size.w;
    }

    M3 backward_conv(M3 X, M3 dh){
        //std::printf("%d=%d, %d=%d, %d=%d, %d=%d\n", dh.d1, this->output_size.c, dh.d2, this->output_size.h, dh.d3, this->output_size.w, this->numKernels, dh.d1);
        assert(dh.d3 == this->numKernels && this->output_size.c == dh.d3 && dh.d1 == this->output_size.h && dh.d2 == this->output_size.w && "BACKWARD CONV ASSERT");
        assert(this->input_size.c == X.d3);
        const u32 input_h = this->input_size.h;
        const u32 input_w = this->input_size.w;

        const u32 output_w = this->output_size.w;
        const u32 output_h = this->output_size.h;

        const u32 k_h = input_h - output_h + 1;
        const u32 k_w = input_w - output_w + 1;

        M3 dx = M3::zeros(input_h,input_w, this->input_size.c);

        assert(input_h == X.d1 && input_w == X.d2 && X.d3 == this->input_size.c);

        for(u32 p=0;p<this->numKernels;++p){
            f32 bias = 0.0f;
            for (int c=0;c<this->input_size.c;++c){
                bias = 0.0;
                for (int i = 0; i < output_h; i++) {
                    for (int j = 0; j < output_w; j++) {
                        bias += dh(i,j,p);
                        for (int k = 0; k < k_h; k++) {
                            for (int l = 0; l < k_w; l++) {
                                if(i+k<input_h && j+l < input_w){
                                    dx.set(i+k,j+l,c, dx(i+k,j+l,c) + (dh(i, j, p) * kernels(k,l,c,p)));
                                }

                                //std::printf("dw[%u,%u] = dh[%u,%u] * f[%u,%u] = %f\n", k,l, i,j, i+k, j+l, X(i+k,j+l,c) * dh(i,j,p));
                                dkernels.set(k,l,c,p, dkernels(k,l,c,p) + dh(i, j, p) * X(i+k, j+l, c));

                            }
                        }
                    }
                }
            }
            db.data[p] += bias;
        }

        return dx;
    }

    void resetGradients(){
        const u32 input_h = this->input_size.h;
        const u32 input_w = this->input_size.w;

        const u32 output_w = this->output_size.w;
        const u32 output_h = this->output_size.h;

        const u32 k_h = input_h - output_h + 1;
        const u32 k_w = input_w - output_w + 1;
        dkernels = M4::zeros(k_h, k_w, input_size.c, numKernels);
        db = M::zeros(1, numKernels);
    }

    void updateKernels(f32 lr, u32 batch_size = 1){
        // const u32 output_w = this->output_size.w;
        // const u32 output_h = this->output_size.h;

        // vdkernels   = vdkernels * 0.9f - lr * dkernels;
        // vdb         = vdb       * 0.9f - lr * db;
        
        // kernels = kernels + vdkernels;
        // bias = bias + vdb;
        lr = lr * (1.0f / (f32) batch_size);
        // f32 momentum = 0.9f;
        // vdkernels = vdkernels * momentum + dkernels * 0.1f;
        // vdb = vdb * momentum + db * 0.1f;

        // Update the kernels and bias using the momentum variables
        kernels -= dkernels * lr;
        bias -= db * lr;

        // kernels -= dkernels * lr;
        // bias -= db * lr;
        // for(u32 p=0;p<this->numKernels;++p){
        //     for (int c=0;c<this->input_size.c;++c){
        //         for (int i = 0; i < output_h; i++) {
        //             for (int j = 0; j < output_w; j++) {
        //                 f32 old = kernels(p,c,i,j);
        //                 f32 dk = dkernels(p,c,i,j);
              
        //                 kernels.set(p,c,i,j, old - dk * lr);
        //             }
        //         }
        //     }
        // }
    }
};

struct MaxPooling{
    Size3D kernelsize;
    Size3D inputsize;
    Size3D outputsize;
    M3 grad_input;
    M3 d;
    static MaxPooling* create(i32 h, i32 w, i32 channels, i32 kh, i32 kw){
        assert(h > 0 && channels > 0 && w > 0);
        MaxPooling *l = (MaxPooling *)PushSize(&MemoryArena, sizeof(MaxPooling));
        //setRandomUniform(-0.05, 0.05);

        l->d = M3::zeros(channels, h, w);

        l->kernelsize.h = kh;
        l->kernelsize.w = kw;

        l->inputsize.c = channels;
        l->inputsize.h = h;
        l->inputsize.w = w;
        
        int output_height = h / kh;
        int output_width =  w / kw;
        
        l->outputsize.h = output_height;
        l->outputsize.w = output_width;
        l->outputsize.c = channels;
        l->grad_input = M3::zeros(l->inputsize.h, l->inputsize.w, l->inputsize.c);

        return l;
    }

    u32 getLinearFlattenedSize(){
        return outputsize.h * outputsize.w * outputsize.c;
    }

    Size3D getOutputSize(){
        return outputsize;
    }

    void resetGradients(){
        grad_input = M3::zeros(inputsize.h, inputsize.w, inputsize.c);
    }

    M3 forward(M3 x){
        assert(x.d1 == inputsize.h && x.d2 == inputsize.w && x.d3 == inputsize.c);
        // u32 output_h = outputsize.h;
        // u32 output_w = outputsize.w;

        u32 upsampling_width = (inputsize.w / 2) * 2;
        u32 upsampling_height = (inputsize.h / 2) * 2;

        d = M3::zeros(upsampling_height, upsampling_width, inputsize.c);
        int input_h = x.d1;
        int input_w = x.d2;
        int input_c = x.d3;

        u32 stride = kernelsize.w;
        u32 kernel_size = stride;

        int output_h = (input_h - kernel_size) / stride + 1;
        int output_w = (input_w - kernel_size) / stride + 1;
        int output_c = input_c;

        M3 output = M3::zeros(output_h, output_w, output_c);
        M3 indices = M3::zeros(output_h, output_w, output_c);

        for (int k = 0; k < output_c; ++k) {
            for (int i = 0; i < output_h; ++i) {
                for (int j = 0; j < output_w; ++j) {
                    float max_val = -1e12;
                    int max_idx_i = -1, max_idx_j = -1;
                    for (int p = i*stride; p < i*stride+kernel_size; ++p) {
                        for (int q = j*stride; q < j*stride+kernel_size; ++q) {
                            if (x(p, q, k) > max_val) {
                                max_val = x(p, q, k);
                                max_idx_i = p;
                                max_idx_j = q;
                            }
                        }
                    }
                    output.set(i, j, k, max_val);
                    d.set(i, j, k,max_idx_i*input_w + max_idx_j) ;
                }
            }
        }
        
        //indices.print();
        return output;
    }

    M3 backward(M3 grad_output){
        M3 output = M3::zeros(d.d1, d.d2, d.d3);
        u32 output_h = grad_output.d1;
        u32 output_w = grad_output.d2;
        u32 kernel_size = kernelsize.h;
        u32 stride = kernel_size;

        for (u32 i = 0; i < inputsize.c; ++i) {
            for (u32 l = 0; l < output_h; ++l) {
                for (u32 p = 0; p < output_w; ++p) {
                    u32 index = d(l, p, i);

                    u32 row = index / inputsize.w;
                    u32 col = index % inputsize.w;

                    grad_input.set(row, col, i, grad_input(row,col,i) + grad_output(l, p, i));
                }
            }
        }

        return grad_input;
        //std::printf("%u, %u, %u, %u\n", grad.d1, grad.d2, grad.d3, u);
    }
};
#endif
