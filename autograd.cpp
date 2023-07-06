#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <stack>
#include <stdio.h>
#include <functional>
#include <vector>
#include <set>
#include <stack>
#include <stdio.h>

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


struct Var{
    f32 data;
    f32 grad;

    std::function<void(void)> _backward;
    std::vector<Var*> children;

    //Var(): data(0), grad(0){ }
    Var(const Var& other){
        data = other.data;
        grad = other.grad;
        _backward = other._backward;
        children = other.children;

        std::printf("COPYING\n");
    }
    Var(f32 data): data(data), grad(0){ 

    }

    Var operator+(Var& other) {
        Var out(data + other.data);
        
        out.AddChild(*this);
        out.AddChild(other);

        out._backward = [this, &out, &other](){
            std::printf("HOLA ORIGINAL BACKWARD - %f\n", out.grad);
            this->grad += 1.0 * out.grad;
            other.grad += 1.0 * out.grad;
        };

        return out;
    }

    void AddChild(Var& c){
        children.push_back(&c);
    }

    bool operator<(const Var& other) const{
        return this->data < other.data;
    }

    void Backward(){
        this->_backward();
  
        std::set<Var*> visited;
        std::stack<Var*> stack;
        std::vector<Var*> topo;

        stack.push(this);

        while(!stack.empty()){
            Var* current = stack.top();
            stack.pop();

            if(visited.find(current) == visited.end()){
                visited.insert(current);
                for (Var* child : current->children) {
                    stack.push(child);
                }
            }else {
                topo.push_back(current);
            }
        }

        this->grad = 1.0;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            Var* node = *it;
            if(node->_backward){
                node->_backward();
            }
        }
    }

    char* ToString() const {
        static char buffer[7]; // only 7 digits
        std::sprintf(buffer, "%.2f", grad);
        return buffer;
    }

    void Print() const {
        std::printf("%s\n", ToString());
    }
};


int main(int argc, char** argv) {
    Var x(2.0);
    Var z = x + x;

    std::printf("AQUI ESTA DEFINIDA?\n");

    // z.grad = 1.0f;
    // z._backward();
    //x._backward();
    z.Backward();

    x.Print();

    return 0;
}
