// If your target is limited in memory remove this macro to save 10K RAM

#define ONCOMPUTER 0
#include "nn.h"

#define T_BEAM_V10 // ttgo-t-beam
// #define T_BEAM_LORA_32 // ttgo-lora32-v1

#if defined(T_BEAM_LORA_32)
#define LED LED_BUILTIN
#define LED_ON      HIGH
#define LED_OFF     LOW
#elif defined(T_BEAM_V10)
#define LED 4
#define LED_ON      LOW
#define LED_OFF     HIGH
#endif

static u32 memoryUsed = 0;
static f32 lr = 0.001f;

struct train_data{
    M input;
    M target;
};

struct train_data_cnn{
    M3 input;
    M target;
};

#define OUTPUT_SIZE 25

// CNN DEFINES
#define SIZE 28
#define CHANNELS 1
#define KERNEL_SIZE 5
#define OUT_CHANNELS 8

static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal


// /** Audio buffers, pointers and selectors */
// typedef struct {
//     int16_t buffer[EI_CLASSIFIER_RAW_SAMPLE_COUNT];
//     uint8_t buf_ready;
//     uint32_t buf_count;
//     uint32_t n_samples;
// } inference_t;

// static inference_t inference;

// typedef uint8_t scaledType;

// static scaledType microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
//     numpy::int16_to_float(&inference.buffer[offset], out_ptr, length);
//     return 0;
// }

static Layer *l1;
static Layer *l2;
//static Layer* l1;

// FF added fct declarations for platformio
void initNetworkModel();
void sendFloat (float arg);
void sendInt (int arg);
//void read_bias(Conv2D* l);
//void read_weights(Layer* l);




void read_weights(Layer* l) {
    //Serial.println("Receiving weights..");
    float* weights = l->w.data;

    for (uint16_t i = 0; i < l->w.rows * l->w.cols; ++i) {
        //Serial.write('n');
        while (Serial.available() < 4);

        char bytes[4];
        Serial.readBytes(bytes, 4);
        weights[i] = *reinterpret_cast<float*>(bytes);
    }

    // for (uint16_t i = 0; i < l->w.rows * l->w.cols; ++i) {
    //     Serial.print(l->w.data[i]);
    //     Serial.print(" ");
    // }
    // Serial.print(l->w.std());
}


void send_weights(MaxPooling*l){
    float* weights = l->grad_input.data;

    for (u32 i = 0; i < l->grad_input.d1 * l->grad_input.d2 * l->grad_input.d3; ++i) {
        Serial.write('n');
        sendFloat(weights[i]);
    }
}

void send_weights(Conv2D*l){
    float* weights = l->kernels.data;

    for (u32 i = 0; i < l->kernels.d1 * l->kernels.d2 * l->kernels.d3 * l->kernels.d4; ++i) {
        Serial.write('n');
        sendFloat(weights[i]);
    }
}

void send_bias(Conv2D*l){
    float* weights = l->bias.data;

    for (u16 i = 0; i < l->bias.rows * l->bias.cols; ++i) {
        Serial.write('n');
        sendFloat(weights[i]);
    }
}

void send_weights(Layer *l){
    float* weights = l->w.data;

    for (u16 i = 0; i < l->w.rows * l->w.cols; ++i) {
        Serial.write('n');
        sendFloat(weights[i]);
    }
}

void send_bias(Layer *l){
    float* weights = l->b.data;

    for (uint16_t i = 0; i < l->b.rows * l->b.cols; ++i) {
        Serial.write('n');
        sendFloat(weights[i]);
    }
}


void read_bias(Layer* l) {
    //Serial.println("Receiving bias..");
    float* bias = l->b.data;
    for (uint16_t i = 0; i < l->b.rows * l->b.cols; ++i) {
        //Serial.write('n');
        while (Serial.available() < 4);

        char bytes[4];
        Serial.readBytes(bytes, 4);
        bias[i] = *reinterpret_cast<float*>(bytes);
    }

    // for (uint16_t i = 0; i < l->b.rows * l->b.cols; ++i) {
    //     Serial.print(l->b.data[i]);
    //     Serial.print(" ");
    // }
    // Serial.print(l->b.std());
}

void read_weights(Conv2D* l) {
    //Serial.println("Receiving weights..");
    float* weights = l->kernels.data;
    for (uint16_t i = 0; i < l->kernels.d1 * l->kernels.d2 * l->kernels.d3 * l->kernels.d4; ++i) {
        //Serial.write('n');
        while (Serial.available() < 4);

        char bytes[4];
        Serial.readBytes(bytes, 4);
        weights[i] = *reinterpret_cast<float*>(bytes);
    }

    // for (uint16_t i = 0; i < l->w.rows * l->w.cols; ++i) {
    //     Serial.print(l->w.data[i]);
    //     Serial.print(" ");
    // }
    // Serial.print(l->w.std());
}

void read_weights(MaxPooling* l) {
    //Serial.println("Receiving weights..");
    // float* weights = l->grad_input.data;
    // for (uint16_t i = 0; i < l->grad_input.d1 * l->grad_input.d2 * l->grad_input.d3; ++i) {
    //     //Serial.write('n');
    //     while (Serial.available() < 4);

    //     char bytes[4];
    //     Serial.readBytes(bytes, 4);
    //     //weights[i] = *reinterpret_cast<float*>(bytes);
    // }

    // for (uint16_t i = 0; i < l->w.rows * l->w.cols; ++i) {
    //     Serial.print(l->w.data[i]);
    //     Serial.print(" ");
    // }
    // Serial.print(l->w.std());
}

void read_bias(Conv2D* l) {
    //Serial.println("Receiving bias..");
    float* bias = l->bias.data;
    for (uint16_t i = 0; i < l->bias.rows * l->bias.cols; ++i) {
        //Serial.write('n');
        while (Serial.available() < 4);

        char bytes[4];
        Serial.readBytes(bytes, 4);
        bias[i] = *reinterpret_cast<float*>(bytes);
    }

    // for (uint16_t i = 0; i < l->b.rows * l->b.cols; ++i) {
    //     Serial.print(l->b.data[i]);
    //     Serial.print(" ");
    // }
    // Serial.print(l->b.std());
}

void read_bias(MaxPooling* l) {
}


void read_layer_weights(Layer* l){
    read_bias(l);
    read_weights(l);
}

void read_layer_weights(Conv2D* l){
    read_bias(l);
    read_weights(l);
}

void read_layer_weights(MaxPooling* l){
    read_bias(l);
    read_weights(l);
}

void send_layer_weights(Layer* l){
    send_weights(l);
    send_bias(l);
}

void send_layer_weights(MaxPooling* l){
    send_weights(l);
    //send_bias(l);
}

void sendLayerMetaData(Layer* l){
    sendInt(-1); // Dense layer
    sendInt(l->w.rows);
    sendInt(l->w.cols);
}

void sendLayerMetaData(MaxPooling* l){
    sendInt(-2); // Max pool layer
    sendInt(l->grad_input.d1);
    sendInt(l->grad_input.d2);
    sendInt(l->grad_input.d3);
}


void sendLayerMetaData(Conv2D* l){
    sendInt(-3); // CNN
    sendInt(l->kernels.d1);
    sendInt(l->kernels.d2);
    sendInt(l->kernels.d3);
    sendInt(l->kernels.d4);
}

// SEND GRADIENTS
void send_gradients(Conv2D*l){
    float* weights = l->dkernels.data;

    for (u32 i = 0; i < l->dkernels.d1 * l->dkernels.d2 * l->dkernels.d3 * l->dkernels.d4; ++i) {
        Serial.write('n');
        sendFloat(weights[i]);
    }
}

void send_gradient_bias(Conv2D*l){
    float* weights = l->db.data;

    for (u16 i = 0; i < l->db.rows * l->db.cols; ++i) {
        Serial.write('n');
        sendFloat(weights[i]);
    }
}

void send_gradient_bias(Layer*l){
    float* weights = l->db.data;

    for (u16 i = 0; i < l->db.rows * l->db.cols; ++i) {
        Serial.write('n');
        sendFloat(weights[i]);
    }
}

void send_gradients(Layer *l){
    float* weights = l->dw.data;

    for (u16 i = 0; i < l->dw.rows * l->dw.cols; ++i) {
        Serial.write('n');
        sendFloat(weights[i]);
        // sendFloat(weights[i]);
    }
}

void send_gradients(MaxPooling *l){
    float* weights = l->grad_input.data;

    for (u32 i = 0; i < l->grad_input.d1 * l->grad_input.d2 * l->grad_input.d3; ++i) {
        Serial.write('n');
        sendFloat(weights[i]);
    }
}


void send_layer_gradients(Layer* l){
    send_gradients(l);
    send_gradient_bias(l);
}

void send_layer_gradients(MaxPooling* l){
    send_gradients(l);
    //send_gradient_bias(l);
}

void send_layer_weights(Conv2D* l){
    send_weights(l);
    send_bias(l);
}

void send_layer_gradients(Conv2D* l){
    send_gradients(l);
    send_gradient_bias(l);
}

void initNetworkModel(){
    Serial.println("Start receiving model");
    char signal;
    do {
        signal = Serial.read();
        Serial.println(memoryUsed);
    } while(signal != 's'); // s -> START

    Serial.println("start");
    Serial.write("i");

    // READ LEARNING RATE
    while(Serial.available()<4);
    char bytes[4];
    Serial.readBytes(bytes, 4);
    memcpy(&lr, bytes, sizeof(f32));


    // How many layers?
    sendInt(2);
    sendLayerMetaData(l1);
    sendLayerMetaData(l2);
    //sendLayerMetaData(l1);


    // TODO : Change to read_weights(l1): this will read bias and weights within the same function.
    read_layer_weights(l1);
    //read_layer_weights(pl1);
    read_layer_weights(l2);
    //read_layer_weights(l1);
    // read_bias(l1);
    // read_weights(l1);

    // read_bias(l2);
    // read_weights(l2);
}

train_data receive_sample(u32 input_size){
    train_data data = {};
    data.input = M::zeros(1, input_size);
    data.target = M::zeros(1, OUTPUT_SIZE);

    f32* input = data.input.data;
    f32* target = data.target.data;

    for(u16 i=0;i<input_size;++i){
        // Serial.write('n');
        while (Serial.available() < 4);

        char bytes[4];
        Serial.readBytes(bytes, 4);
        input[i] = *reinterpret_cast<float*>(bytes);
        
    }
    
    for(u16 i=0;i<OUTPUT_SIZE;++i){
        //Serial.write('n');
        while (Serial.available() < 4);

        char bytes[4];
        Serial.readBytes(bytes, 4);
        target[i] = *reinterpret_cast<float*>(bytes);
    }

    return data;
}


train_data_cnn receive_sample_cnn(u32 height, u32 width, u32 channels, u32 output_size){
    train_data_cnn data = {};
    data.input = M3::zeros(height, width, channels);
    data.target = M::zeros(1, output_size);

    f32* input = data.input.data;
    f32* target = data.target.data;


    for(u32 i=0;i<height*width*channels;++i){
        // Serial.write('n');
        //sendInt(i);
        while (Serial.available() < 4);

        char bytes[4];
        Serial.readBytes(bytes, 4);
        input[i] = *reinterpret_cast<float*>(bytes);
    }
    
    for(u16 i=0;i<output_size;++i){
        while (Serial.available() < 4);

        char bytes[4];
        Serial.readBytes(bytes, 4);
        target[i] = *reinterpret_cast<float*>(bytes);
    }

    return data;
}

float readFloat() {
    byte res[4];
    while(Serial.available() < 4) {}
    for (int n = 0; n < 4; n++) {
        res[n] = Serial.read();
    }
    return *(float *)&res;
}

void sendM(M arg)
{
    for(u32 i=0;i<arg.cols * arg.rows;++i){
        Serial.write('n');
        sendFloat(arg.data[i]);
    }
}

void sendInferenceResult(M arg, f32 loss)
{
    for(u32 i=0;i<arg.cols * arg.rows;++i){
        Serial.write('n');
        sendFloat(arg.data[i]);
    }

    Serial.write('n');
    sendFloat(loss);
}

void sendFloat (float arg)
{
    // get access to the float as a byte-array:
    byte * data = (byte *) &arg; 

    // write the data to the serial
    Serial.write (data, sizeof (arg));
}

void sendInt (int arg)
{
    // get access to the float as a byte-array:
    byte * data = (byte *) &arg; 

    // write the data to the serial
    Serial.write (data, sizeof (arg));
}

M train(M input, M target, f32& loss){
    l1->resetGradients();
    l2->resetGradients();
    M a = Tanh(l1->forward(input));
    M b = Tanh(l2->forward(a));
    M _d4 = MsePrime(input, b);
    M _d1 = l2->backward(_d4) * TanhPrime(a);

    // accumulate gradients
    l1->dw += l1->getDelta(_d1, input);
    l1->db += _d1;

    l2->dw += l2->getDelta(_d4, a);
    l2->db += _d4;

    l1->UpdateWeights(lr);
    l2->UpdateWeights(lr);
    loss = Mse(input, b);
    return b;
}

M predict(M input, M target, f32& loss){
    M a = Tanh(l1->forward(input));
    M b = Tanh(l2->forward(a));
    
    // calculate error
    loss = Mse(input, b);
    return b;
}

void setup(){
    Serial.begin(9600);
    InitMemory(1024 * 430);

    l1 = Layer::create(25, 30);
    l2 = Layer::create(30, 25);

    memoryUsed = MemoryArena.Used;

    //pinMode(LEDR, OUTPUT);
    //pinMode(LEDG, OUTPUT);
    //pinMode(LEDB, OUTPUT);
    pinMode(LED, OUTPUT);

    //digitalWrite(LEDR, HIGH);
    //digitalWrite(LEDG, HIGH);
    //digitalWrite(LEDB, HIGH);
    digitalWrite(LED, LED_ON);
    digitalWrite(LED, LED_OFF);

    // put your setup code here, to run once:
    randomSeed(0);

    initNetworkModel();
    digitalWrite(LED_BUILTIN, LOW);    // OFF   
}

void loop(){
    if(Serial.available() > 0){
        char inByte = Serial.read();
        if(inByte == 't'){
            //train_data_cnn sample = receive_sample_cnn(SIZE,SIZE,CHANNELS, 10);
            train_data sample = receive_sample(25);
            f32 error = 0;
            M output = train(sample.input, sample.target, error);
            //sendFloat(error);
            sendInferenceResult(output, error);
        } else if(inByte == 'p'){
            //train_data_cnn sample = receive_sample_cnn(SIZE,SIZE,CHANNELS, 10);

            train_data sample = receive_sample(25);
            //M input = receive_sample_inference();
            float loss = 0.0;
            M output = predict(sample.input, sample.target, loss);
            sendInferenceResult(output, loss);
            MemoryArena.Used = memoryUsed;
        } else if(inByte == 'f'){
            // START FEDERATED LEARNING
        } else if(inByte == 'g'){
            send_layer_weights(l1);
            //send_layer_weights(pl1);
            send_layer_weights(l2);
            //send_layer_weights(l1);
            // send_weights(l1);
            // send_bias(l1);
            // send_weights(l2);
            // send_bias(l2);
        } else if(inByte == 'r'){
            // ALWAYS READ BIAS FIRST!!!
            read_layer_weights(l1);
            //read_layer_weights(pl1);
            read_layer_weights(l2);
            //read_layer_weights(l1);

            // read_bias(l1);
            // read_weights(l1);

            // read_bias(l2);
            // read_weights(l2);
        }
    }
    MemoryArena.Used = memoryUsed;
}