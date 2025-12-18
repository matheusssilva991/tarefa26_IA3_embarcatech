#include <cstdio>
#include "pico/stdlib.h"

// -------------------------------------------------------------------
// TensorFlow Lite Micro (via pico-tflmicro)
// -------------------------------------------------------------------
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// HEADER DO NOVO MODELO (Gerado via xxd)
#include "model_data.h" 

// API em C que será chamada pelo main.c
#include "tflm_wrapper.h"

// -------------------------------------------------------------------
// Objetos estáticos do TFLM
// -------------------------------------------------------------------
namespace {

// Tamanho da arena de tensores.
// 8KB (8 * 1024) geralmente sobra para MLPs simples float32.
constexpr int kTensorArenaSize = 8 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// Logger de erros (ponteiro é depreciado em versões novas, mas mantido para compatibilidade)
static tflite::MicroErrorReporter micro_error_reporter;
static tflite::ErrorReporter* error_reporter = &micro_error_reporter;

// Modelo
static const tflite::Model* model = nullptr;

// Registrador de operações.
// Breast Cancer MLP usa: FullyConnected, Relu (ou Tanh), Softmax e Reshape.
// Aumentei para <6> para garantir espaço se precisar adicionar algo novo.
static tflite::MicroMutableOpResolver<6> resolver;

// Intérprete e tensores
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;

}  // namespace

// -------------------------------------------------------------------
// Inicializa o modelo TFLM
// -------------------------------------------------------------------
int tflm_init_model(void) {
    // Carrega o modelo do array hexadecimal.
    // O nome 'model_tflite' vem do arquivo gerado pelo 'xxd -i model.tflite'
    model = tflite::GetModel(model_tflite);
    
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Erro: Versao do Schema do modelo incompativel.\n");
        return -1;
    }

    // Registra as operações necessárias para o modelo rodar.
    // Se o seu modelo usar algo diferente (ex: Logistic/Sigmoid), adicione aqui.
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddSoftmax();
    resolver.AddReshape();
    // resolver.AddLogistic(); // Descomente se usou Sigmoid na saída em vez de Softmax

    // Cria o intérprete estático
    static tflite::MicroInterpreter static_interpreter(
        model,
        resolver,
        tensor_arena,
        kTensorArenaSize,
        nullptr,  
        nullptr,  
        false     
    );

    interpreter = &static_interpreter;

    // Aloca memória para os tensores na Arena
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("Erro: AllocateTensors falhou. Memoria insuficiente?\n");
        return -2;
    }

    // Obtém ponteiros
    input_tensor  = interpreter->input(0);
    output_tensor = interpreter->output(0);

    if (!input_tensor || !output_tensor) {
        printf("Erro ao obter tensores de entrada/saida.\n");
        return -3;
    }

    // Validação de dimensões para Debug
    // O input do Breast Cancer deve ser (1, 30)
    if (input_tensor->dims->data[1] != 30) {
        printf("ALERTA: O modelo espera %d features, mas o codigo C esta enviando 30.\n", 
               input_tensor->dims->data[1]);
    }

    return 0;
}

// -------------------------------------------------------------------
// Executa inferência (Adaptado para 30 inputs -> 2 outputs)
// -------------------------------------------------------------------
int tflm_infer(const float in_features[30], float out_scores[2]) {
    
    if (!interpreter || !input_tensor || !output_tensor) {
        printf("Erro: Interpretador nao inicializado.\n");
        return -1;
    }

    // Copia 30 features para o tensor de entrada
    for (int i = 0; i < 30; i++) {
        input_tensor->data.f[i] = in_features[i];
    }

    // Executa a inferência
    if (interpreter->Invoke() != kTfLiteOk) {
        printf("Erro: Invoke falhou.\n");
        return -2;
    }

    // Copia as saídas (2 classes: Maligno/Benigno)
    // O tamanho deve bater com NUM_CLASSES no main.c
    int num_classes = output_tensor->dims->data[1]; 
    for (int i = 0; i < num_classes; i++) {
        out_scores[i] = output_tensor->data.f[i];
    }

    return 0;
}
