#include <stdio.h>
#include "pico/stdlib.h"

// Wrapper do TensorFlow Lite Micro (certifique-se que você tem esses arquivos no projeto)
#include "tflm_wrapper.h"

// Arquivo gerado pelo script Python (contém test_data_norm e test_data_norm_labels)
#include "dados_teste_norm.h" 

#define NUM_CLASSES 2             // Breast Cancer: 0 (Maligno) e 1 (Benigno)
#define NUM_FEATURES 30           // O modelo espera 30 entradas

// Matriz de confusão 2x2: Linha=Real, Coluna=Predito
static int conf_matrix[NUM_CLASSES][NUM_CLASSES];

/*
 * Função argmax
 * Retorna o índice da maior probabilidade.
 * Adaptada para 2 classes.
 */
int argmax(const float v[NUM_CLASSES]) {
    if (v[0] > v[1]) return 0;
    return 1;
}

// Função principal
int main() {

    // Inicializa printf via USB
    stdio_init_all();
    sleep_ms(2000); // Aguarda conexão serial

    printf("\n=== TinyML Breast Cancer - Validacao ===\n");

    // Inicializa o modelo (carrega model_data.h internamente no wrapper)
    if (tflm_init_model() != 0) {
        printf("ERRO: Falha ao inicializar modelo TFLite.\n");
        return 1;
    }

    printf("Modelo inicializado com sucesso!\n");
    // test_data_norm_count vem definido dentro de dados_teste_norm.h
    printf("Iniciando inferencia em %d amostras de teste...\n", test_data_norm_count);

    // Zera matriz de confusão
    for (int i = 0; i < NUM_CLASSES; i++)
        for (int j = 0; j < NUM_CLASSES; j++)
            conf_matrix[i][j] = 0;

    int correct = 0;

    /*
     * Loop de inferência
     */
    for (int i = 0; i < test_data_norm_count; i++) {
        
        float scores[NUM_CLASSES];  // Array para receber as probabilidades [0, 1]
        
        // --- Passo 1: Obter Entrada ---
        // A entrada já vem normalizada do Python.
        // Acessamos a linha 'i' da matriz gerada no .h
        // Nota: O wrapper geralmente aceita o array diretamente.
        // Se o seu tflm_wrapper exigir float* não-const, pode ser necessário copiar,
        // mas aqui passamos o endereço da linha atual.
        const float *input_sample = test_data_norm[i];

        // --- Passo 2: Inferência ---
        // Envia as 30 features para o modelo
        tflm_infer(input_sample, scores);

        // --- Passo 3: Pós-processamento ---
        int pred = argmax(scores);          // Classe prevista
        int real = test_data_norm_labels[i]; // Classe real (do arquivo .h)

        // Contabiliza acertos
        if (pred == real) correct++;

        // Atualiza matriz de confusão
        conf_matrix[real][pred]++;

        // Imprime detalhes das primeiras 10 amostras para debug
        if (i < 10) {
            printf("Amostra %02d | Real: %d | Pred: %d | Prob(0-Mal): %.3f  Prob(1-Ben): %.3f\n",
                i, real, pred, scores[0], scores[1]);
        }
    }

    // --- Exibe Resultados Finais ---
    printf("\n=== Matriz de Confusao ===\n");
    printf("Legenda: 0 = Maligno, 1 = Benigno\n\n");
    printf("         Pred 0   Pred 1\n");
    
    // Linha 0 (Real Maligno)
    printf("Real 0   %6d   %6d\n", conf_matrix[0][0], conf_matrix[0][1]);
    
    // Linha 1 (Real Benigno)
    printf("Real 1   %6d   %6d\n", conf_matrix[1][0], conf_matrix[1][1]);

    // Cálculo da Acurácia
    float accuracy = (float)correct / test_data_norm_count;
    printf("\nAcuracia final: %.2f%% (%d / %d)\n", accuracy * 100.0f, correct, test_data_norm_count);

    printf("\nTeste concluido.\n");

    while(1) {
        tight_loop_contents(); // Mantém o processador em espera eficiente
    }
}