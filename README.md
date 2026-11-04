# Neurônio do zero: classificação de câncer de mama

Um único neurônio artificial (regressão logística com sigmoid), implementado do zero em Python com NumPy, que classifica tumores de mama como **malignos** ou **benignos**.

## Objetivo

Este é um projeto de aprendizado. O objetivo foi entender o que acontece por trás de um modelo de machine learning, sem usar funções prontas como `model.fit()`. O scikit-learn é usado **apenas para carregar os dados**. Todo o resto foi escrito à mão:

- divisão entre treino e teste
- normalização dos dados
- função sigmoid
- treino com gradiente descendente
- cálculo da acurácia

## Dataset

[Breast Cancer Wisconsin](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html), disponível no scikit-learn:

- 569 pacientes
- 30 características numéricas de cada tumor (raio, textura, perímetro, área etc.)
- Classes: `0` = maligno, `1` = benigno

## Como funciona

1. **Divisão treino/teste (80/20):** os índices são embaralhados com semente fixa e depois divididos. O teste nunca é usado durante o treino.
2. **Normalização:** cada coluna é transformada com `(valor - média) / desvio padrão`. Média e desvio são calculados **só com o treino** e reaproveitados no teste, para evitar vazamento de dados.
3. **Inicialização:** um peso para cada característica (30) e um bias, todos começando em zero.
4. **Soma ponderada:** `z = X · pesos + bias`.
5. **Sigmoid:** `1 / (1 + e^(-z))` transforma `z` em uma probabilidade entre 0 e 1.
6. **Erro e gradiente:** o erro é `p - y`, e o gradiente indica quanto cada peso contribuiu para ele.
7. **Atualização:** os pesos andam no sentido oposto ao gradiente (gradiente descendente), multiplicados pela taxa de aprendizado.
8. **Previsão:** probabilidade maior ou igual a 0,5 vira classe 1 (benigno), caso contrário classe 0 (maligno).
9. **Avaliação:** a acurácia é a fração de acertos no conjunto de teste.

## Como executar

```bash
pip install -r requirements.txt
python neuronio.py
```

## Resultados

Acurácia no conjunto de teste (114 pacientes): **96,49%**

Hiperparâmetros usados: taxa de aprendizado `0.01` e `1000` épocas.

O programa também imprime, para os 10 primeiros pacientes do teste, o que o modelo previu e o valor real.

## O que aprendi

- **Por que normalizar:** as colunas têm escalas muito diferentes, e sem normalização o treino fica lento e instável.
- **Por que separar o teste:** só assim dá para saber se o modelo aprendeu o padrão ou apenas decorou os dados de treino.
- **Vazamento de dados:** calcular média e desvio com o teste infla o resultado.
- **Depuração:** um bug clássico que tive foi usar `peso` e `pesos` como variáveis diferentes. O modelo treinava, mas os pesos nunca eram atualizados.
- **Gradiente e backpropagation:** com um único neurônio, o gradiente sai de uma fórmula direta. Em redes com várias camadas, o backpropagation faz esse cálculo camada por camada.

## Próximos passos

- Acompanhar a perda durante o treino e plotar a curva
- Comparar a acurácia no treino e no teste
- Implementar a matriz de confusão
- Testar diferentes taxas de aprendizado e números de épocas
- Evoluir para uma rede com várias camadas
