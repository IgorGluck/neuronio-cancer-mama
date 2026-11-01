import numpy as np
from sklearn.datasets import load_breast_cancer

#carregar os dados do conjunto de dados de câncer de mama
X, y = load_breast_cancer(return_X_y=True)


#verifica o tamnho dos dados:
#print("Shape of X:", X.shape);
#print("Shape of y:", y.shape);  

# separar treino e teste




np.random.seed(42)  # para reprodutibilidade
embaralhamento = np.random.permutation(len(X))  
corte = int(0.8 * len(X))                       # ponto de corte: 455

indices_treino = embaralhamento[:corte]
indices_teste = embaralhamento[corte:]

X_treino = X[indices_treino]
y_treino = y[indices_treino]
X_teste = X[indices_teste]
y_teste = y[indices_teste]

#normalizar os dados

media =np.mean(X_treino, axis=0)  # calcular a média de cada coluna
desvio_padrao = np.std(X_treino, axis=0)  # calcular o desvio padrão de cada coluna  


X_treino = (X_treino - media) / desvio_padrao   
X_teste = (X_teste - media) / desvio_padrao

#Criar os pesos,bias,taxa de aprendizado e epocas iniciais da rede neural

peso = np.zeros(X_treino.shape[1])  # inicializar os pesos com zeros    

bias = 0.0  # inicializar o bias com zero       

rate_aprendizado = 0.01  # taxa de aprendizado

epocas = 1000  # número de épocas para treinamento

#funcao sigmoid

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#loop de treinamento 

for epoca in range(epocas):

    z = np.dot(X_treino, peso) + bias  # calcular a combinação linear
    p = sigmoid(z)  # aplicar a função sigmoid
    erro = p - y_treino  
    #gradiente dos pesos e bias 

    gradiente_peso = np.dot(X_treino.T, erro) / len(X_treino)   
    peso = peso - rate_aprendizado * gradiente_peso  # atuliza os pesos para o contrario do gradiente
    bias = bias - rate_aprendizado * np.mean(erro)  # atualizar o bias

#previsão do modelo nos dados de teste  

z_teste = np.dot(X_teste, peso) + bias
p_teste = sigmoid(z_teste)  # aplicar a função sigmoid nos dados de teste   
previsto = (p_teste >= 0.5).astype(int)  # probabilidades em 0 ou 1


#ver a resposta do modelo para os dados de teste e os valores reais  

for i in range(10):

    if previsto[i] == 1:
        resposta_modelo = "benigno"
    else:
        resposta_modelo = "maligno"
    if y_teste[i] == 1:
        resposta_real = "benigno"
    else:
        resposta_real = "maligno"
    print(f"Modelo: {resposta_modelo}, Real: {resposta_real}")
       






 
   






#acuracia do modelo
acuracia = np.mean(previsto == y_teste)*100
print(f"Acurácia do modelo: {acuracia:.2f}%"    )