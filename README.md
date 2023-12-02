# neural_net_fromScratch
Implementation of a single perceptron NN to classify Academic IDs. 

Disciplina de Inteligência Artificial Aplicada do Curso de Ciência de Dados da Uninter. 

### Modelagem
A estrutura do RU é dada pela formulação da palavra a seguir: 
$$RU = a_1 a_2 a_3 a_4 a_5 a_6 a_7$$
sobre a qual $a_i \in \mathbb{N}, a_i = 0, 1, 2, \ldots , 9$

Seja X um vetor representando um RU. O classificador $f$ opera sobre $X$ pela arquitetura de um Perceptron, modelado pela equação a seguir: 

$$f(X) = \psi(X \cdot W + b)$$

onde a função de ativação $\psi$ é dada por:  

$$\psi(z) = \begin{cases}
		1  & z \geq 0 \\
		-1 & z < 0    \\
	\end{cases}$$

\section{Treinamento da Rede}
Treinar uma rede neural envolve duas etapas: \textit{foward propagation} e \textit{back propagation}. Na primeira, é realizado a predição de um dado a partir de um vetor $X_i$. A etapa seguinte envolve a validação do dado predito, comparando-o com um rótulo de referência e ajustando os pesos $W$ da rede. 

A partir de uma predição $\hat{y}$, caso $\hat{y} \neq y$, devemos atualizar os pesos W como segue: 

$$\mathcal{E} = y - \hat{y}$$

Os pesos do vetor $W$ na iteração seguinte serão dados pela formula a seguir, onde $\eta$ é o \textit{learning rate}, tipicamente entre 0.01 e 0.2: 
$$W_j^{(n+1)} = W_j^{(n)} + \eta\mathcal{E}_jX_j$$


### Resultados
#### Treinamento
![Processo de treinamento](/resultados/treinamento.png)

### Acurácia 
Dividindo-se o total de predições corrretas pelo total geral de predições no dataset de treino, temos:

![Acurácia](/resultados/resultado.png)


