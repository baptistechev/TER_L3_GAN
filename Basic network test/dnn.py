import numpy as np 

x_entrer = np.array(([3, 1.5], [2, 1], [4, 1.5], [3, 1], [3.5,0.5], [2,0.5], [5.5,1], [1,1], [4,1.5]), dtype=float) # données d'entrer
y = np.array(([1], [0], [1],[0],[1],[0],[1],[0]), dtype=float) # données de sortie /  1 = rouge /  0 = bleu

# Changement de l'échelle de nos valeurs pour être entre 0 et 1
x_entrer = x_entrer/np.amax(x_entrer, axis=0) # On divise chaque entré par la valeur max des entrées

# On récupère ce qu'il nous intéresse
X = np.split(x_entrer, [8])[0] # Données sur lesquelles on va s'entrainer, les 8 premières de notre matrice
xPrediction = np.split(x_entrer, [8])[1] # Valeur que l'on veut trouver

class DeepNeuralNetwork(object):

    def __init__(self):

        self.input = 2
        self.hidden = 3
        self.depth = 2
        self.output = 1

        self.eta = 0.01

        self.params = {}

        self.layers_dim = [self.input]
        for i in range(self.depth-1):
            self.layers_dim.append(self.hidden)
        self.layers_dim.append(self.output)

        for i in range(1, self.depth+1):
            self.params['W' + str(i)] = np.random.randn(self.layers_dim[i], self.layers_dim[i-1])*0.01
            self.params['b' + str(i)] = np.zeros((self.layers_dim[i], 1))

        # print(self.layers_dim)
        # for i in range(self.depth):
        #     print('W'+ str(i+1) + " :")
        #     print(self.params['W'+ str(i+1)])
        #     print('b'+ str(i+1) + " :")
        #     print(self.params['b'+ str(i+1)])


    def forward(self, X):
        A = X
        cache = {}
        for i in range(1, self.depth):
            W, b = self.params['W'+str(i)], self.params['b'+str(i)]
            Z = W.dot(A) + b
            A = self.relu(Z)
            cache['Z'+str(i)] = Z
            cache['A'+str(i)] = A

        #last layer
        W, b = self.params['W'+str(i+1)], self.params['b'+str(i+1)]
        Z = W.dot(A) + b
        A = self.sigmoid(Z)
        cache['Z'+str(i+1)] = Z
        cache['A'+str(i+1)] = A

        return cache, A

    def sigmoid(self, s):
        return 1 / (1+np.exp(-s))

    def relu(self, s): 
        return np.maximum(s, 0)
    
    def sigmoid_grad(self, A, Z):
        grad = np.multiply(A, 1-A)
        return grad


    def relu_grad(self, A, Z):
        grad = np.zeros(Z.shape)
        grad[Z>0] = 1
        return grad

    def compute_cost(self, A, Y):
        """
        For binary classification, both A and Y would have shape (1, m), where m is the batch size
        """
        assert A.shape == Y.shape
        m = A.shape[1]
        s = np.dot(Y, np.log(A.T)) + np.dot(1-Y, np.log((1 - A).T))
        loss = -s/m
        return np.squeeze(loss)

    def backward(self, X, y, cache):
        """
        params: weight [W, b]
        cache: result [A, Z]
        Y: shape (1, m)
        """

        grad = {}
        m = y.shape[1]
        cache['A0'] = X

        for l in range(self.depth, 0, -1):
            A, A_prev, Z = cache['A' + str(l)], cache['A' + str(l-1)], cache['Z' + str(l)]
            W = self.params['W'+str(l)]
            if l == self.depth:
                dA = -np.divide(y, A) + np.divide(1 - y, 1 - A)
                dZ = dA * self.sigmoid_grad(A, Z)
            else:
                dZ = dA * self.relu_grad(A,Z)
            dW = dZ.dot(A_prev.T)/m
            db = np.sum(dZ, axis=1, keepdims=True)/m
            dA = W.T.dot(dZ)

            grad['dW'+str(l)] = dW
            grad['db'+str(l)] = db

        for i in range(1, self.depth+1):
            dW, db = grad['dW'+str(i)], grad['db'+str(i)]
            self.params['W'+str(i)] -= self.eta*dW
            self.params['b'+str(i)] -= self.eta*db

    def train(self, X, y):
        cache, A = self.forward(X)
        print(self.compute_cost(A, y))
        self.backward(X,y,cache)

    def predict(self):
        
        print(self.forward(X)[1])
        print("Donnée prédite apres entrainement: ")
        print("Entrée : \n" + str(xPrediction))
        print("Sortie : \n" + str(self.forward(xPrediction)[1]))

        if(self.forward(xPrediction)[1] < 0.5):
            print("La fleur est BLEU ! \n")
        else:
            print("La fleur est ROUGE ! \n")

X = X.T
y = y.T
xPrediction = xPrediction.T
NN = DeepNeuralNetwork()
for i in range(10000):
    NN.train(X,y)

NN.predict()