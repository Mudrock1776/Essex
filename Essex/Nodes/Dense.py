import numpy as np, numpy

class Dense:
    def __init__(self, inputs:int = None, learningRate:float = 0.1, weights:list = None, bias:float = None, activationFunction = None, dActivationFunction = None) -> None:
        if weights is not None:
            self.weights = np.array(weights)
        else:
            self.weights = np.random.rand(inputs)
        
        if bias is not None:
            self.bias = bias
        else:
            self.bias = np.random.random()

        self.learningRate = learningRate
        
        self.activationFunction = activationFunction
        self.dActivationFunction = dActivationFunction

        if self.activationFunction == "sig":
            self.activationFunction = self.sigmoid
            self.dActivationFunction = self.dSigmoid
        elif self.activationFunction == "tanh":
            self.activationFunction = np.tanh
            self.dActivationFunction = self.dTanh

    def sigmoid(self, value):
        value = np.exp(value * -1)
        value += 1
        value = 1/value
        return value
    
    def dSigmoid(self, value):
        value = self.sigmoid(value)
        value = value * (1-value)
        return value
    
    def dTanh(self,value):
        value = np.tanh(value)
        value = value **2
        value = 1-value
        return value
    
    def predict(self,inputs):
        value = np.dot(inputs,self.weights) + self.bias
        if self.activationFunction is not None:
            value = self.activationFunction(value)
        return value
    
    def train(self,dError,inputs):
        weightChange = np.zeros(len(self.weights))
        biasChange = 0
        dErrordInput = []

        for i in range(len(inputs)):
            if self.activationFunction is not None:
                layer1 = np.dot(inputs[i],self.weights) + self.bias
                trainingCalculation = self.dActivationFunction(layer1) * dError[i]
            else:
                trainingCalculation = dError[i]
            
            weightChange += trainingCalculation * inputs[i]
            biasChange += trainingCalculation
            dErrordInput.append(trainingCalculation * self.weights)
        
        self.weights -= weightChange * self.learningRate
        self.bias -= biasChange * self.learningRate

        return dErrordInput

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    Sig = Dense(4)
    cumError = []
    inputs = [[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],[1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1]]
    for i in range(len(inputs)):
        inputs[i] = np.array(inputs[i])
    desiredOutcome = [0,0,.5,.5,.5,.5,1,1,0,0,.5,.5,.5,.5,1,1]
    for z in range(100):
        Err = 0
        for i in range(len(inputs)):
            prediction = Sig.predict(inputs[i])
            dError = [2*(prediction - desiredOutcome[i])]
            Sig.train(dError,[inputs[i]])
        
        for i in range(len(inputs)):
            prediction = Sig.predict(inputs[i])
            Err += (prediction - desiredOutcome[i]) **2
        cumError.append(Err)
    
    plt.plot(cumError)
    plt.title("Cumulative Error Sigmoid")
    plt.savefig("sigGraph")