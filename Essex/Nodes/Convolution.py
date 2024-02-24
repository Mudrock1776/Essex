import numpy as np, numpy

class Convolution:
    def __init__(self, filterSize:str = "2x2", learningRate:float = 0.1, filter:list = None) -> None:
        if filter is not None:
            self.filter = np.array(filter)
        else:
            x,y = filterSize.split("x")
            x = int(x)
            y = int(y)
            self.filter = np.random.randn(y,x)
        
        self.learningRate = learningRate
    
    def predict(self, inputs):
        outputArray = []
        for i in range(len(inputs) - len(self.filter)+1):
            outputRow = []
            for n in range(len(inputs[i]) - len(self.filter[0])+1):
                output = 0
                for z in range(len(self.filter)):
                    for c in range(len(self.filter[z])):
                        output += self.filter[z][c] * inputs[i+z][n+c]
                outputRow.append(output)
            outputArray.append(np.array(outputRow))
        return np.array(outputArray)

    def getModel(self):
        return self.filter

if __name__ == "__main__":
    Conv = Convolution()
    inputs = np.random.randn(4,4)
    output = Conv.predict(inputs)
    print(inputs)
    print(output)