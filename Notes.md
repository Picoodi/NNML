# Those are my notes that i wrote in the process of building this Network

## Neuron
A Neuron works getting an input from all the neurons on the previous layer.     
It adds a specific weight to each specific input adds those togehter and in the end the bias.     
In the code we have a neuron with three previouse neurons.


```python

inputs = [1.2, 5.1, 2.1]
weights = [3.1,2.1,8.7]
bias = 3

output = inputs[0]*weights[0]+inputs[1]*weights[1]+inputs[2]*weights[2] + bias
print(output)

```

Next we can also do this with more neurons so we can bild a layer. Same math and just adding the neurons themselves also.

```python
inputs = [1, 2, 3,2.5]

weights = [0.2,0.8,-0.5,1]
weights2 = [0.5,-0.91,0.26,-0.5]
weights3 = [-0.26,-0.27,0.17,0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

output = [inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + inputs[3]*weights[3] + bias1,
          inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
          inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]
          
print(output)

```

If you didnt know you can tweak the weights and the bias too change your outputs so in the end you hopefully get the right result.  
And thats also why its called machine learning cause you need the machine to learn whats right so it works afterwards. But more on that later.  

At first we make the code more compact with 2for loops 
```python

inputs = [1, 2, 3,2.5]

weights = [[0.2,0.8,-0.5,1],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]


biases = [2,3,0.5]



layer_outputs = [] #output of the current layer
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0 #output of given neurons

    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    
    neuron_output += neuron_bias

    layer_outputs.append(neuron_output)


print(layer_outputs)


```


## Math and NumPy 
We need some more complex math and for that we can use Numpy a python library to make our lives a bit more easy.

### Shape
```python
l = [1,5,6,2]

lol = [
      [1,5,6,2],
      [3,2,1,3]
      ]

lolol = [
        [[1,5,6,2],[3,2,1,3]],
        [[5,2,2,2],[6,4,8,4]],
        [[2,8,5,3],[1,1,9,4]]
        ]
```
The shape of l would be (4,) and the type a 1D array and Vector     
The shape of lol would be (2, 4) cause the list lol itself has 2 lists itself and each of those has 4 data inside. 
Its also a 2D Array and Matrix.
The shape of lolol is (3, 2, 4) and is a 3d Array sometimes also called a Tensor.


To get the dot_product of vectors is when we multiply the elements of a vector/list element whise.  
We use numpy now and also get our 3 Outputs like before.    
We have to put the weights in before the inputs cause weights is more dimentional.

```python
import numpy as np
inputs = [1, 2, 3,2.5]

weights = [[0.2,0.8,-0.5,1],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]


biases = [2,3,0.5]



output = np.dot(weights, inputs) +biases
print(output)

```

## Batches
Batches are there to run stuff in parallel and just give multiple inputs.   
But we have to transpose the weights matrix because of matrix math cause    
otherwise we will get a shape error.
```python
import numpy as np
inputs = [[1, 2, 3,2.5],
          [2.0,5.0,-1.0,2.0],
          [-1.5,2.7,3.3,-0.8]]



weights = [[0.2,0.8,-0.5,1],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]
           ]


biases = [2,3,0.5]



output = np.dot(inputs, np.array(weights).T) +biases
print(output)
```

## Adding a Layer
Now we gonna add a layer by just simply copying the weigths and biases  
then doing the math on layer1 and taking that as input for layer 1.
```python
import numpy as np
inputs = [[1, 2, 3,2.5],
          [2.0,5.0,-1.0,2.0],
          [-1.5,2.7,3.3,-0.8]]



weights = [[0.2,0.8,-0.5,1],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]
           ]


biases = [2,3,0.5]


weights2 = [[0.1,-0.14,0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]
           ]


biases2 = [-1,2, -0.5]


layer_1_output = np.dot(inputs, np.array(weights).T) +biases
print(layer_1_output)
print("----------------------")
layer_2_output = np.dot(layer_1_output, np.array(weights2).T) +biases2
print(layer_2_output)
```

## Classing things
Now we gonna go and build classes of the different stuff we need.   
Als we create a NumPy seed of 0 so no biases can be 3 and we can prevent a dead network.
### Classing a Layer
```python
import numpy as np

np.random.seed(0)


X = [[1, 2, 3,2.5],
    [2.0,5.0,-1.0,2.0],
    [-1.5,2.7,3.3,-0.8]]


class Layer_Dense:
    def __init__(self,n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons )
        self.biases =  0.10 * np.zeros((1, n_neurons ))



    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases




#first number needs to be the output of the layer before
layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
layer2.forward(layer1.output)
print(layer2.output)

```

