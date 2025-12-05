from NN import Variable
from NN import MLP
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch
import torch.optim as optim
from NN_numpy import *

def test1():        # object creation
    print("\n test 1 ")
    x = Variable(10)
    print(x)
    print(x.__repr__())

    y = Variable(9.22)
    print(y)
    print(y.__repr__())

def test2():    # addition
    print("\n test 2 ")
    x = Variable(2)
    y = Variable(3)

    print(x+y)
    print(y+x)
    print(x + 3)
    print(2 + y)
    print(x + 3.0)
    print(y + 2.0)

def test3():    # neg
    print("\n test 3")
    x = Variable(5)
    print(-x)
    y = Variable(-5)
    print(-y)

def test4():    # mul
    print("\n test 4")
    x = Variable(2)
    y = Variable(3)

    print(x*y)
    print(y*x)
    print(x * 3)
    print(2 * y)
    print(x * 3.0)
    print(y * 2.0)

def test5():    # pow
    print("\n test 5")
    x = Variable(5)

    print(x**2)
    print(x**2.0)
    print(x**-1)


def test6():    # subtraction
    print("\n test 6")
    x = Variable(2)
    y = Variable(3)

    print(x-y)
    print(y-x)
    print(x - 3)
    print(2 - y)
    print(x - 3.0)
    print(y - 2.0)

def test7():    # division
    print("\n test 7")
    x = Variable(10)
    y = Variable(2)

    print(x/y)
    print(y/x)
    print(x / 2)
    print(10 / y)
    print(x / 2.0)
    print(y / 10.0)

def test8():
    print("\n test 8")
    x = Variable(10)
    y = Variable(5)

    z = x + y
    z.grad = 1

    z._backward()
    print(f"\nz : \n{z.__repr__()}")
    print(f"\nx :\n{x.__repr__()}")
    print(f"\ny :\n{y.__repr__()}")

def test9():
    print("\n test 9")
    x = Variable(10)
    y = Variable(5)

    z = x * y
    z.grad = 1
    z._backward()
    print(f"\nz : \n{z.__repr__()}")
    print(f"\nx :\n{x.__repr__()}")
    print(f"\ny :\n{y.__repr__()}")

def test10():
    print("\n test 10")
    x = Variable(10)
    y = Variable(5)

    z = x * y
    a = z * 2
    a.grad = 1
    a.backprop()
    print(f"\na : \n{a.__repr__()}")
    print(f"\nz : \n{z.__repr__()}")
    print(f"\nx :\n{x.__repr__()}")
    print(f"\ny :\n{y.__repr__()}")

def test11():
    print("\n test 11")
    print("\tNeural network simulation")

    weights1 = []
    weights2 = []
    out_weights = []

    y_true = Variable(0.8)

    print("\nlayer 1 weights:")
    for i in range(3):
        wi = []
        print(f" neuron {i+1} weights:")
        for i in range(3):
            wi.append(Variable(np.random.rand()))
            print(wi[-1])
        weights1.append(wi)
    
    print("\nlayer 2 weights:")
    for i in range(2):
        wi = []
        print(f" neuron {i+1} weights:")
        for i in range(3):
            wi.append(Variable(np.random.rand()))
            print(wi[-1])
        weights2.append(wi)
    
    print("\noutput layer weights:")
    out_weights.append(Variable(np.random.rand()))
    print( out_weights[-1])
    out_weights.append(Variable(np.random.rand()))
    print( out_weights[-1])

    print("\ninputs:")
    inputs = []
    for i in range(3):
        inp = Variable(np.random.rand())
        inputs.append(inp)
        print(inp)
    
    losses = []
    epochs = 20
    for epoch in range(epochs):
        # forward pass
        layer1_out = []
        for neuron_weights in weights1:
            neuron_out = sum( (w*i for w,i in zip(neuron_weights, inputs)), Variable(0) ).exp()
            layer1_out.append(neuron_out)
        
        layer2_out = []
        for neuron_weights in weights2:
            neuron_out = sum( (w*i for w,i in zip(neuron_weights, layer1_out)), Variable(0) ).exp()
            layer2_out.append(neuron_out)
        
        final_out = sum( (w*i for w,i in zip(out_weights, layer2_out)), Variable(0) )
        # loss computation (MSE)
        loss = (final_out - y_true)**2

        # backward pass
        loss.grad = 1
        loss.backprop()
        losses.append(loss.value)

        # udate weights
        learning_rate = 0.01
        for neuron_weights in weights1:
            for w in neuron_weights:
                w.value -= learning_rate * w.grad
        for neuron_weights in weights2: 
            for w in neuron_weights:
                w.value -= learning_rate * w.grad
        for w in out_weights:
            w.value -= learning_rate * w.grad

        print(f"\n epoch {epoch+1} : loss = {loss.value}")
        # zero gradients
        for neuron_weights in weights1:
            for w in neuron_weights:
                w.grad = 0.0
        for neuron_weights in weights2:
            for w in neuron_weights:
                w.grad = 0.0
        for w in out_weights:
            w.grad = 0.0   

    # plot loss curve   
    plt.plot(range(epochs), losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss curve")
    plt.show() 

    
def test12():
    mlp = MLP(3, [4,4,2,1])
    x = [[1,2,3],
         [2,3,4],
         [4,5,6]]
    
    y= [6,9,15]

    mlp.fit(x,y, epochs=5)
    
    
    

if __name__ == "__main__":
    # test1()
    # test2()
    # test3()
    # test4()
    # test5()
    # test6()
    # test7()

    # test8()
    # test9()
    # test10()
    # test11()
    
    inputs = np.random.rand(50,12)
    print(f"Input shape : {inputs.shape}")
    outputs = 2*np.sum(inputs, axis = 1)
    print(f"Output shape : {outputs.shape}")

    model = Model()
    model.add(Dense(12, 24))
    model.add(ReLU())
    model.add(Dense(24,32))
    model.add(ReLU())
    model.add(Dense(32, 16))
    model.add(ReLU())
    model.add(Dense(16, 1))

    model.set(loss=MeanSquaredErrorLoss(), optimizer=Optimizer_Adam(learning_rate=0.001))

    model.finalize()

    model.train(inputs, outputs, epochs=10)

    print("\n\n\tpytorch\n\n")
    model = nn.Sequential(
        nn.Linear(12, 24),
        nn.ReLU(),
        nn.Linear(24, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    inputs_torch = torch.from_numpy(inputs).float()
    outputs_torch = torch.from_numpy(outputs).float()

    epochs = 10

    for epoch in range(epochs):
        optimizer.zero_grad()

        predictions = model(inputs_torch)    # forward pass
        loss = criterion(predictions, outputs_torch)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss = {loss.item():.6f}")

    #test12()
