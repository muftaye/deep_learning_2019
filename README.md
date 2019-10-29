# deep_learning_2019
Applied Deep Learning Course at Columbia University


3a Implement ReLu and compare against a previous activation function
------------------------------------------------------------------------

Since both have negative values, more nodes are activating creating a dense activation.  Also sigmoid and tanh tend to have very small or vanishing gradients towards the end of their functions preventing slower learning.

ReLu appears to learn better, likely due to the vanishing gradient not being issue as it would be with sigmoid and tanh.  My ReLu also appears to be quicker, likely due to the sparse activation caused by all the negative values being zeroed and the simplicity in its computation.

Since tanh is steeper between -2 and 2, larger gradient, it pushes classification more quickly to each side. 

What do I actually observe?


3b Optimizer and initializer and soup
-----------------------------------------------------------------------
Since Momentum helps accelerate gradient descent with more steep curves, we can see convergence tends to occur quicker.

Adam, unlike Momentum has an adaptive learning rate, as well as the moving average of the gradient similar to Momentum.  This can increase 

So yes, optimizers like Momentum and Adam do make a difference, especially in performance gains.

Different weight initiliaztion strategies also make a difference because they can potentially help achieve different results.  In gradient descent you may reach a local minimum and but there may be other better local minima or a global minima.  Weight initialization strategies can help find those better results.

The wrong weight initialization can also cause exploded values since matrix multiplication occurs and even vanishing outputs when initialization is too small preventing the model from learning well.


Show test of bad optimizer (SGD with no learning rate or momentum) and bad initializer (potentially try mean=0 and std=0.001).



# Creating a model
from keras.models import Sequential
from keras.layers import Dense

# Custom activation function
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects


def custom_activation(x):
    return (K.sigmoid(x) * 5) - 1

get_custom_objects().update({'custom_activation': Activation(custom_activation)})

# Usage
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation(custom_activation, name='SpecialActivation'))
print(model.summary())




from keras import optimizers

model = Sequential()
model.add(Dense(64, kernel_initializer='uniform', input_shape=(10,)))
model.add(Activation('softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
