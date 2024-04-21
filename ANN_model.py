import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import time

pd.set_option('display.max_columns', None)

# MLP model
class MLP(nn.Module):
  # number of input features, number of nodes in each hidden layer (list), 
  # number of output features, dropout rate, activation function, loss function
  def __init__(self, 
               input_size, 
               hidden_sizes, 
               output_size, 
               dropout_rate=0.5, 
               activation=F.relu, 
               loss=nn.CrossEntropyLoss()):
    super(MLP, self).__init__()
    self.activation = activation
    self.loss = loss
  
    self.hidden = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0], bias=True)])
    self.dropout = nn.ModuleList([nn.Dropout(dropout_rate)])

    for i in range(1, len(hidden_sizes)):
      self.hidden.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i], bias=True))
      self.dropout.append(nn.Dropout(dropout_rate))

    self.output = nn.Linear(hidden_sizes[-1], output_size, bias=True)

  def forward(self, x):
    for layer in self.hidden:
      x = self.activation(layer(x))
    
    if self.loss != nn.CrossEntropyLoss():
      x = F.log_softmax(x, dim=1)
      
    return self.output(x)
  

# CNN model
class CNN(nn.Module):
  # number of nodes in each hidden layer (list), number of output features, 
  # dropout rate, number of layers, activation function, loss function
  def __init__(self, kernel_sizes, 
               out_channels, 
               dropout_rate=0.5, 
               layers=1, 
               activation=F.relu, 
               loss=nn.CrossEntropyLoss(), 
               stride=1, 
               padding=1, 
               pool_size=2,
               pool_stride=2):
    super(CNN, self).__init__()

    self.activation = activation
    self.loss = loss
    self.layers = layers
    self.pool_size = pool_size


    self.conv1 = nn.Conv2d(1, out_channels, kernel_sizes[0], stride=stride, padding=padding)
    self.pool1 = nn.MaxPool2d(pool_size, pool_stride)

    conv_layer = (28 - kernel_sizes[0] + 2*padding) // stride + 1
    pool_layer = (conv_layer - pool_size) // pool_stride + 1

    #print("conv, pool: ", conv_layer, pool_layer)

    self.conv1_drop = nn.Dropout2d(dropout_rate)
    for i in range(1, layers):
      self.add_module('conv'+str(i+1), nn.Conv2d(out_channels, out_channels, kernel_sizes[i], stride=stride, padding=padding))
      self.add_module('conv'+str(i+1)+'_pool', nn.MaxPool2d(pool_size, pool_stride))
      
      conv_layer = (pool_layer - kernel_sizes[i] + 2*padding) // stride + 1
      pool_layer = (conv_layer - pool_size) // pool_stride + 1

      #print("conv, pool: ", conv_layer, pool_layer)

      self.add_module('conv'+str(i+1)+'_drop', nn.Dropout2d(dropout_rate))
      
    self.num_flatten_nodes = out_channels * (pool_layer ** 2)
    #print("num_flatten_nodes: ", self.num_flatten_nodes)
    self.fc1 = nn.Linear(self.num_flatten_nodes, 10)
    
  def forward(self, x):
    x = self.activation(self.conv1(x))
    x = self.pool1(x)
    x = self.conv1_drop(x)

    for i in range(1, self.layers):
      x = self.activation(self._modules['conv'+str(i+1)](x))
      x = self._modules['conv'+str(i+1)+'_pool'](x)
      x = self._modules['conv'+str(i+1)+'_drop'](x)

    #print("shape of x: ", x.shape)
    x = x.view(-1, self.num_flatten_nodes)
    #print("shape of x (2d): ", x.shape)
  
    x = self.activation(self.fc1(x))
    x = F.dropout(x, training=self.training)
    return x
  
  
def show_images(images, n=10):
  for i in range(n):
    plt.subplot(2, n//2, i+1)
    plt.imshow(images[i].reshape(28, 28), cmap='Blues')
    plt.axis('off')

############### Copied from the lecture notes ####################
# Training function
def train_ANN_model(num_epochs, training_data, device, CUDA_enabled, is_MLP, ANN_model, loss_func, optimizer, mini_batch_size=100):
    train_losses = []
    ANN_model.train() # to set the model in training mode. Only Dropout and BatchNorm care about this flag.
    for epoch_cnt in range(num_epochs):
        for batch_cnt, (images, labels) in enumerate(training_data):
            # Each batch contain batch_size (100) images, each of which 1 channel 28x28
            # print(images.shape) # the shape of images=[100,1,28,28]
            # So, we need to flatten the images into 28*28=784
            # -1 tells NumPy to flatten to 1D (784 pixels as input) for batch_size images
            if (is_MLP):
                # the size -1 is inferred from other dimensions
                images = images.reshape(-1, 784) # or images.view(-1, 784) or torch.flatten(images, start_dim=1)

            if (device.type == 'cuda' and CUDA_enabled):
                images = images.to(device) # moving tensors to device
                labels = labels.to(device)

            optimizer.zero_grad() # set the cumulated gradient to zero
            # print("shape of images: ", images.shape)
            output = ANN_model(images) # feedforward images as input to the network
            # print("ANN model parameters: ", ANN_model.parameters())
            # print("shape of output: ", output.shape)
            # print("shape of labels: ", labels.shape)
            loss = loss_func(output, labels) # computing loss

            #print("Loss: ", loss)
            #print("Loss item: ", loss.item())
            train_losses.append(loss.item())
            # PyTorch's Autograd engine (automatic differential (chain rule) package) 
            loss.backward() # calculating gradients backward using Autograd
            optimizer.step() # updating all parameters after every iteration through backpropagation

            # Display the training status
            if (batch_cnt+1) % mini_batch_size == 0:
                print(f"Epoch={epoch_cnt+1}/{num_epochs}, batch={batch_cnt+1}/{num_train_batches}, loss={loss.item()}")
    return train_losses

# Testing function
def test_ANN_model(device, CUDA_enabled, is_MLP, ANN_model, testing_data, mini_batch_size=100):
    # torch.no_grad() is a decorator for the step method
    # making "require_grad" false since no need to keeping track of gradients    
    predicted_digits=[]
    # torch.no_grad() deactivates Autogra engine (for weight updates). This help run faster
    with torch.no_grad():
        ANN_model.eval() # # set the model in testing mode. Only Dropout and BatchNorm care about this flag.
        total_samples, total_correct = 0, 0
        for batch_cnt, (images, labels) in enumerate(testing_data):
            if (is_MLP):
                images = images.reshape(-1, 784) # or images.view(-1, 784) or torch.flatten(images, start_dim=1)

            if (device.type == 'cuda' and CUDA_enabled):
                images = images.to(device) # moving tensors to device
                labels = labels.to(device)
            
            output = ANN_model(images)
            _, prediction = torch.max(output,1) # returns the max value of all elements in the input tensor
            predicted_digits.append(prediction)
            num_samples = labels.shape[0]
            num_correct = (prediction==labels).sum().item()

            total_samples += num_samples
            total_correct += num_correct

            if (batch_cnt+1) % mini_batch_size == 0:
                print(f"batch={batch_cnt+1}/{num_test_batches}")
                
        accuracy = total_correct/total_samples
        print("> Number of samples=", num_samples, "number of correct prediction=", num_correct, "accuracy=", accuracy)
    return predicted_digits, accuracy

# check cpu and setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if (torch.cuda.is_available()):
    print("The CUDA version is", torch.version.cuda)
    # Device configuration: use GPU if available, or use CPU
    cuda_id = torch.cuda.current_device()
    print("ID of the CUDA device:", cuda_id)
    print("The name of the CUDA device:", torch.cuda.get_device_name(cuda_id))
    print("GPU will be utilized for computation.")
else:
    print("CUDA is not supported in your machine. Only CPU will be used for computation.")

# Using Apple's Metal Performance Shaders (MPS) for computation
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# if (torch.backends.mps.is_available()):
#     print("MPS will be utilized for computation.")
#     x = torch.ones(1, device=device)
#     print("The device is:", x.device)
# else:
#     print("MPS is not supported in your machine. Only CPU will be used for computation.")


# convert the image into tensor and normalize the pixel values
transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# Download and load the dataset from the torch vision library to the directory specified by current directory
train_dataset=datasets.MNIST(root='./data', train=True, transform=transforms, download=True)
test_dataset=datasets.MNIST(root='./data', train=False, transform=transforms, download=False)
print("> Shape of training data:", train_dataset.data.shape)
print("> Shape of testing data:", test_dataset.data.shape)
print("> Classes:", train_dataset.classes)


# Use DataLoader to load the data in batches
mini_batch_size=100
train_loader=DataLoader(dataset=train_dataset, batch_size=mini_batch_size, shuffle=True)
test_loader=DataLoader(dataset=test_dataset, batch_size=mini_batch_size, shuffle=False)
num_train_batches, num_test_batches = len(train_loader), len(test_loader)
print("> Mini batch size:", mini_batch_size)
print("> Number of training batches:", num_train_batches)
print("> Number of testing batches:", num_test_batches)


# display image from the first batch
images, labels = next(iter(train_loader))
show_images(images)


# ### MLP model hyperparameters ###
# # Define hyperparameters
# input_size = 784
# hidden_sizes = [[14, 7, 3], [14, 7], [14, 3], [14]]
# output_size = 10
# dropout_rates = [0.1, 0.3, 0.5]
# activation = [F.relu, F.tanh, F.sigmoid]
# loss = nn.CrossEntropyLoss()
# num_epochs = [1, 3, 5, 10]
# alpha = [0.1, 0.01, 0.001]  # learning rate
# gamma = [0.1, 0.2, 0.3]  # momentum
# MLP_optimizer = [optim.SGD, optim.Adam, optim.RMSprop]

# is_MLP = True
# CUDA_enabled = True

# # Search for the best hyperparameters
# for hidden_size in hidden_sizes:
#   for dropout_rate in dropout_rates:
#     for act in activation:
#       for opt in MLP_optimizer:
#         for a in alpha:
#           for g in gamma:
#             MLP_model = MLP(input_size, hidden_size, output_size, dropout_rate, act, loss)
#             if (device.type == 'cuda' and CUDA_enabled):
#               MLP_model = MLP_model.to(device=device)

                
#             if (opt == optim.SGD):
#               optimizer = opt(MLP_model.parameters(), lr=a, momentum=g)
#             else:
#               optimizer = opt(MLP_model.parameters(), lr=a)

#             # Track training time
#             for num_epoch in num_epochs:
#               start_time = time.time()
#               train_losses = train_ANN_model(num_epoch, train_loader, device, CUDA_enabled, is_MLP, MLP_model, loss, optimizer)
#               end_time = time.time()

#               predicted_digits, accuracy = test_ANN_model(device, CUDA_enabled, is_MLP, MLP_model, test_loader)

#               # Present the parameters in a pandas dataframe
#               df = pd.DataFrame({'number of epochs': [num_epoch], 'hidden_size': [hidden_size], 'activation': [act], 'optimizer': [opt], 'alpha': [a], 'gamma': [g], 'accuracy': [accuracy], 'training_time': [end_time-start_time]})
#               print(df)

#               # Save to a csv file
#               df.to_csv('MLP_hyperparameters.csv', mode='a', header=False)

### CNN model hyperparameters ###
# define extra hyperparameters for CNN, the rest are the same as MLP

# kernel_sizes = [[5], [5, 3], [7, 5, 3]] # number of filters
# feature_maps = [10]
# dropout_rates = [0.4, 0.5]
# activation = [F.relu, F.tanh, F.sigmoid]
# loss = nn.CrossEntropyLoss()
# num_epochs = [5, 10, 15]
# alpha = [0.01, 0.001]  # learning rate
# gamma = [0.8, 0.9]  # momentum
# MLP_optimizer = [optim.SGD, optim.Adam, optim.RMSprop]
# stride = [1, 2]
# padding = [1, 2]
# pool_size = 4 # kernel size for pooling
# pool_stride = 2

# best params
params = [[5], 10, 0.5, F.relu, optim.SGD, 0.001, 0.9, 1, 1]
num_epoch = 5
loss = nn.CrossEntropyLoss()
pool_size = 4
pool_stride = 2

is_MLP = False
CUDA_enabled = True

# Create grid search combination using itertools cartesian product
# grid_search = itertools.product(kernel_sizes, feature_maps, dropout_rates, activation, MLP_optimizer, alpha, gamma, stride, padding)
# Search for the best hyperparameters

kernel_size, feature_maps, dropout_rate, act, opt, a, g, s, p = params

CNN_model = CNN(kernel_size, feature_maps, dropout_rate, len(kernel_size), act, loss, s, p, pool_size, pool_stride)

if (device.type == 'cuda' and CUDA_enabled):
  CNN_model = CNN_model.to(device=device)

alpha, gamma = 0.001, 0.9

if (opt == optim.SGD):
  optimizer = opt(CNN_model.parameters(), lr=a, momentum=g)
else:
  optimizer = opt(CNN_model.parameters(), lr=a)

# Training 
start_time = time.time()
train_losses = train_ANN_model(num_epoch, train_loader, device, CUDA_enabled, is_MLP, CNN_model, loss, optimizer)
end_time = time.time()

predicted_digits, accuracy = test_ANN_model(device, CUDA_enabled, is_MLP, CNN_model, test_loader)

# Present the parameters in a pandas dataframe
df = pd.DataFrame({'number of epochs': [num_epoch], 'kernel_size': [kernel_size], 'feature_maps': [feature_maps], 'drop_out_rate': [dropout_rate], 'activation': [act.__name__], 'optimizer': [opt.__name__], 'alpha': [a], 'gamma': [g], 'stride': [s], 'padding':[p], 'accuracy': [accuracy], 'training_time': [end_time-start_time]})
print(df)

# Save to a csv file
# df.to_csv('CNN_hyperparameters.csv', mode='a', header=False)