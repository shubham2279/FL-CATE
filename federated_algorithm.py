from datetime import datetime
import random as ran
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
# recipe is used to calculate object size in bytes and suggested to be used by
# python documentation
import recipe as re
import sys

# *****************************************************************************

# constant variables
# learning rate
ALPHA = 0.1
FLUX = 0.15 # percentage value 0-1
ENERGY = 4 # energy units cost per bit
WEIGHTS = 15 # max weight
NODES = 6 # num of nodes in network including sink
MAX_COST = 1000
TRANSMITION_COST = 10
ITERATIONS = 5
# *****************************************************************************
# setup of the neural network model's layers initialisation
class Model_Node(Model):
    # initialise model using the tensorflow recommended layers
    
    grads = []
    num_of_vars = 0
    node_id = None
    
    def __init__(self, nid):
        super(Model_Node, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)
        self.num_of_vars = -1
        self.node_id = nid

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)
    def __getitem__(self, item): return self
    # initialising loss computatuin function
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)

    # initialising stochastic gradient descent function
    optimizer = tf.keras.optimizers.SGD(learning_rate=ALPHA)

    # setting up the metrics collection tensors
    train_loss = tf.keras.metrics.Mean(name='train_loss'+str(node_id))
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy'+str(node_id))

    test_loss = tf.keras.metrics.Mean(name='test_loss'+str(node_id))
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='test_accuracy'+str(node_id))
    
    # model train function
    @tf.function
    def train_step(self, images, labels, glo=False, g=None):
        # if glo=true then it updates the global model using the gradient
        # given in argument g
        if glo:
            self.optimizer.apply_gradients(zip(g, self.trainable_variables),
                                           experimental_aggregate_gradients=True)
            return
        with tf.GradientTape() as tape:
            predictions = self(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.trainable_variables);
        if self.num_of_vars < 0: self.num_of_vars = len(self.trainable_variables)
        # applies training gradients using stochastic gradient descend
        self.optimizer.apply_gradients(zip(gradients,
                                           self.trainable_variables),
                                       experimental_aggregate_gradients=False)
        # calculates loss and accuracy
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)
        x = self.num_of_vars
        self.grads = gradients[-x:]
        # returns the gradient
        return self.grads

    # model predict function
    @tf.function
    def test_step(self, images, labels):
        # predicts the label
        predictions = self(images, training=False)
        t_loss = self.loss_object(labels, predictions)
        # calculates loss and accuracy
        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

# *****************************************************************************

# function for averaging the gradients
def avgs(matrices = []):
    segmentIDs = [0] * len(matrices)
    return tf.math.segment_mean(matrices, segmentIDs)

# function returning the shortest paths for each node
def shortest_paths():
    paths = []
    for i in range(1, NODES):
        path_cost = MAX_COST
        path = None
        for p in fwd_table[i]:
            cost = len(p) * TRANSMITION_COST
            for hop in p: cost += edges[hop]
            if cost < path_cost: path_cost, path = cost, p
        paths.append((path, path_cost))
    return paths

# *****************************************************************************

# array of nodes in network
nodes = [0, 1, 2, 3, 4, 5]
# dictionary of edges in the graph and their weights
edges = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0,
         'f': 0, 'g': 0, 'h': 0, 'i': 0}
# forwarding table containing all routing paths
fwd_table = [None, ['a', 'cb'], ['b', 'ca'],
             ['da', 'dcb', 'iea', 'iecb', 'ifb', 'ifca'],
             ['ea', 'ecb', 'fb', 'fca'],
             ['gb', 'gca', 'hfb', 'hfca', 'hea', 'hecb']]

# *****************************************************************************

start_time =  datetime.now().strftime("%H:%M:%S")

# sim run
mnist = tf.keras.datasets.mnist
print("loading dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(5)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

init = None
nds = [[], [], [], [], []]
i = 0
for img, lbl in train_ds:
        if i == 0: init = (img, lbl); i += 1; continue
        nds[i%5].append((img, lbl))
        i += 1
print("dataset split")
# *****************************************************************************
print("initialiasing nodes")
nodes = []
for i in range(0, NODES):
    nodes.append(Model_Node(i)); print("node", i, "initialised")
    nodes[-1].train_loss.reset_states()
    nodes[-1].train_accuracy.reset_states()
    nodes[-1].test_loss.reset_states()
    nodes[-1].test_accuracy.reset_states()
# *****************************************************************************
j = 0
grad, acc, bits = [], [], []
gs = [[], [], [], [], [], []]
# initial train of all nodes
for i in range(0, NODES):
    nodes[i].train_step(init[0], init[1])
    nodes[i].set_weights(nodes[0].get_weights())
# main training and testing loop
for i in range(0, len(nds[0])):
    print("Iteration: ", i)
    # train the local models
    for j in range(1, NODES):
        print("training local model ", j)
        g = nodes[j].train_step(nds[j-1][i][0], nds[j-1][i][1])
        # save gradients in list 
        for k in range(0, 6): gs[k].append(g[k])
    print("computing avgs")
    for j in range(0, 6): grad.append(avgs(gs[j])[0])
    print("avgs computed")
    # update the global model
    nodes[0].train_step(None, None, glo=True, g=grad)
    print("global model updated")
    # update local models on the updated global
    for j in range(1, NODES): nodes[j].set_weights(nodes[0].get_weights())
    # test updated global model
    for ti, tl in test_ds: nodes[0].test_step(ti, tl)
    # gather metrics for graphs
    acc.append(nodes[0].test_accuracy.result() * 100)
    print("Accuracy: ", acc[-1])
    size = 0
    for g in grad: size += re.total_size(g)
    bits.append(size)
    print("Bits: ", size)
    # reset variables for next iteration
    grad = []
    gs = [[], [], [], [], [], []]
    nodes[0].test_loss.reset_states()
    nodes[0].test_accuracy.reset_states()
# run another last iteration to account for the leftover records
for i in range(2, NODES):
    g = nodes[i].train_step(nds[i-1][-1][0], nds[i-1][-1][1])
    for k in range(0, 6): gs[k].append(g[k])
for j in range(0, 6): grad.append(avgs(gs[j])[0])
# final update of the global model
nodes[0].train_step(None, None, glo=True, g=grad)
# test updated global model
for ti, tl in test_ds: nodes[0].test_step(ti, tl)
# gather metrics for graphs
acc.append(nodes[0].test_accuracy.result() * 100)
print("Accuracy: ", acc[-1])
size = 0
for g in grad: size += re.total_size(g)
bits.append(size)
print("Bits: ", size)
# *****************************************************************************
# plot and save graphs to file
print("plotting figures")
plt.figure(1, figsize=(20,20))
plt.title('FL Accuracy', fontsize=25)
plt.plot(range(0, len(acc)), acc, alpha=0.8)
plt.savefig("baseline_acc.png", bbox_inches='tight')

plt.figure(2, figsize=(20,20))
plt.title('FL size', fontsize=25)
plt.plot(range(0, len(bits)), bits, alpha=0.8)
plt.savefig("baseline_size.png", bbox_inches='tight')


print("script finished running")
print("start time: ", start_time, " --------- end time:",
      datetime.now().strftime("%H:%M:%S"))
