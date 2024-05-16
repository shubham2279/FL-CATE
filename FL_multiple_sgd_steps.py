from datetime import datetime
import random as ran
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import sys

# *****************************************************************************

# constant variables
# learning rate
ALPHA = 0.1
FLUX = 0.15 # percentage value 0-1
ENERGY = 2 # energy units cost per bit
WEIGHTS = 15 # max weight
NODES = 6 # num of nodes in network including sink
MAX_COST = 1000
TRANSMITION_COST = 10
SGD_STEPS = 5
# *****************************************************************************
# array of nodes in network
paths = {3 : "310", 4: "420", 5: "520", 1 : "10", 2 : "20"}
costs = {0: 0, 1: 3, 2: 1, 3: 0.5, 4: 2 , 5:1.5}
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

# function returning the enegry cost
def compute_transmission_cost(node):
    cost = 0
    for c in paths[node]: cost += (costs[int(c)] * ENERGY)
    return cost
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
ecost = 0
ecosts, acc, los, bits = [], [], [], []
gs = [[], [], [], [], [], []]
# initial train of all nodes
for i in range(0, NODES):
    nodes[i].train_step(init[0], init[1])
    if i == 0: continue
    nodes[i].set_weights(nodes[0].get_weights())
# main training and testing loop
for i in range(0, len(nds[0])):
    idiv = i % SGD_STEPS
    print("Iteration: ", i)
    # train the local models
    for j in range(1, NODES):
        print("training local model ", j)
        g = nodes[j].train_step(nds[j-1][i][0], nds[j-1][i][1])
        # save gradients in list 
        #for k in range(0, 6): gs[k].append(g[k])
    if idiv == (SGD_STEPS - 1):
        ws = nodes[0].get_weights()
        for j in range(1, NODES): 
            g = nodes[j].get_weights()
            for k in range(0, 6): gs[k].append(g[k])
        for j in range(0, 6):
            for k in range(0, 5): gs[j][k] = ws[j] - gs[j][k]
        for j in range(0, 6): ws[j] = (gs[j][0] + gs[j][1] + gs[j][2]
                                       + gs[j][3] + gs[j][4]) / 5
        nodes[0].optimizer.apply_gradients(zip(ws,nodes[0].trainable_variables),
                                           experimental_aggregate_gradients=True)
        # gather metrics for graphs
        for ti, tl in test_ds: nodes[0].test_step(ti, tl)
        acc.append(nodes[0].test_accuracy.result() * 100)
        los.append(nodes[0].test_loss.result())
        print("Accuracy: ", acc[-1])
        print("Loss: ", los[-1])
        for j in range(1, 6):
            ct = compute_transmission_cost(j)
            for k in range(0, 6): ecost += (gs[k][j-1].nbytes * 8 * ct)
        ecosts.append(ecost)
        print("Energy cost: ", ecost)
        size = 0
        for g in gs: 
            for w in g: size += w.nbytes * 8
        bits.append(size)
        print("Bits: ", bits[-1])
        gs = [[], [], [], [], [], []]
        for j in range(1, NODES):
            nodes[j].set_weights(nodes[0].get_weights())
        nodes[0].test_loss.reset_states()
        nodes[0].test_accuracy.reset_states()
# run another last iteration to account for the leftover records
print("Iteration: ", i+1)
print("Remainder: ", i % SGD_STEPS)
ws = nodes[0].get_weights()
for j in range(1, NODES): 
    g = nodes[j].get_weights()
    for k in range(0, 6): gs[k].append(g[k])
for j in range(0, 6):
    for k in range(0, 5): gs[j][k] = ws[j] - gs[j][k]
for j in range(0, 6): ws[j] = (gs[j][0] + gs[j][1] + gs[j][2]
                               + gs[j][3] + gs[j][4]) / 5
nodes[0].optimizer.apply_gradients(zip(ws,nodes[0].trainable_variables),
                                   experimental_aggregate_gradients=True)
# gather metrics for graphs
for ti, tl in test_ds: nodes[0].test_step(ti, tl)
acc.append(nodes[0].test_accuracy.result() * 100)
los.append(nodes[0].test_loss.result())
print("Accuracy: ", acc[-1])
print("Loss: ", los[-1])
for j in range(1, 6):
    ct = compute_transmission_cost(j)
    for k in range(0, 6): ecost += (gs[k][j-1].nbytes * 8 * ct)
ecosts.append(ecost)
print("Energy cost: ", ecost)
size = 0
for g in gs: 
    for w in g: size += w.nbytes * 8
bits.append(size)
print("Bits: ", bits[-1])
# *****************************************************************************
# plot and save graphs to file
print("plotting figures")
plt.figure(1, figsize=(20,20))
plt.title('FL Accuracy', fontsize=25)
plt.plot(range(0, len(acc)), acc, alpha=0.8)
pngname = "baseline_acc_%d" % SGD_STEPS
pngname += "sgd.png"
print(pngname)
plt.savefig(pngname, bbox_inches='tight')

plt.figure(2, figsize=(20,20))
plt.title('FL size', fontsize=25)
plt.plot(range(0, len(bits)), bits, alpha=0.8)
pngname = "baseline_size_%d" % SGD_STEPS
pngname += "sgd.png"
print(pngname)
plt.savefig(pngname, bbox_inches='tight')

plt.figure(3, figsize=(20,20))
plt.title('FL Energy Consupmtioon', fontsize=25)
plt.plot(range(0, len(ecosts)), ecosts, alpha=0.8)
pngname = "baseline_energy_%d" % SGD_STEPS
pngname += "sgd.png"
print(pngname)
plt.savefig(pngname, bbox_inches='tight')
# *****************************************************************************
print("script finished running")
print("start time: ", start_time, " --------- end time:",
      datetime.now().strftime("%H:%M:%S"))
