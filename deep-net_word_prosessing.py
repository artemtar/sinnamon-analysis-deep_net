import tensorflow as tf
import pickle
import numpy as np
data = []
with open('./data/train_and_test.pickle','rb') as f:
    data = pickle.load(f)

train_x, train_y, test_x, test_y = data

print(train_x[1])

n_nodes_l1 = 1500
n_nodes_l2 = 1500
n_nodes_l3 = 1500

n_classes = 2
batch_size = 50

x = tf.placeholder('float', [None, len(train_x[1])])
y = tf.placeholder('float')

# model


def deepWeb(data):
    layer1 = {'weights': tf.Variable(tf.random_normal([len(train_x[1]), n_nodes_l1])),
              'biases': tf.Variable(tf.random_normal([n_nodes_l1]))}
    layer2 = {'weights': tf.Variable(tf.random_normal([n_nodes_l1, n_nodes_l2])),
              'biases': tf.Variable(tf.random_normal([n_nodes_l2]))}
    layer3 = {'weights': tf.Variable(tf.random_normal([n_nodes_l2, n_nodes_l3])),
              'biases': tf.Variable(tf.random_normal([n_nodes_l3]))}

    layerOut = {'weights': tf.Variable(tf.random_normal([n_nodes_l3, n_classes])),
            'biases': tf.Variable(tf.random_normal([n_classes]))}


    #activation function
    l1 = tf.add(tf.matmul(x, layer1['weights']), layer1['biases'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, layer2['weights']), layer2['biases'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2, layer3['weights']), layer3['biases'])
    l3 = tf.nn.relu(l3)

    outl = tf.add(tf.matmul(l3, layerOut['weights']), layerOut['biases'])

    return outl

def train(input):
    prediction = deepWeb(input)
    #how label is decided?
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

    n_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):
            epoch_cost = 0
            i = 0
            while i < len(train_x):
                batch_x = np.array(train_x[i : i + batch_size])
                batch_y = np.array(train_y[i : i + batch_size])
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_cost += c
                i += batch_size

            print('Epoch', epoch + 1, 'out of', n_epochs, 'cost', epoch_cost)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, "float"))
            print("Accuracy ", accuracy.eval({x: test_x, y: test_y}))

train(x)
