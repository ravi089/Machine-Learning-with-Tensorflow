# Logistic Regression <MNIST hand written digit>
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters.
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# Construct weights and biases.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Loss and Optimizers.
prediction = tf.nn.softmax(tf.add(tf.matmul(X, W), b))
loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(prediction),
                                    reduction_indices = 1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Training.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0.0
        n_batches = int(mnist.train.num_examples/batch_size)
        for i in range(n_batches):
            train_x, train_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, loss], feed_dict={X: train_x,
                                                          Y: train_y})
            avg_cost += c/n_batches
        if (epoch + 1) % display_step == 0:
            print ('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))
    print ('Training Done!')
    # Test model
    prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    print ('Accuracy:', accuracy.eval({X: mnist.test.images,
                                       Y: mnist.test.labels}))
  
       
