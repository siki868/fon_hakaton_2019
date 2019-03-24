import numpy as np
import math
import tensorflow as tf

# 5.3 * gramaza polena = 
def scaleBetween(x, a, b, minX, maxX):
    return (b - a) * ((x - minX)/ (maxX - minX)) + a

def get_polen(temp, wind, hum):
    w1 = np.random.uniform(4)
    w2 = np.random.uniform(2)
    w3 = np.random.uniform(7)
    w = math.fabs(w1*temp + w2*wind + w3*hum)
    return w/13



if __name__ == "__main__":
    kolona = 5
    filename = 'csvT/weatherHistory.csv'
    # temperature_file = 'csvT/temperature.csv'
    # wind_file = 'csvT/wind_speed.csv'
    # X = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=(4))
    # print(np.amin(X))

    # hum_x = np.genfromtxt(hum_file, delimiter=',', usecols=(kolona), skip_header=2, filling_values=70.0)
    # temperature_x = np.genfromtxt(temperature_file, delimiter=',', usecols=(kolona), skip_header=2, filling_values=65.0)
    # wind_x = np.genfromtxt(wind_file, delimiter=',', usecols=(kolona), skip_header=2, filling_values=8.0)
    nb_features = 3
    temperature_x = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=(3))
    hum_x = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=(5))
    wind_x = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=(6))

    # temperature_x = temperature_x - 273.15
    # wind_x = wind_x * 2.7

    dX = np.stack((temperature_x, hum_x, wind_x))
    dY = []
    for i in range(len(temperature_x)):
        dY.append(get_polen(dX[0][i], dX[1][i], dX[2][i]))
    dY = np.array(dY)
    
    dX = dX.T

    nb_samples = dX.shape[0]
    indices = np.random.permutation(nb_samples)
    dX = dX[indices]
    dY = dY[indices]

    X = tf.placeholder(shape=(None, nb_features), dtype=tf.float32)
    Y = tf.placeholder(shape=(None), dtype=tf.float32)
    w = tf.Variable(tf.zeros(nb_features))
    bias = tf.Variable(0.0)

    w_col = tf.reshape(w, (nb_features, 1))
    hyp = tf.add(tf.matmul(X, w_col), bias)

    # Korak 3: Funkcija troška i optimizacija.
    Y_col = tf.reshape(Y, (-1, 1))
    loss = tf.reduce_mean(tf.square(hyp - Y_col))
    opt_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Izvršavamo 50 epoha treninga.
        nb_epochs = 50
        for epoch in range(nb_epochs):
            
            # Stochastic Gradient Descent.
            epoch_loss = 0
            for sample in range(nb_samples):
                feed = {X: dX[sample].reshape((1, nb_features)), 
                        Y: dY[sample]}
                _, curr_loss = sess.run([opt_op, loss], feed_dict=feed)
                epoch_loss += curr_loss
            
            # U svakoj petoj epohi ispisujemo prosečan loss.
            epoch_loss /= nb_samples
            if (epoch + 1) % 5 == 0:
                print('Epoch: {}/{}| Avg loss: {:.5f}'.format(epoch+1, nb_epochs, 
                                                    epoch_loss))

    w_val = sess.run(w)
    bias_val = sess.run(bias)
    print('w = ', w_val, 'bias = ', bias_val)