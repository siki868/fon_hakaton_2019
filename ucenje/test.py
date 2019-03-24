import numpy as np
import math
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from numpy.linalg import inv

# 5.3 * gramaza polena = 
def scaleBetween(x, a, b, minX, maxX):
    return (b - a) * ((x - minX)/ (maxX - minX)) + a

# def get_polen(temp, hum):
#     w1 = np.random.uniform(0.03220143 - 0.1, 0.03220143 + 0.1)
#     w2 = -5.325593
#     w3 = np.random.uniform(-0.17517795 - 0.1, -0.17517795 + 0.1)
#     w = math.fabs(w1*temp +w3*hum + 10.737404)
#     return w


def get_polen(temp, wind, hum):
    w1 = np.random.uniform(4)
    w2 = np.random.uniform(2)
    w3 = np.random.uniform(7)
    w = math.fabs(w1*temp + w2*wind + w3*hum)
    return w/12



if __name__ == "__main__":
    kolona = 5
    filename = 'weatherHistory.csv'
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

    dX = np.stack((temperature_x, wind_x, hum_x))
    dY = []
    for i in range(len(temperature_x)):
        dY.append(get_polen(dX[0][i], dX[1][i], dX[2][i]))
    dY = np.array(dY)
    
    dX = dX.T
    
    
    #mesanje
    nb_samples = dX.shape[0]
    indices = np.random.permutation(nb_samples)
    dX = dX[indices]
    dY = dY[indices]
    
    
#     dX = (dX- np.mean(dX, axis=0)) / np.std(dX, axis=0)
#     dY = (dY - np.mean(dY)) / np.std(dY)
    
#     ax = Axes3D(plt.figure())
#     ax.scatter(dX[:,0], dX[:,1], dY)
#     ax.set_xlabel('Temperatura')
#     ax.set_ylabel('Vlaznost vazduha')
#     ax.set_zlabel('Polen')
  

  
    print(dY)
    print(np.amax(dY))
    
    X = tf.placeholder(shape=(None, nb_features), dtype=tf.float32)
    Y = tf.placeholder(shape=(None), dtype=tf.float32)
    w = tf.Variable(tf.zeros(nb_features))
    bias = tf.Variable(0.0)

    w_col = tf.reshape(w, (nb_features, 1))
    hyp = tf.add(tf.matmul(X, w_col), bias)

    # Korak 3: Funkcija troška i optimizacija.
    Y_col = tf.reshape(Y, (-1, 1))
    loss = tf.reduce_mean(tf.square(hyp - Y_col))
    opt_op = tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        nb_epochs = 5
        for epoch in range(nb_epochs):
            
            epoch_loss = 0
            for sample in range(nb_samples):
                feed = {X: dX[sample].reshape((1, nb_features)), 
                        Y: dY[sample]}
                _, curr_loss = sess.run([opt_op, loss], feed_dict=feed)
                epoch_loss += curr_loss
            
            epoch_loss /= nb_samples
            if (epoch + 1) % 5 == 0:
                print('Epoch: {}/{}| Avg loss: {:.5f}'.format(epoch+1, nb_epochs, 
                                                    epoch_loss))

        w_val = sess.run(w)
        bias_val = sess.run(bias)
        print('w = ', w_val, 'bias = ', bias_val)
#         x1s, x2s = np.meshgrid(np.linspace(-2.0, 35, 100), 
#                          np.linspace(0, 100, 100))
#         ys = x1s * w_val[0] + x2s * w_val[1] + bias_val
#         ax.plot_surface(x1s, x2s, ys, color='g', alpha=0.5)

        final_loss = sess.run(loss, feed_dict={X: dX, Y: dY})
        print('Finalni loss: {:.5f}'.format(final_loss))
        
        svi_x = []
        svi_y = []
        plt.xlabel('Prave vrednosti')
        plt.ylabel('Pretpostavljene vrednosti')
        for i in range(200):
          indx = np.random.randint(len(dX))
          svi_x.append(dY[indx])
          svi_y.append(math.fabs(dX[indx][0]*w_val[0] + dX[indx][1]*w_val[1] + dX[indx][2]*w_val[2] + bias_val))


        prava_x = np.arange(0, 10, 1)
        prava_y = np.arange(0, 10, 1)

        svi_x = np.array(svi_x)
        svi_y = np.array(svi_y)
        svi_x = np.array((np.ones(200), svi_x))
        svi_x = svi_x.T

        w = np.dot(inv(np.dot(svi_x.T, svi_x)), np.dot(svi_x.T, svi_y))
        print(f'Theta: {w}')
        linija = w[0] + w[1]*svi_x[:,1]
        plt.scatter(svi_x[:,1], svi_y, c='b')
        obicna = plt.plot(prava_x, prava_y, c='r')
        lr, = plt.plot(svi_x[:,1], linija, c='g', label='LR')
        plt.legend(handles=[lr])
        plt.show()