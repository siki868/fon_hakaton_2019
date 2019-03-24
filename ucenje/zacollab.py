import numpy as np
import math
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



# 5.3 * gramaza polena = 
def scaleBetween(x, a, b, minX, maxX):
    return (b - a) * ((x - minX)/ (maxX - minX)) + a

# def get_polen(temp, hum):
#     w1 = np.random.uniform(0.03220143 - 0.1, 0.03220143 + 0.1)
#     w2 = -5.325593
#     w3 = np.random.uniform(-0.17517795 - 0.1, -0.17517795 + 0.1)
#     w = math.fabs(w1*temp +w3*hum + 10.737404)
#     return w


def get_polen(temp, wind):
    w1 = np.random.uniform(4)
    w2 = np.random.uniform(2)
    w = math.fabs(w1*temp + w2*wind)
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
    nb_features = 2
    temperature_x = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=(3))
    wind_x = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=(6))

    # temperature_x = temperature_x - 273.15
    # wind_x = wind_x * 2.7

    dX = np.stack((temperature_x, wind_x))
    dY = []
    for i in range(len(temperature_x)):
        dY.append(get_polen(dX[0][i], dX[1][i]))
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
    print(dX)
    print(np.amax(dY))
    
    nb_samples = dX.shape[0]
    indices = np.random.permutation(nb_samples)
    dX = dX[indices]
    dY = dY[indices]
    
    
    

    # Korak 2: Model.
    # Primetiti 'None' u atributu shape placeholdera i -1 u 'tf.reshape'.
    print('Ovde1')
    X = tf.placeholder(shape=(None, nb_features), dtype=tf.float32)
    Y = tf.placeholder(shape=(None), dtype=tf.float32)
    w = tf.Variable(tf.zeros(nb_features))
    bias = tf.Variable(0.0)

    print('Ovde2')
    w_col = tf.reshape(w, (nb_features, 1))
    hyp = tf.add(tf.matmul(X, w_col), bias)

    # Korak 3: Funkcija troška i optimizacija.
    Y_col = tf.reshape(Y, (-1, 1))
    loss = tf.reduce_mean(tf.square(hyp - Y_col))
    opt_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
    print('Ovde3')
    # Korak 4: Trening.
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      # Izvršavamo 50 epoha treninga.
      nb_epochs = 10
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

      # Ispisujemo i plotujemo finalnu vrednost parametara.
      w_val = sess.run(w)
      bias_val = sess.run(bias)
      print('w = ', w_val, 'bias = ', bias_val)
      
      
      
      neko_x = dX[:200]
      neko_y = dY[:200]
    
      ax = Axes3D(plt.figure())
      ax.set_xlabel('Temperatura')
      ax.set_ylabel('Vlaznost')
      ax.set_zlabel('Gramaza polena po metru kubnom')

      
      p = ax.scatter(neko_x[:,0], neko_x[:,1], neko_y, c=neko_y*10)
      cbar=plt.colorbar(p, ax=ax)
      cbar.set_label('Polen')
      
      x1s, x2s = np.meshgrid(np.linspace(0.0, 35.0, 100), 
                             np.linspace(0.0, 30.0, 100))
      ys = x1s * w_val[0] + x2s * w_val[1] + bias_val
      ax.plot_surface(x1s, x2s, ys, color='g', alpha=0.5)

      # Ispisujemo finalni MSE.
      final_loss = sess.run(loss, feed_dict={X: dX, Y: dY})
      print('Finalni loss: {:.5f}'.format(final_loss))
    
#         # Ispisujemo finalni MSE.
#         final_loss = sess.run(loss, feed_dict={X: dX, Y: dY})
#         print('Finalni loss: {:.5f}'.format(final_loss))
        
#         svi_x = []
#         svi_y = []
#         plt.xlabel('Prave vrednosti')
#         plt.ylabel('Guessed values')
#         for i in range(200):
#           indx = np.random.randint(len(dX))
#           svi_x.append(dY[indx])
#           svi_y.append(math.fabs(dX[indx][0]*w_val[0] + dX[indx][1]*w_val[1] + dX[indx][2]*w_val[2] + bias_val))
        

#         plt.scatter(svi_x, svi_y, c='b')
#         plt.show()