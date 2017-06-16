import input_data
bonedata=input_data.read_data_sets("",one_hot=True)

#start tensorflow interactiveSession
import tensorflow as tf
sess=tf.InteractiveSession()

#weight initialization
def weight_variable(shape):
 initial=tf.truncated_normal(shape,stddev=0.1)
 return tf.Variable(initial)

def bias_variable(shape):
 initial=tf.constant(0.1,shape=shape)
 return tf.Variable(initial)
 
 #convolution
 def conv2d(x,w):
   return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
 #pooling
 def max_pool_2*2(x):
   return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
   
 #creat model
 x=tf.placeholder("float",[None,1577])
 y_=tf.placeholder("float",[None,1])
 
 w=tf.Varible(tf.zeros([1577,1]))
 b=tf.Varible(tf.zeros([1]))
 
 y=tf.nn.softmax(tf.matmul(x,w)+b)
 
 #convolutional layer
 
 w_conv1=weight_variable([5,5,1,32])
 b_conv1=bias_variable([32])
 
 x_image=tf.reshape(x,[-1,28,28,1])
 
 h_conv1=tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
 h_pool1=max_pool_1*4(h_conv1)
 
 w_conv2=weight_variable([5,5,32,64])
 b_conv2=bias_variable([64])
 
 h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
 h_pool2=max_pool_1*4(h_conv2)
 
 #define the connected layer
 
