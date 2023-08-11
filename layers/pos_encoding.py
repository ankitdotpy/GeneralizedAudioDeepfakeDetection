import numpy as np
import tensorflow as tf

def positional_encoding(length,depth):
    depth = depth/2
    position = np.arange(length)[:,np.newaxis] #(length,1)
    depths = np.arange(depth)[np.newaxis,:]/depth #(1,depth)
    
    angle_rates = 1/(10000**depths) # (1/10000^(2i/d_model))
    angle_rads = position * angle_rates
    
    pos_encoding = np.concatenate([np.sin(angle_rads),np.cos(angle_rads)],axis=-1)
    
    return tf.cast(pos_encoding,dtype=tf.float32)