import tensorflow as tf

class GlobalSelfAttention(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()
        
    def call(self,x):
        attn_output = self.mha(query=x,
                              key=x,
                              value=x)
        x = self.add([x,attn_output])
        x = self.layernorm(x)
        return x