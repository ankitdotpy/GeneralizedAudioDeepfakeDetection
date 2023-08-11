from self_attention import GlobalSelfAttention
from pos_encoding import positional_encoding
import tensorflow as tf

class FeedForward(tf.keras.layers.Layer):
    def __init__(self,d_model,dff,dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff,activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()
        
    def call(self,x):
        x = self.add([x,self.seq(x)])
        x = self.layernorm(x)
        return x
    
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,*,d_model,num_heads,dff,dropout_rate=0.1):
        super().__init__()
        self.self_attention = GlobalSelfAttention(num_heads=num_heads,
                                                 key_dim=d_model,
                                                 dropout=dropout_rate)
        self.ffn = FeedForward(d_model,dff)
        
    def call(self,x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x
    
class Encoder(tf.keras.layers.Layer):
    def __init__(self,*,num_layers,d_model,num_heads,dff,dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_encoding = positional_encoding(length=2018,depth=d_model)
        self.enc_layers = [EncoderLayer(d_model=d_model,
                                       num_heads=num_heads,
                                       dff=dff,
                                       dropout_rate=dropout_rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self,x):
        length = tf.shape(x)[1]
        x = x + self.pos_encoding[tf.newaxis,:length,:]
        
        x = self.dropout(x)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        
        return x