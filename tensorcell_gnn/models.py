import tensorflow as tf
import spektral as spktrl

keras = tf.keras



class GAT(keras.models.Model):
    def __init__(self, n_gat_channels, dropout_rate, gat_dropout_rate, n_attn_heads, regularizer_rate, n_layers=1):
        
        super().__init__()

        self.hidden = []
    
        for i in range(n_layers):
            self.hidden.append(
                (
                    keras.layers.Dropout(dropout_rate),

                    spktrl.layers.GATConv(
                        channels=n_gat_channels,
                        attn_heads=n_attn_heads,
                        activation='selu',
                        dropout_rate=gat_dropout_rate,
                        kernel_initializer='lecun_normal',
                        kernel_regularizer=keras.regularizers.l2(regularizer_rate)   
                    )
                ))
        
        self.dropout_last = keras.layers.Dropout(dropout_rate)

        self.conv_last = spktrl.layers.GATConv(
            channels=64,
            attn_heads=n_attn_heads,
            concat_heads=True,
            activation='selu',
            dropout_rate=gat_dropout_rate,
            kernel_initializer='lecun_normal',
            kernel_regularizer=keras.regularizers.l2(regularizer_rate)
        )  
        
        self.global_pool = spktrl.layers.GlobalSumPool()   

        self.dense_1 = keras.layers.Dense(
            64, 
            activation='selu',
            kernel_initializer='lecun_normal'
        )
        
        self.dense_2 = keras.layers.Dense(
            32, 
            activation='selu',
            kernel_initializer='lecun_normal'
        )
        
        self.out = keras.layers.Dense(1)


    def call(self, inputs):
        
        x, a = inputs
                
        for layer in self.hidden:
            x = layer[0](x)
            x = layer[1]([x, a])
        
        x = self.dropout_last(x)
        x = self.conv_last([x, a])
        
        x = self.global_pool(x)
        x = self.dense_1(x)
        x = self.dense_2(x)

        output = self.out(x)

        return output