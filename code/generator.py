dropout_value = 0.4
cnn_filters = [20, 10]
cnn_kernels = [2, 3]
cnn_strides = [1, 1]
dec_convs = []
leaky_relu_alpha = 0.2
latent_vector = 20
timesteps = 15
word_index = 38

dec_inputs = Input(shape=(latent_vector,),
                   name="Generator_Input")
decoded = RepeatVector(timesteps, name="gen_repeate_vec")(dec_inputs)
decoded = LSTM(word_index, return_sequences=True, name="gen_LSTM")(decoded)
decoded = Dropout(dropout_value)(decoded
)
for i in range(2):
    conv = Conv1D(cnn_filters[i],
                  cnn_kernels[i],
                  padding='same',
                  strides=cnn_strides[i],
                  name='gen_conv%s' % i)(decoded)
    conv = LeakyReLU(alpha=leaky_relu_alpha)(conv)
    conv = Dropout(dropout_value, name="gen_dropout%s" % i)(conv)
    dec_convs.append(conv)

decoded = concatenate(dec_convs)
decoded = TimeDistributed(Dense(word_index, activation='softmax'), name='decoder_end')(decoded)  

G = Model(inputs=dec_inputs, outputs=decoded, name='Generator')