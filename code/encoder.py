dropout_value = 0.5
cnn_filters = [20, 10]
cnn_kernels = [2, 3]
cnn_strides = [1, 1]
leaky_relu_alpha = 0.2
timesteps = 15
word_index = 38
latent_vector = 20

discr_inputs = Input(shape=(timesteps, word_index),
					 name="Discriminator_Input")
for i in range(2):
	conv = Conv1D(cnn_filters[i],
				  cnn_kernels[i],
				  padding='same',
				  strides=cnn_strides[i],
				  name='discr_conv%s' % i)(discr_inputs)
	conv = BatchNormalization()(conv)
	conv = LeakyReLU(alpha=leaky_relu_alpha)(conv)
	conv = Dropout(dropout_value, name='discr_dropout%s' % i)(conv)
	conv = AveragePooling1D()(conv)
	enc_convs.append(conv)

discr = concatenate(enc_convs)
discr = LSTM(latent_vector)(discr)

E = Model(inputs=discr_inputs, outputs=discr, name='Encoder')
