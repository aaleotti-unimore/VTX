#   optimizers
discr_opt = RMSprop(
    lr=0.01,
    clipvalue=1.0,
    decay=1e-8)

gan_opt = adam(
    lr=0.0001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8,
    decay=1e-8,
    clipvalue=1.0)  

#   compilation
gan.compile(loss='binary_crossentropy', optimizer=discr_opt)
disc.trainable = True
disc.compile(loss='binary_crossentropy', optimizer=gan_opt)

for epoch in range(200):
     for index in range(int(X_train.shape[0] / BATCH_SIZE)):

        noise = np.random.normal(size=(BATCH_SIZE, latent_dim))  # random latent vector
        alexa_domains = X_train[(index * BATCH_SIZE):(index + 1) * BATCH_SIZE]

        # Generating domains from generator
        generated_domains = genr.predict(noise, verbose=0)   

        labels_size = (BATCH_SIZE, 1)
        labels_real = np.random.uniform(0.9, 1.1, size=labels_size)  # ~1 = real. Label Smoothing technique
        labels_fake = np.zeros(shape=labels_size)  # 0 = fake
        # alternate training mode:
        if index % 2 == 0:
            training_domains = alexa_domains
            labels = labels_real
        else:
            training_domains = generated_domains
            labels = labels_fake

       
        disc.trainable = True
        disc_history = disc.train_on_batch(training_domains, labels)
        disc.trainable = False

        # training generator model inside the adversarial model
        noise = np.random.normal(size=(BATCH_SIZE, latent_dim))  # random latent vectors.
        misleading_targets = np.random.uniform(0.9, 1.1, size=labels_size)
        gan_history = gan.train_on_batch(noise, misleading_targets)