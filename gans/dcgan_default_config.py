
batchsize = 32
epochs_list = [100, 120, 150, 180]  # 和lenth_tag一一对应
dims_list = [32, 64, 128, 256]  # 和tagnum_tag一一对应

optimizer = "Adam"
optimizer_options = [0.0002, 0.5]

loss = 'binary_crossentropy'


generator_activation = 'LeakyReLU'
generator_activation_alpha = 0.2
generator_batchnormalization_momentum = 0.8

discriminator_activation = 'LeakyReLU'
discriminator_activation_alpha = 0.2
