# Preprocessor_GAN
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow as tf
import random
import numpy as np
import gans.dcgan_config as dcgan_config


def choose_activation(name, alpha):
    if name == 'LeakyReLU':
        from tensorflow.keras.layers import LeakyReLU
        return LeakyReLU(alpha=alpha)
    else:
        print("activation not supported")
        exit(0)


def choose_optimizer(name):
    if name == 'Adam':
        return Adam(dcgan_config.optimizer_options[0], dcgan_config.optimizer_options[1])
    else:
        print("optimizer not supported")
        exit(0)


class GAN:
    def __init__(self, raw_dataset_dict, augmentation_percentage_list, taglist, if_keep_raw):
        self.if_keep_raw = if_keep_raw
        self.taglist = taglist
        self.batchsize = dcgan_config.batchsize
        self.raw_dataset_dict = raw_dataset_dict
        self.new_dataset_dict = None
        self.augmentation_percentage_list = augmentation_percentage_list
        self.dataset_name = list(raw_dataset_dict.keys())[0]
        self.data_lenth = len(self.raw_dataset_dict[self.dataset_name][0][0])
        self.raw_dataset_x_train_num = len(
            (self.raw_dataset_dict[self.dataset_name][0]))
        self.tags = list(set(self.raw_dataset_dict[self.dataset_name][1]))
        self.epochs = dcgan_config.epochs_list[self.taglist[0]]
        self.dim = dcgan_config.dims_list[self.taglist[1]]
        self.data_lenth = len(self.raw_dataset_dict[self.dataset_name][0][0])
        self.input_shape = (self.data_lenth, 1, 1)
        self.output_shape = (self.data_lenth, 1, 1)
        self.optimizer = choose_optimizer(dcgan_config.optimizer)
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss=dcgan_config.loss, optimizer=self.optimizer, metrics=['accuracy'])
        self.generator = self.build_generator()
        z = Input(shape=(self.dim,))
        output = self.generator(z)
        frozen_D = Model(
            inputs=self.discriminator.inputs,
            outputs=self.discriminator.outputs)
        frozen_D.trainable = False
        validity = frozen_D(output)
        self.combined = Model(z, validity)
        self.combined.compile(loss=dcgan_config.loss, optimizer=self.optimizer)

    def build_generator(self):
        model = Sequential()
        model.add(Dense(4 * 4 * 4, activation="relu", input_dim=self.dim))
        model.add(Reshape((4, 4, 4)))
        model.add(UpSampling2D())
        model.add(Conv2D(4 * 4 * 4, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(self.data_lenth, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("tanh"))
        model.add(Reshape((self.data_lenth, 1, 16*16)))
        model.add(Conv2D(1, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("tanh"))
        # model.add(Reshape((self.data_lenth, 1, 1)))
        model.summary()

        noise = Input(shape=(self.dim,))
        output = model(noise)

        return Model(noise, output)

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(64, kernel_size=3, strides=2,
                         input_shape=self.output_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        output = Input(shape=self.output_shape)
        validity = model(output)

        return Model(output, validity)

    def reInit(self):
        del self.discriminator
        del self.generator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss=dcgan_config.loss, optimizer=self.optimizer, metrics=['accuracy'])
        self.generator = self.build_generator()
        z = Input(shape=(self.dim,))
        output = self.generator(z)
        frozen_D = Model(
            inputs=self.discriminator.inputs,
            outputs=self.discriminator.outputs)
        frozen_D.trainable = False
        validity = frozen_D(output)
        self.combined = Model(z, validity)
        self.combined.compile(loss=dcgan_config.loss, optimizer=self.optimizer)

    def train(self, epochs):
        x_train = []
        y_train = []
        index = 0
        for tag in self.tags:
            if self.augmentation_percentage_list[index] <= 0.1:
                index += 1
                continue
            gl = []
            dl = []
            self.reInit()
            all_training_datas = []
            for i in range(len(self.raw_dataset_dict[self.dataset_name][0])):
                if self.raw_dataset_dict[self.dataset_name][1][i] == tag:
                    all_training_datas.append(
                        self.raw_dataset_dict[self.dataset_name][0][i])
            data_size = len(all_training_datas)
            batch_size = self.batchsize
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))
            for epoch in range(epochs):
                training_datas = []
                if data_size >= self.batchsize:
                    randlist = random.sample(
                        range(0, data_size), self.batchsize)
                    for i in randlist:
                        training_datas.append(all_training_datas[i])
                else:
                    for i in range(self.batchsize):
                        training_datas.append(
                            all_training_datas[i % data_size])

                training_datas = np.array(training_datas)
                training_datas = training_datas.reshape(
                    (batch_size, self.data_lenth, 1, 1))
                noise = np.random.normal(0, 1, (batch_size, self.dim))
                gen_outputs = self.generator.predict(noise)
                d_loss_real = self.discriminator.train_on_batch(
                    training_datas, valid)
                d_loss_fake = self.discriminator.train_on_batch(
                    gen_outputs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                noise = np.random.normal(0, 1, (batch_size, self.dim))
                g_loss = self.combined.train_on_batch(noise, valid)
                print(self.dataset_name+'_'+str(tag)+":%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (
                    epoch, d_loss[0], 100 * d_loss[1], g_loss))
                gl.append(g_loss)
                dl.append(d_loss[0])
                if epoch == epochs - 1:
                    # 对应每个tag 乘以 augmentation_percentage
                    noise = np.random.normal(0, 1, (int(
                        self.augmentation_percentage_list[index] * self.raw_dataset_x_train_num / len(self.tags)), self.dim))
                    outputs = self.generator.predict(noise)
                    for x in range(int(self.augmentation_percentage_list[index] * self.raw_dataset_x_train_num / len(self.tags))):
                        output = []
                        for i in outputs[x]:
                            output.append(i[0][0])
                        x_train.append(output)
                        y_train.append(tag)
            index += 1
        return [x_train, y_train]

    def preprocess(self):
        results = self.train(self.epochs)
        if results[0]:
            if(self.if_keep_raw):
                self.new_dataset_dict = {
                    self.dataset_name: (
                        np.array(
                            results[0]+list(self.raw_dataset_dict[self.dataset_name][0])),
                        np.array(
                            results[1]+list(self.raw_dataset_dict[self.dataset_name][1])),
                        self.raw_dataset_dict[self.dataset_name][2],
                        self.raw_dataset_dict[self.dataset_name][3]
                    )
                }
            else:
                self.new_dataset_dict = {
                    self.dataset_name: (
                        np.array(results[0]),
                        np.array(results[1]),
                        self.raw_dataset_dict[self.dataset_name][2],
                        self.raw_dataset_dict[self.dataset_name][3]
                    )
                }
        else:
            self.new_dataset_dict = {
                self.dataset_name: (
                    self.raw_dataset_dict[self.dataset_name][0],
                    self.raw_dataset_dict[self.dataset_name][1],
                    self.raw_dataset_dict[self.dataset_name][2],
                    self.raw_dataset_dict[self.dataset_name][3]
                )
            }
        return self.new_dataset_dict


def clean():
    K.clear_session()
    tf.reset_default_graph()
    tf.Graph()


def trainGAN(dataset, augmentation_percentage_list, taglist, if_keep_raw):
    model = GAN(dataset, augmentation_percentage_list, taglist, if_keep_raw)
    dataset = model.preprocess()
    clean()
    print("finish"+list(dataset.keys())[0])
    return dataset
