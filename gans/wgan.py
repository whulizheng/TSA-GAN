from __future__ import print_function, division
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop
import random
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def choose_activation(name):
    if name == 'LeakyReLU':
        from tensorflow.keras.layers import LeakyReLU
        return LeakyReLU()
    elif name == "ReLU":
        from tensorflow.keras.layers import ReLU
        return ReLU()
    else:
        print("activation not supported")
        exit(0)


class GAN():
    def __init__(self, raw_dataset_dict,
                 augmentation_percentage_list, if_keep_raw, advanced):
        self.advanced = advanced
        self.edge = advanced["model_collapse_edge"]
        self.if_keep_raw = if_keep_raw
        self.batchsize = advanced["batchsize"]
        self.raw_dataset_dict = raw_dataset_dict
        self.new_dataset_dict = None
        self.augmentation_percentage_list = augmentation_percentage_list
        self.dataset_name = list(raw_dataset_dict.keys())[0]
        self.data_lenth = len(self.raw_dataset_dict[self.dataset_name][0][0])
        self.raw_dataset_x_train_num = len(
            (self.raw_dataset_dict[self.dataset_name][0]))
        self.tags = list(set(self.raw_dataset_dict[self.dataset_name][1]))
        self.epochs = advanced["epochs"]
        self.dim = advanced["dims"]
        self.output_shape = (self.data_lenth,)

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 2
        self.clip_value = 0.02
        self.D_optimizer = RMSprop(lr=0.00005)
        self.G_optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.wasserstein_loss,
                                   optimizer=self.D_optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.dim, ))
        output = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.discriminator(output)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
                              optimizer=self.G_optimizer,
                              metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()
        model.add(
            Dense(self.advanced["G_network_layers"][0],
                  input_shape=(self.dim,)))
        model.add(choose_activation(self.advanced["G_activation"]))
        for dense in self.advanced["G_network_layers"][1:]:
            model.add(Dense(dense))
            model.add(choose_activation(self.advanced["G_activation"]))
            if(self.advanced["generator_batchnormalization"]):
                model.add(BatchNormalization(
                    momentum=self.advanced[
                        "generator_batchnormalization_momentum"]))
        model.add(Dense(self.data_lenth))

        # model.add(Reshape((self.data_lenth, 1, 1)))
        model.summary()
        noise = Input(shape=(self.dim, ))
        output = model(noise)
        return Model(noise, output)

    def build_discriminator(self):
        model = Sequential()
        model.add(
            Dense(self.advanced["D_network_layers"][0],
                  input_shape=(self.data_lenth,)))
        model.add(choose_activation(self.advanced["D_activation"]))
        for dense in self.advanced["D_network_layers"][1:]:
            model.add(Dense(dense))
            model.add(choose_activation(self.advanced["D_activation"]))
            if(self.advanced["discriminator_batchnormalization"]):
                model.add(BatchNormalization(
                    momentum=self.advanced[
                        "discriminator_batchnormalization_momentum"]))
        model.add(Dense(1))
        model.summary()
        output = Input(shape=self.output_shape)
        validity = model(output)

        return Model(output, validity)

    def reInit(self):
        del self.discriminator
        del self.generator
        # Build and compile the critic
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.wasserstein_loss,
                                   optimizer=self.D_optimizer,
                                   metrics=['accuracy'])
        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.dim, ))
        output = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.discriminator(output)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
                              optimizer=self.G_optimizer,
                              metrics=['accuracy'])

    def train(self, epochs):
        training_information = {
            "loss": []
        }
        x_train = []
        y_train = []
        index = 0
        for tag in self.tags:
            c = 0
            if self.augmentation_percentage_list[index] == 0:
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
                if c >= self.edge:
                    print("MC detected")
                    break
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
                noise = np.random.normal(0, 1, (batch_size, self.dim))
                gen_outputs = self.generator.predict(noise)
                d_loss_real = self.discriminator.train_on_batch(
                    training_datas, valid)
                d_loss_fake = self.discriminator.train_on_batch(
                    gen_outputs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                noise = np.random.normal(0, 1, (batch_size, self.dim))

                # Clip discriminator weights
                for layer in self.discriminator.layers:
                    weights = layer.get_weights()
                    weights = [
                        np.clip(w, -self.clip_value, self.clip_value)
                        for w in weights
                    ]
                    layer.set_weights(weights)

                g_loss = self.combined.train_on_batch(noise, valid)
                if(1 - d_loss[0] > 1):
                    c += 1
                else:
                    c = 0
                # Plot the progress
                print("%d [D loss: %f] [G loss: %f]" %
                      (epoch, 1 - d_loss[0], 1 - g_loss[0]))
                gl.append(1 - g_loss[0])
                dl.append(1 - d_loss[0])
                if not epoch % 30:
                    self.update_sample(gen_outputs[0], gl, dl)
                if epoch == epochs - 1:
                    noise = np.random.normal(0, 1, (int(
                        self.augmentation_percentage_list[
                            index] * self.raw_dataset_x_train_num / len(
                                self.tags)), self.dim))
                    outputs = self.generator.predict(noise)
                    for output in outputs:
                        x_train.append(output.tolist())
                        y_train.append(tag)
            training_information["loss"].append([gl, dl])
            index += 1
        return [x_train, y_train], training_information

    def update_sample(self, sample, gl, dl):
        sample_dir = "GANs//temp//"+"sample.png"
        loss_dir = "GANs//temp//"+"loss.png"
        plt.plot(sample)
        plt.savefig(sample_dir)
        plt.clf()
        plt.close()
        plt.plot(gl)
        plt.plot(dl)
        plt.savefig(loss_dir)
        plt.clf()
        plt.close()

    def preprocess(self):
        results, training_information = self.train(self.epochs)
        if results[0]:
            if(self.if_keep_raw):
                self.new_dataset_dict = {
                    self.dataset_name: (
                        np.array(
                            results[0]+list(
                                self.raw_dataset_dict[self.dataset_name][0])),
                        np.array(
                            results[1]+list(
                                self.raw_dataset_dict[self.dataset_name][1]))
                    )
                }
            else:
                self.new_dataset_dict = {
                    self.dataset_name: (
                        np.array(results[0]),
                        np.array(results[1])
                    )
                }
        else:
            self.new_dataset_dict = {
                self.dataset_name: (
                    self.raw_dataset_dict[self.dataset_name][0],
                    self.raw_dataset_dict[self.dataset_name][1]
                )
            }
        return self.new_dataset_dict, training_information


def clean():
    K.clear_session()
    tf.reset_default_graph()
    tf.Graph()


def trainGAN(dataset, augmentation_percentage_list, if_keep_raw, advanced):
    model = GAN(dataset, augmentation_percentage_list, if_keep_raw, advanced)
    dataset, training_information = model.preprocess()
    clean()
    print("finish"+list(dataset.keys())[0])
    return dataset, training_information
