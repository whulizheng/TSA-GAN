from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import numpy as np


def mc_count(file_path, dataset_name, tag):
    with open(file_path, mode="ta", encoding="utf-8") as ta:
        ta.write(dataset_name+','+str(tag)+"\n")


def choose_activation(name):
    if name == 'LeakyReLU':
        from tensorflow.keras.layers import LeakyReLU
        return LeakyReLU()
    else:
        print("activation not supported")
        exit(0)


def choose_optimizer(name, optimizer_options):
    if name == 'Adam':
        return Adam(optimizer_options[0], optimizer_options[1])
    else:
        print("optimizer not supported")
        exit(0)


def mix_weights(w1, w2):
    return w1


class GAN:
    def __init__(self, raw_dataset_dict, augmentation_percentage_list,
                 if_keep_raw, advanced):
        self.advanced = advanced
        self.G_layer_lenth = len(advanced["G_network_layers"])+2
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
        self.input_shape = (self.data_lenth, 1, 1)
        self.output_shape = (self.data_lenth, 1, 1)
        self.optimizer = choose_optimizer(
            advanced["optimizer"], advanced["optimizer_options"])
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss=advanced["loss"],
            optimizer=self.optimizer,
            metrics=['accuracy'])
        self.generator = self.build_generator()
        z = Input(shape=(self.dim,))
        output = self.generator(z)
        frozen_D = Model(
            inputs=self.discriminator.inputs,
            outputs=self.discriminator.outputs)
        frozen_D.trainable = False
        validity = frozen_D(output)
        self.combined = Model(z, validity)
        self.combined.compile(loss=advanced["loss"], optimizer=self.optimizer)

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
            loss=self.advanced["loss"],
            optimizer=self.optimizer,
            metrics=['accuracy'])
        self.generator = self.build_generator()
        z = Input(shape=(self.dim,))
        output = self.generator(z)
        frozen_D = Model(
            inputs=self.discriminator.inputs,
            outputs=self.discriminator.outputs)
        frozen_D.trainable = False
        validity = frozen_D(output)
        self.combined = Model(z, validity)
        self.combined.compile(
            loss=self.advanced["loss"], optimizer=self.optimizer)

    def train(self, epochs):
        training_information = {
            "loss": []
        }
        x_train = []
        y_train = []
        index = 0
        for tag in self.tags:
            models_tmp = {}
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
            epoch = 0
            while 1:
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
                training_datas.resize(list(training_datas.shape)+[1])
                noise = np.random.normal(0, 1, (batch_size, self.dim))
                if c >= self.edge:
                    print("mode collapse detected!!!!!!!")
                    print("############")
                    mc_count(r'GANs/temp/mclog.txt', self.dataset_name, tag)
                    # Model Collapse Detected #
                    for o in range(int(self.advanced["model_collect_num"])):
                        model_tmp = []
                        for layer in self.generator.layers:
                            layer_tmp = layer.get_weights()
                            model_tmp.append(layer_tmp)
                            '''
                            weights = [
                                np.clip(w, 0.05, 0.05)
                                for w in weights
                            ]
                            layer.set_weights(weights)
                            '''
                        for u in range(
                                int(self.advanced["sampling_frequency"])):
                            g_loss = self.combined.train_on_batch(noise, valid)
                        models_tmp[str(o)] = model_tmp
                    # mixing
                    # for ii in range(len(models_tmp)):
                    #     model_new = mix_weights(models_tmp[str(ii)], model_new)
                    # model_new = np.add(models_tmp[str(ii)], model_new)
                    # model_new = np.true_divide(model_new, np.full(np.array(model_new).shape, len(models_tmp)+1)
                    # recover
                    tmp_layer_1 = []
                    for p in range(len(models_tmp["0"][1])):
                        o = random.randint(
                            0, int(self.advanced["model_collect_num"])-1)
                        tmp_layer_1.append(models_tmp[str(o)][1][p])
                    self.generator.layers[1].set_weights(tmp_layer_1)
                    epoch = epoch - self.edge
                    print("finish recovering")
                gen_outputs = self.generator.predict(noise)
                d_loss_real = self.discriminator.train_on_batch(
                    training_datas, valid)
                d_loss_fake = self.discriminator.train_on_batch(
                    gen_outputs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                noise = np.random.normal(0, 1, (batch_size, self.dim))
                g_loss = self.combined.train_on_batch(noise, valid)
                if(d_loss[0] > 1):
                    c += 1
                else:
                    c = 0
                print(self.dataset_name+'_'+str(tag) +
                      ":%d[D loss:real:%f, fake:%f,acc.:%.2f%%][G loss:%f]" % (
                    epoch + 1,
                    d_loss_real[0],
                    d_loss_fake[0],
                    100 * d_loss[1],
                    g_loss))
                gl.append(g_loss)
                dl.append(d_loss[0])
                if not epoch % 10:
                    self.update_sample(gen_outputs[0], gl, dl)
                epoch = epoch + 1
                if epoch == epochs - 1:
                    noise = np.random.normal(0, 1, (int(
                        self.augmentation_percentage_list[
                            index] * self.raw_dataset_x_train_num / len(
                                self.tags)), self.dim))
                    outputs = self.generator.predict(noise)
                    for output in outputs:
                        tmp = []
                        for i in output:
                            tmp.append(i[0][0])
                        x_train.append(tmp)
                        y_train.append(tag)
                    break
            training_information["loss"].append([gl, dl])
            index += 1
        return [x_train, y_train], training_information

    def update_sample(self, sample, gl, dl):
        tmp = []
        for i in sample:
            tmp.append(i[0][0])
        sample_dir = "GANs//temp//"+"sample.png"
        loss_dir = "GANs//temp//"+"loss.png"
        plt.plot(tmp)
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
    tf.Graph()


def trainGAN(dataset, augmentation_percentage_list, if_keep_raw, advanced):
    model = GAN(dataset, augmentation_percentage_list, if_keep_raw, advanced)
    dataset, training_information = model.preprocess()
    clean()
    print("finish"+list(dataset.keys())[0])
    return dataset, training_information
