from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, MaxPooling2D, multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D, UpSampling1D, ZeroPadding1D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, RMSprop
import keras.backend as K
from functools import partial

import matplotlib.pyplot as plt

import sys
import glob

import numpy as np

from AudioFileManager import *

import tensorflow as tf
from keras.layers import Conv2DTranspose, Lambda

class RandomWeightedAverage(_Merge):
	def __init__(self, batchSize):
		self.batchSize = batchSize
		super(RandomWeightedAverage, self).__init__()

	def _merge_function(self, inputs):
		alpha = 1*K.random_uniform((self.batchSize, 1, 1))
		return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class GAN():

	def __init__(self, length, batchSize, nClasses, epochToLoad = -1):

		self.batchSize = batchSize
		self.nClasses = nClasses

		self.outputShape = (length, 1)
		self.nRandom = 100

		optimizerD = Adam(lr=1e-4, beta_1=0.5, beta_2=0.9)
		optimizerG = Adam(lr=1e-4, beta_1=0.5, beta_2=0.9)

		self.generator = self.BuildGenerator()
		self.critic, self.auxiliary = self.BuildcriticAndRecogniser()

		self.generator.trainable = False

		real = Input(shape=self.outputShape)

		z_disc = Input(shape=(self.nRandom,))
		salient_disc = Input(shape=(self.nClasses,))
		fake = self.generator([z_disc, salient_disc])

		fake = self.critic(fake)
		valid = self.critic(real)

		interpolated = RandomWeightedAverage(batchSize)([real, fake])
		validity_interpolated = self.critic(interpolated)

		partialGpLoss = partial(self.GradientPenaltyLoss,
						  averagedSamples=interpolated)
		partialGpLoss.__name__ = 'gradient_penalty' # Keras requires function names


		self.criticModel = Model(inputs=[real, z_disc, salient_disc], outputs=[valid, fake, validity_interpolated])
		self.criticModel.compile(loss=[self.WassersteinLoss, self.WassersteinLoss, partialGpLoss], optimizer=optimizerD, loss_weights=[1, 1, 10])


		self.critic.trainable = False
		self.generator.trainable = True

		z_gen = Input(shape=(self.nRandom,))
		salient_gen = Input(shape=(self.nClasses,))
		generated = self.generator([z_gen, salient_gen])
		valid = self.critic(generated)
		category = self.auxiliary(generated)

		self.generatorModel = Model([z_gen, salient_gen], [valid, category])
		self.generatorModel.compile(loss=[self.WassersteinLoss, self.MutualInformationLoss], optimizer=optimizerG, loss_weights=[1,1])

		self.startEpoch = 0
		if epochToLoad > 0:
			self.generator.load_weights("SavedModel/generator{}.h5".format(startEpoch))
			self.critic.load_weights("SavedModel/critic{}.h5".format(startEpoch))
			self.auxiliary.load_weights("SavedModel/auxiliary{}.h5".format(startEpoch))
			self.startEpoch = epochToLoad

	def BuildGenerator(self):
		
		kernelSize = 25
		alpha = 0.2

		random = Input(shape=(self.nRandom,))
		salient = Input(shape=(self.nClasses,))

		x = Dense(256)(salient)
		x = Activation('relu')(x)
		x = Dense(256)(x)
		x = Activation('relu')(x)
		x = Dense(256)(x)
		x = Activation('relu')(x)
		x = Dense(256)(x)
		x = Activation('relu')(x)
		x = Dense(256)(x)
		x = Activation('relu')(x)
		x = Dense(self.nRandom)(x)

		latent = multiply([random, x])

		x = Dense(4*4*64*32)(random)
		x = Activation('relu')(x)
		x = Reshape((4*4, 64*32))(x)

		x = Lambda(lambda x: K.expand_dims(x, axis=2))(x)

		x = Conv2DTranspose(filters=512, kernel_size=(kernelSize, 1), strides=(4, 1), padding='same')(x)
		x = Activation('relu')(x)

		x = Conv2DTranspose(filters=256, kernel_size=(kernelSize, 1), strides=(4, 1), padding='same')(x)
		x = Activation('relu')(x)

		x = Conv2DTranspose(filters=128, kernel_size=(kernelSize, 1), strides=(4, 1), padding='same')(x)
		x = Activation('relu')(x)

		x = Conv2DTranspose(filters=64, kernel_size=(kernelSize, 1), strides=(4, 1), padding='same')(x)
		x = Activation('relu')(x)

		x = Conv2DTranspose(filters=1, kernel_size=(kernelSize, 1), strides=(4, 1), padding='same')(x)

		x = Lambda(lambda x: K.squeeze(x, axis=2))(x)

		model = Model([random, salient], x)

		model.summary()

		return model

	def BuildcriticAndRecogniser(self):
	
		kernel_size = 25
		alpha = 0.2

		inputsD = Input(shape=self.outputShape)

		x = Conv1D(64, kernel_size=kernel_size, strides=4, input_shape=self.outputShape, padding="same")(inputsD)
		x = LeakyReLU(alpha=alpha)(x)

		x = Conv1D(128, kernel_size=kernel_size, strides=4, input_shape=self.outputShape, padding="same")(x)
		x = LeakyReLU(alpha=alpha)(x)

		x = Conv1D(256, kernel_size=kernel_size, strides=4, input_shape=self.outputShape, padding="same")(x)
		x = LeakyReLU(alpha=alpha)(x)

		x = Conv1D(512, kernel_size=kernel_size, strides=4, input_shape=self.outputShape, padding="same")(x)
		x = LeakyReLU(alpha=alpha)(x)

		x = Conv1D(1024, kernel_size=kernel_size, strides=4, input_shape=self.outputShape, padding="same")(x)
		x = LeakyReLU(alpha=alpha)(x)

		embedding = Flatten()(x)
		validity = Dense(1, activation='linear')(embedding)


		inputsC = Input(shape=self.outputShape)

		x = Conv1D(64, kernel_size=kernel_size, strides=4, input_shape=self.outputShape, padding="same")(inputsC)
		x = LeakyReLU(alpha=alpha)(x)

		x = Conv1D(128, kernel_size=kernel_size, strides=4, input_shape=self.outputShape, padding="same")(x)
		x = LeakyReLU(alpha=alpha)(x)

		x = Conv1D(256, kernel_size=kernel_size, strides=4, input_shape=self.outputShape, padding="same")(x)
		x = LeakyReLU(alpha=alpha)(x)

		x = Conv1D(512, kernel_size=kernel_size, strides=4, input_shape=self.outputShape, padding="same")(x)
		x = LeakyReLU(alpha=alpha)(x)

		x = Conv1D(1024, kernel_size=kernel_size, strides=4, input_shape=self.outputShape, padding="same")(x)
		x = LeakyReLU(alpha=alpha)(x)

		embedding = Flatten()(x)
		label = Dense(self.nClasses, activation='softmax')(embedding)

		return Model(inputsD, validity), Model(inputsC, label)

	def MutualInformationLoss(self, c, c_given_x):
		eps = 1e-8
		conditionalEntropy = K.mean(-K.sum(K.log(c_given_x + eps) * c, axis=1))
		entropy = K.mean(-K.sum(K.log(c + eps) * c, axis=1))
		return conditionalEntropy + entropy

	def WassersteinLoss(self, y_true, y_pred):
		return K.mean(y_true * y_pred)

	def GradientPenaltyLoss(self, y_true, y_pred, averagedSamples):
		gradients = K.gradients(y_pred, averagedSamples)[0]
		gradientsSqr = K.square(gradients)
		gradientsSqrSum = K.sum(gradientsSqr, axis=np.arange(1, len(gradientsSqr.shape)))
		gradientL2Norm = K.sqrt(gradientsSqrSum)
		gradientPenalty = K.square(1 - gradientL2Norm)
		return K.mean(gradientPenalty)

	def Train(self, epochs, wav, sampleInterval):

		valid = -np.ones((self.batchSize, 1))
		fake =  np.ones((self.batchSize, 1))
		dummy =  np.zeros((self.batchSize, 1))

		for epoch in range(self.startEpoch, epochs):

			selection = np.zeros((self.batchSize, self.outputShape[0], 1))
			gridSize = 16384
			for i in range(self.batchSize):
				r = np.random.randint((int(len(wav)-self.outputShape[0])/gridSize))
				selection[i,:,0] = wav[r*gridSize:r*gridSize+self.outputShape[0]]

			noise = np.random.normal(0, 1, (self.batchSize, self.nRandom))
			labelsAsInts = np.random.randint(0, self.nClasses, self.batchSize)
			salient = np.zeros((self.batchSize, self.nClasses))
			for i in range(self.batchSize):
				salient[i, labelsAsInts[i]] = 1

			for i in range(5):

				d_loss = self.criticModel.train_on_batch([selection, noise, salient], [valid, fake, dummy])

			g_loss = self.generatorModel.train_on_batch([noise, salient], [valid, salient])

			print ("%d [D loss: %.2f] [Q loss: %.2f] [G loss: %.2f]" % (epoch, d_loss[0], g_loss[2], g_loss[1]))

			if epoch % sampleInterval == 0:
				self.Sample(epoch)

			if epoch % 500 == 0:
				self.generator.save_weights("SavedModel/generator{}.h5".format(epoch))
				self.critic.save_weights("SavedModel/critic{}.h5".format(epoch))
				self.auxiliary.save_weights("SavedModel/auxiliary{}.h5".format(epoch))

	def Sample(self, epoch):
		noise = np.random.normal(0, 1, (self.nClasses, self.nRandom))
		salient = np.zeros((self.nClasses, self.nClasses))
		for i in range(self.nClasses):
			salient[i,i] = 1
		outputs = self.generator.predict([noise, salient])
		
		for j in range(self.nClasses):
			wav = outputs[j,:,0]

			WriteMonoWav("Outputs/raw/{}_{}.wav".format(epoch, j), wav, 16000)
			WriteMonoWav("Outputs/normalized/{}_{}.wav".format(epoch, j), wav/np.max(np.abs(wav)), 16000)
		
if __name__ == '__main__':
	import tensorflow as tf
	from keras.backend.tensorflow_backend import set_session
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.3
	set_session(tf.Session(config=config))

	sequenceLength = 16384

	allWavs = np.zeros(0)

	for filename in glob.glob("drums/train/*.wav"):
		print("Loading {}".format(filename))
		rate, wav = ReadWavAsMono(filename)
		allWavs = np.concatenate([allWavs, wav])

	WriteMonoWav("TrainingWaveform.wav", allWavs, 16000)

	gan = GAN(sequenceLength, batchSize=32, nClasses=32, epochToLoad=-1)
	gan.Train(epochs=300000, wav=allWavs, sampleInterval=10)
