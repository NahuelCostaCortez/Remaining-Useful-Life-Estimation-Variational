import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Masking

# keeping the random seed constant from one experiment to the next makes it 
# easier to interpret the effects of hyper-parameters values 
seed = 99
random.seed(seed)
tf.random.set_seed(seed)

def create_model(timesteps, input_dim, intermediate_dim, batch_size, latent_dim, epochs, optimizer):
	# Setup the network parameters:
	timesteps = timesteps
	input_dim = input_dim
	intermediate_dim = intermediate_dim
	batch_size = batch_size
	latent_dim = latent_dim
	epochs = epochs
	if optimizer == 'adam':
		optimizer = keras.optimizers.Adam(learning_rate=0.0001)
	else:
		print("unimplemented optimizer")
		exit(-1)
	masking_value = -99.

	class Sampling(keras.layers.Layer):
		"""Uses (z_mean, sigma) to sample z, the vector encoding an engine trajetory."""
		def call(self, inputs):
			mu, sigma = inputs
			batch = tf.shape(mu)[0]
			dim = tf.shape(mu)[1]
			epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
			return mu + tf.exp(0.5 * sigma) * epsilon
	
	# ----------------------- Encoder -----------------------
	inputs = Input(shape=(timesteps, input_dim,), name='encoder_input')

	mask = Masking(mask_value=masking_value)(inputs)
	
	# LSTM encoding
	h = Bidirectional(LSTM(intermediate_dim))(mask) 

	# VAE Z layer
	mu = Dense(latent_dim)(h)
	sigma = Dense(latent_dim)(h)

	z = Sampling()([mu, sigma])

	# Instantiate the encoder model:
	encoder = keras.Model(inputs, [z, mu, sigma], name='encoder')
	print(encoder.summary())
	# -------------------------------------------------------
	
	# ----------------------- Regressor --------------------
	reg_latent_inputs = Input(shape=(latent_dim,), name='z_sampling_reg')
	reg_intermediate = Dense(200, activation='tanh')(reg_latent_inputs)
	reg_outputs = Dense(1, name='reg_output')(reg_intermediate)
	# Instantiate the classifier model:
	regressor = keras.Model(reg_latent_inputs, reg_outputs, name='regressor')
	print(regressor.summary())
	# -------------------------------------------------------

	''' uncomment if needed
	# ----------------------- Decoder --------------------
	latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
	h_decoded = RepeatVector(timesteps)(latent_inputs)
	h_decoded = Bidirectional(LSTM(intermediate_dim, return_sequences=True))(h_decoded) 
	# decoded layer
	outputs = LSTM(input_dim, return_sequences=True)(h_decoded) 

	# Instantiate the decoder model:
	decoder = keras.Model(latent_inputs, outputs, name='decoder')
	print(decoder.summary())
	# -------------------------------------------------------
	'''
	
	# -------------------- Wrapper model --------------------
	class RVE(keras.Model):
		def __init__(self, encoder, regressor, decoder=None, **kwargs):
			super(RVE, self).__init__(**kwargs)
			self.encoder = encoder
			self.regressor = regressor
			self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
			self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
			self.reg_loss_tracker = keras.metrics.Mean(name="reg_loss")
			self.decoder = decoder
			if self.decoder!=None:
				self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
			

		@property
		def metrics(self):
			if self.decoder!=None:
				return [
					self.total_loss_tracker,
					self.kl_loss_tracker,
					self.reg_loss_tracker,
					self.reconstruction_loss_tracker
				]
			else:
				return [
					self.total_loss_tracker,
					self.kl_loss_tracker,
					self.reg_loss_tracker,
				]

		def train_step(self, data):
			x, target_x = data
			with tf.GradientTape() as tape:
				# kl loss
				z, mu, sigma = self.encoder(x)
				kl_loss = -0.5 * (1 + sigma - tf.square(mu) - tf.exp(sigma))
				kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
				# Regressor
				reg_prediction = self.regressor(z)
				reg_loss = tf.reduce_mean(
					keras.losses.mse(target_x, reg_prediction)
				)
				# Reconstruction
				if self.decoder!=None:
					reconstruction = self.decoder(z)
					reconstruction_loss = tf.reduce_mean(
						keras.losses.mse(x, reconstruction)
					)
					total_loss = kl_loss + reg_loss + reconstruction_loss
					self.reconstruction_loss_tracker.update_state(reconstruction_loss)
				else:
					total_loss = kl_loss + reg_loss
			grads = tape.gradient(total_loss, self.trainable_weights)
			self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
			self.total_loss_tracker.update_state(total_loss)
			self.kl_loss_tracker.update_state(kl_loss)
			self.reg_loss_tracker.update_state(reg_loss)
			return {
				"loss": self.total_loss_tracker.result(),
				"kl_loss": self.kl_loss_tracker.result(),
				"reg_loss": self.reg_loss_tracker.result(),
			}


		def test_step(self, data):
			x, target_x = data

			# kl loss
			z, mu, sigma = self.encoder(x)
			kl_loss = -0.5 * (1 + sigma - tf.square(mu) - tf.exp(sigma))
			kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
			# Regressor
			reg_prediction = self.regressor(z)
			reg_loss = tf.reduce_mean(
				keras.losses.mse(target_x, reg_prediction)
			)
			# Reconstruction
			if self.decoder!=None:
				reconstruction = self.decoder(z)
				reconstruction_loss = tf.reduce_mean(
					keras.losses.mse(x, reconstruction)
				)
				total_loss = kl_loss + reg_loss + reconstruction_loss
			else:
				total_loss = kl_loss + reg_loss

			return {
				"loss": total_loss,
				"kl_loss": kl_loss,
				"reg_loss": reg_loss,
			}
	# -------------------------------------------------------

	rve = RVE(encoder, regressor)
	rve.compile(optimizer=optimizer)

	return rve
