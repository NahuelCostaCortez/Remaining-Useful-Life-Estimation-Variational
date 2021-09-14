import model
import utils
import random
import numpy as np


if __name__ == "__main__":
	# ------------------------------ DATA -----------------------------------
	dataset = input("Enter dataset (FD001, FD002, FD003, FD004): ")
	# sensors to work with: T30, T50, P30, PS30, phi
	sensors = ['s_3', 's_4', 's_7', 's_11', 's_12']
	# windows length
	sequence_length = 30
	# smoothing intensity
	alpha = 0.1
	# max RUL
	threshold = 125
	
	x_train, y_train, x_val, y_val, x_test, y_test = utils.get_data(dataset, sensors, 
	sequence_length, alpha, threshold)
	# -----------------------------------------------------------------------
	
	# ----------------------------- MODEL -----------------------------------
	timesteps = x_train.shape[1]
	input_dim = x_train.shape[2]
	intermediate_dim = 300
	batch_size = 128
	latent_dim = 2
	epochs = 10000
	optimizer = 'adam'
	
	RVAE = model.create_model(timesteps, 
			input_dim, 
			intermediate_dim, 
			batch_size, 
			latent_dim, 
			epochs, 
			optimizer,
			)
	
	# Callbacks for training
	model_callbacks = utils.get_callbacks(RVAE, x_train, y_train)
	# -----------------------------------------------------------------------

	# --------------------------- TRAINING ---------------------------------
	results = RVAE.fit(x_train, y_train,
			shuffle=True,
			epochs=epochs,
			batch_size=batch_size,
			validation_data= (x_val, y_val),
			callbacks=model_callbacks, verbose=2)
	# -----------------------------------------------------------------------

	# -------------------------- EVALUATION ---------------------------------
	RVAE.load_weights('./checkpoints/checkpoint')
	train_mu = utils.viz_latent_space(RVAE.encoder, np.concatenate((x_train, x_val)), np.concatenate((y_train, y_val)))
	test_mu = utils.viz_latent_space(RVAE.encoder, x_test, y_test.clip(upper=threshold))
	# Evaluate
	y_hat_train = RVAE.regressor.predict(train_mu)
	y_hat_test = RVAE.regressor.predict(test_mu)

	utils.evaluate(np.concatenate((y_train, y_val)), y_hat_train, 'train')
	utils.evaluate(y_test, y_hat_test, 'test')
	# -----------------------------------------------------------------------