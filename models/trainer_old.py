import numpy as np
import jax
import jax.numpy as jnp
import optax


def train_model_old(model, splits, N, B, S, hyperparams=None):
	#region hyperparamters ===============================================================
	seeds = jax.random.PRNGKey(0), jax.random.PRNGKey(1), jax.random.PRNGKey(2), jax.random.PRNGKey(3), jax.random.PRNGKey(4)

	global manual_hyperparameters
	hyperparameters = manual_hyperparameters.copy()
	if hyperparams is not None:
		for key in hyperparams:
			hyperparameters[key] = hyperparams[key]

	max_epochs = hyperparameters['max_epochs']
	max_patience = hyperparameters['max_patience']

	learning_rate = hyperparameters['learning_rate']

	n_dimensions = hyperparameters['dimensions']

	lambdas = Lambdas(
		loss_lambdas=(hyperparameters['loss_man_lambda'], hyperparameters['loss_res_lambda']),
		reg_lambdas=(hyperparameters['wb_reg_lambda'], hyperparameters['wm_reg_lambda']),
		cca_lambdas=(hyperparameters['ccadiag_lambda'], hyperparameters['ccaunit_lambda'])
	)
	#endregion ===========================================================================

	n_neurons = N.shape[1]
	n_states = S.shape[1]
	n_behaviors = B.shape[1]

	# training loop ==============================
	params_per_split = list()
	(losses_train_per_split,
	 man_losses_train_per_split,
	 res_losses_train_per_split,
	 mpe_losses_train_per_split) = list(), list(), list(), list()
	(losses_per_split,
	 man_losses_per_split,
	 res_losses_per_split,
	 mpe_losses_per_split) = list(), list(), list(), list()
	for s_id, (train, eval) in enumerate(splits):
		#region splitting and initializing (this also moves the data to the GPU) ==========================
		N_train, N_eval = jnp.array(N[train]), jnp.array(N[eval])
		S_train, S_eval = jnp.array(S[train]), jnp.array(S[eval])
		B_train, B_eval = jnp.array(B[train]), jnp.array(B[eval])

		params = model.init_params(N_train, B_train, S_train,
								   seeds,
								   (n_neurons, n_dimensions),
								   (n_states, n_dimensions),
								   (n_behaviors, n_neurons),
								   * ((n_dimensions, n_neurons),) if model.get_type() else (),
								  )

		optimizer = optax.adam(learning_rate=optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=max_epochs, alpha=1E-4))
		# optimizer = optax.adam(learning_rate=learning_rate)
		optimizer = optax.masked(optimizer, {'prep': False, 'model': True})
		opt_state = optimizer.init(params)
		#endregion ========================================================================================

		(losses_train,
		 man_losses_train,
		 res_losses_train,
		 mpe_losses_train) = [], [], [], []
		(losses,
		 man_losses,
		 res_losses,
		 mpe_losses) = [], [], [], []

		best_params = params
		(best_loss_train,
		 best_man_loss_train,
		 best_res_loss_train,
		 best_mpe_loss_train) = jnp.finfo(jnp.float32).max, jnp.finfo(jnp.float32).max, jnp.finfo(jnp.float32).max, jnp.finfo(jnp.float32).max
		(best_loss,
		 best_man_loss,
		 best_res_loss,
		 best_mpe_loss) = jnp.finfo(jnp.float32).max, jnp.finfo(jnp.float32).max, jnp.finfo(jnp.float32).max, jnp.finfo(jnp.float32).max

		# lr_step = 0
		patience = 0
		for epoch in range(max_epochs):

			(loss,
			 man_loss,
			 res_loss,
			 mpe_loss) = model.loss_fn_flattened(params, N_eval, B_eval, S_eval, lambdas)

			(opt_state, params, loss_types_train) = model.update(optimizer.update, opt_state, params, N_train, B_train, S_train, lambdas)

			(loss_train,
			 man_loss_train,
			 res_loss_train,
			 mpe_loss_train) = loss_types_train

			#region early stopping ===========================================================================
			if epoch >= max_patience:  # give it an initial time
				is_better = (loss < best_loss)
				best_loss = jnp.where(is_better, loss, best_loss)  # still on GPU

				if loss < best_loss:
					patience = 0

					best_params = params
					(best_loss_train,
					 best_man_loss_train,
					 best_res_loss_train,
					 best_mpe_loss_train) = loss_train, man_loss_train, res_loss_train, mpe_loss_train
					(best_loss,
					 best_man_loss,
					 best_res_loss,
					 best_mpe_loss) = loss, man_loss, res_loss, mpe_loss
				else:
					patience += 1
					if patience > max_patience:
						params = best_params

						losses_train = losses_train + [best_loss_train for _ in range(max_epochs - epoch)]
						man_losses_train = man_losses_train + [best_man_loss_train for _ in range(max_epochs - epoch)]
						res_losses_train = res_losses_train + [best_res_loss_train for _ in range(max_epochs - epoch)]
						mpe_losses_train = mpe_losses_train + [best_mpe_loss_train for _ in range(max_epochs - epoch)]

						losses = losses + [best_loss for _ in range(max_epochs - epoch)]
						man_losses = man_losses + [best_man_loss for _ in range(max_epochs - epoch)]
						res_losses = res_losses + [best_res_loss for _ in range(max_epochs - epoch)]
						mpe_losses = mpe_losses + [best_mpe_loss for _ in range(max_epochs - epoch)]

						print(f'Stopped after {epoch} epochs: ', end='')
						break

						# lr_step += 1
						# optimizer = optax.adam(learning_rate / (10 ** lr_step))
						# optimizer = optax.masked(optimizer, {'prep': False, 'model': True})
						# patience = 0
						# print(f'\nReduced learning rate from {learning_rate / (10 ** (lr_step - 1))} to {learning_rate / (10 ** lr_step)} after epoch {epoch} in split {s_id}', end='')
			#endregion =======================================================================================

			losses_train.append(loss_train)
			man_losses_train.append(man_loss_train)
			res_losses_train.append(res_loss_train)
			mpe_losses_train.append(mpe_loss_train)

			losses.append(loss)
			man_losses.append(man_loss)
			res_losses.append(res_loss)
			mpe_losses.append(mpe_loss)

		params_per_split.append(params)

		losses_train_per_split.append(jax.device_get(losses_train))
		man_losses_train_per_split.append(jax.device_get(man_losses_train))
		res_losses_train_per_split.append(jax.device_get(res_losses_train))
		mpe_losses_train_per_split.append(jax.device_get(mpe_losses_train))

		losses_per_split.append(jax.device_get(losses))
		man_losses_per_split.append(jax.device_get(man_losses))
		res_losses_per_split.append(jax.device_get(res_losses))
		mpe_losses_per_split.append(jax.device_get(mpe_losses))

		print(f'Split {s_id} completed. ', end='')

	# ============================================

	# return
	return (params_per_split,
			np.array(losses_train_per_split), np.array(man_losses_train_per_split), np.array(res_losses_train_per_split), np.array(mpe_losses_train_per_split),
			np.array(losses_per_split), np.array(man_losses_per_split), np.array(res_losses_per_split), np.array(mpe_losses_per_split))