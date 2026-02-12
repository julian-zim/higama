import jax
import jax.numpy as jnp


def train(
	lambdas,
	epochs_max, patience_max,
	model, optimizer,
	opt_state, params,
	N_train, B_train, S_train,
	N_eval,  B_eval,  S_eval,
):

	def loop_condition(carry):
		epoch = carry[0]
		patience = carry[1]
		return jnp.logical_and(epoch < epochs_max, patience < patience_max)

	def loop_body(carry):
		# unpack
		(epoch, patience,
		 opt_state, params,
		 params_best,
		 loss_train_best,	man_loss_train_best,	res_loss_train_best,	mpe_loss_train_best,
		 loss_best,			man_loss_best,			res_loss_best,			mpe_loss_best,
		 losses_train,		man_losses_train,		res_losses_train,		mpe_losses_train,
		 losses,			man_losses,				res_losses,				mpe_losses,
		 ) = carry

		# update
		opt_state_new, params_new, loss_types_train = model.update(
			optimizer.update, opt_state, params, N_train, B_train, S_train, lambdas
		)
		(loss_train,
		 man_loss_train,
		 res_loss_train,
		 mpe_loss_train) = loss_types_train
		(loss,
		 man_loss,
		 res_loss,
		 mpe_loss) = model.loss_fn_flattened(params, N_eval, B_eval, S_eval, lambdas)

		losses_train		= losses_train.at[epoch].set(loss_train)
		man_losses_train	= man_losses_train.at[epoch].set(man_loss_train)
		res_losses_train	= res_losses_train.at[epoch].set(res_loss_train)
		mpe_losses_train	= mpe_losses_train.at[epoch].set(mpe_loss_train)
		losses				= losses.at[epoch].set(loss)
		man_losses			= man_losses.at[epoch].set(man_loss)
		res_losses			= res_losses.at[epoch].set(res_loss)
		mpe_losses			= mpe_losses.at[epoch].set(mpe_loss)

		# early stopping
		warmed_up = epoch >= patience_max
		improved_loss = loss < loss_best
		update_best = jnp.logical_and(warmed_up, improved_loss)

		epoch_new = epoch + 1
		patience_new = jnp.where(warmed_up, jnp.where(improved_loss, 0, patience + 1), patience)
		params_best_new = jax.tree.map(lambda x, y: jnp.where(update_best, x, y), params_new, params_best)

		loss_train_best_new		= jnp.where(update_best, loss_train,		loss_train_best)
		man_loss_train_best_new = jnp.where(update_best, man_loss_train,	man_loss_train_best)
		res_loss_train_best_new = jnp.where(update_best, res_loss_train,	res_loss_train_best)
		mpe_loss_train_best_new = jnp.where(update_best, mpe_loss_train,	mpe_loss_train_best)
		loss_best_new			= jnp.where(update_best, loss,				loss_best)
		man_loss_best_new		= jnp.where(update_best, man_loss,			man_loss_best)
		res_loss_best_new		= jnp.where(update_best, res_loss,			res_loss_best)
		mpe_loss_best_new		= jnp.where(update_best, mpe_loss,			mpe_loss_best)

		# pack
		carry_new = (
			epoch_new, patience_new,
			opt_state_new, params_new,
			params_best_new,
			loss_train_best_new,	man_loss_train_best_new,	res_loss_train_best_new,	mpe_loss_train_best_new,
			loss_best_new,			man_loss_best_new,			res_loss_best_new,			mpe_loss_best_new,
			losses_train,			man_losses_train,			res_losses_train,			mpe_losses_train,
			losses,					man_losses,					res_losses,					mpe_losses,
		)
		return carry_new

	# initialize
	epoch = jnp.array(0, jnp.int32)
	patience = jnp.array(0, jnp.int32)

	losses_train		= jnp.empty((epochs_max,), dtype=jnp.float32)
	man_losses_train	= jnp.empty((epochs_max,), dtype=jnp.float32)
	res_losses_train	= jnp.empty((epochs_max,), dtype=jnp.float32)
	mpe_losses_train	= jnp.empty((epochs_max,), dtype=jnp.float32)
	losses				= jnp.empty((epochs_max,), dtype=jnp.float32)
	man_losses			= jnp.empty((epochs_max,), dtype=jnp.float32)
	res_losses			= jnp.empty((epochs_max,), dtype=jnp.float32)
	mpe_losses			= jnp.empty((epochs_max,), dtype=jnp.float32)

	fmax = jnp.array(jnp.finfo(jnp.float32).max, jnp.float32)
	loss_train_best		= fmax
	man_loss_train_best = fmax
	res_loss_train_best = fmax
	mpe_loss_train_best = fmax
	loss_best			= fmax
	man_loss_best 		= fmax
	res_loss_best 		= fmax
	mpe_loss_best 		= fmax

	params_best = params

	# main loop
	carry = (
		epoch, patience,
		opt_state, params,
		params_best,
		loss_train_best,	man_loss_train_best,	res_loss_train_best,	mpe_loss_train_best,
		loss_best,			man_loss_best,			res_loss_best,			mpe_loss_best,
		losses_train,		man_losses_train,		res_losses_train,		mpe_losses_train,
		losses,				man_losses,				res_losses,				mpe_losses,
	)

	carry = jax.lax.while_loop(loop_condition, loop_body, carry)

	(epoch_end, patience_end,
	 opt_state_end, params_end,
	 params_best,
	 loss_train_best,	man_loss_train_best,	res_loss_train_best,	mpe_loss_train_best,
	 loss_best,			man_loss_best,			res_loss_best,			mpe_loss_best,
	 losses_train,		man_losses_train,		res_losses_train,		mpe_losses_train,
	 losses,			man_losses,				res_losses, 			mpe_losses,
	 ) = carry

	# save
	losses_train		= jnp.where(jnp.arange(epochs_max) < epoch_end, losses_train, loss_train_best)
	man_losses_train	= jnp.where(jnp.arange(epochs_max) < epoch_end, man_losses_train, man_loss_train_best)
	res_losses_train	= jnp.where(jnp.arange(epochs_max) < epoch_end, res_losses_train, res_loss_train_best)
	mpe_losses_train	= jnp.where(jnp.arange(epochs_max) < epoch_end, mpe_losses_train, mpe_loss_train_best)

	losses				= jnp.where(jnp.arange(epochs_max) < epoch_end, losses, loss_best)
	man_losses			= jnp.where(jnp.arange(epochs_max) < epoch_end, man_losses, man_loss_best)
	res_losses			= jnp.where(jnp.arange(epochs_max) < epoch_end, res_losses, res_loss_best)
	mpe_losses			= jnp.where(jnp.arange(epochs_max) < epoch_end, mpe_losses, mpe_loss_best)

	# set final params
	stopped_early = epoch_end < epochs_max
	params_final = jax.tree.map(lambda best, end: jnp.where(stopped_early, best, end), params_best, params_end)

	# return
	return params_final, losses_train, man_losses_train, res_losses_train, mpe_losses_train, losses, man_losses, res_losses, mpe_losses, epoch_end

train = jax.jit(train, static_argnums=(0, 1, 2, 3, 4,))
