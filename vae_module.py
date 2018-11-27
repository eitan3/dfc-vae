import numpy as np
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import os


class VAE_Model(gluon.HybridBlock):
    def __init__(self, ctx, batch_size, latent_space, encoder, encoder_args, decoder, decoder_args, dfc_class):
        super(VAE_Model, self).__init__()
        self.ctx = ctx
        self.batch_size = batch_size
        self.latent_space = latent_space
        with self.name_scope():
            self.encoder = encoder(encoder_args)
            self.decoder = decoder(decoder_args)
            self.dfc_module = dfc_class()
            self.mu = gluon.nn.Dense(latent_space)
            self.mu_bn = gluon.nn.BatchNorm(in_channels=latent_space, center=True, scale=True, prefix='mu_bn',
                                            gamma_initializer=mx.init.MinMaxUniform(0.02, 1.0))
            self.logvar = gluon.nn.Dense(latent_space)
            self.logvar_bn = gluon.nn.BatchNorm(in_channels=latent_space, center=True, scale=True, prefix='logvar_bn',
                                                gamma_initializer=mx.init.MinMaxUniform(0.02, 1.0))

    def reparametrize(self, F, mu, logvar):
        eps = F.random_normal(loc=0, scale=1, shape=(self.batch_size, self.latent_space), ctx=self.ctx)
        z = mu + F.exp(0.5 * logvar) * eps
        return z

    def hybrid_forward(self, F, x):
        encode = self.encoder(x)

        mu = self.mu(encode)
        mu = self.mu_bn(mu)
        logvar = self.logvar(encode)
        logvar = self.logvar_bn(logvar)
        logvar = F.Activation(logvar, act_type='softrelu')
        logvar = logvar + 1e-6
        z = self.reparametrize(F, mu, logvar)

        decode = self.decoder(z)

        real_img_dfc = self.dfc_module(x)
        fake_img_dfc = self.dfc_module(decode)
        dfc_out = [real_img_dfc, fake_img_dfc]

        return decode, mu, logvar, dfc_out


def calc_encoding_moving_var(step, capacity_change_duration, capacity_limit):
    if step > capacity_change_duration:
      c = capacity_limit
    else:
      c = capacity_limit * (step / float(capacity_change_duration))
    return c


def vae_loss(recon_loss, pred, imgs, dfc_out, mu, logvar, gamma, capacity, kl_scale):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """

    # DFC loss
    real_imgs = dfc_out[0]
    fake_imgs = dfc_out[1]
    total_dfc = nd.mean(nd.square(real_imgs[0] - fake_imgs[0]), [1, 2, 3])
    for i in range(1, len(real_imgs)):
        total_dfc = total_dfc + nd.mean(nd.square(nd.flatten(real_imgs[i]) - nd.flatten(fake_imgs[i])), 1)
    dfc_scale = 1.0
    total_dfc = total_dfc * dfc_scale

    # KL Loss
    KL = -0.5 * nd.sum(1 + logvar - mu * mu - nd.exp(logvar), axis=1)

    # Recon Loss
    rc_loss = recon_loss(pred, imgs)

    loss = nd.mean(rc_loss) + nd.mean(total_dfc) + (gamma * nd.abs(nd.mean(KL) - capacity) * kl_scale)
    return loss


def train_vae(ctx, prefix, train_iter, test_iter, encoder, decoder, dfc_class, recon_loss,
              evaluate_accuracy, save_batch_results, beta_args, args):

    # Create model for training
    net = VAE_Model(ctx, args['batch_size'], args['latent_space'], encoder, args['encoder_args'],
                    decoder, args['decoder_args'],
                    dfc_class)

    # Initialize model parameters
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.34), ctx=ctx)

    # Create optimizer
    all_params = net.collect_params()
    for key, val in all_params.items():
        if "dfc_module" in key:
            del all_params._params[key]
    if isinstance(args['optimizer'], mx.optimizer.Optimizer):
        trainer = gluon.Trainer(all_params, args['optimizer'], optimizer_params={})
    else:
        optimizer_params = {'learning_rate': args['lr']}
        for key, val in args['optimizer_args'].items():
            optimizer_params[key] = val
        trainer = gluon.Trainer(all_params, args['optimizer'], optimizer_params)

    # Hybridize for better performance
    net.hybridize()

    # Start training
    global_step = 0
    moving_vars_step = 0
    for epoch in range(args['epochs']):
        train_loss = 0
        for batch_idx, data in enumerate(train_iter):
            # get images
            imgs, _ = data
            imgs = imgs.as_in_context(ctx)

            # calculate gamma and capacity
            gamma = calc_encoding_moving_var(moving_vars_step,
                                             beta_args['gamma_change_duration'], beta_args['gamma_maximum'])
            capacity = calc_encoding_moving_var(moving_vars_step,
                                                beta_args['capacity_change_duration'], beta_args['capacity_maximum'])
            if gamma < beta_args['gamma_minimum']:
                gamma = beta_args['gamma_minimum']

            # record gradients and loss
            with mx.autograd.record():
                pred, mu, logvar, dfc_out = net(imgs)
                kl_scale = beta_args['KL_scale']
                loss = vae_loss(recon_loss, pred, imgs, dfc_out, mu, logvar, gamma, capacity, kl_scale)

            # Record loss
            train_loss += nd.mean(loss).asscalar()

            # Optimizer
            loss.backward()
            trainer.step(args['batch_size'])

            # Update global step and moving vars step
            global_step += 1
            if epoch % beta_args['update_vars_every_X_epoch'] == 0 and epoch >= beta_args['start_moving_vars_epoch']:
                moving_vars_step += 1

            # Print data to screen
            if batch_idx % args['log_interval'] == 0:
                # Calculate accuracy
                batch_accuracy = evaluate_accuracy(imgs, net, ctx)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.7f}\tAcc: {:.7f} \tgamma: {:.7f} \tcap: {:.7f}'.format(
                    epoch, batch_idx * args['batch_size'],
                    train_iter.num_examples, 100. * batch_idx / int(
                        train_iter.num_examples / args['batch_size']), nd.mean(loss).asscalar() / args['batch_size'],
                    batch_accuracy.asscalar(), gamma, capacity))

        print('====> Epoch: {} Average loss: {:.7f}'.format(epoch, train_loss / train_iter.num_examples))

        # Evaluate on test split
        for batch_idx, data in enumerate(test_iter):
            test_data, _ = data
            test_data = test_data.as_in_context(ctx)
            recon_batch, _, _, _ = net(test_data)
            save_batch_results('./vae_imgs/' + prefix + "/" + str(epoch) + '/', test_data.asnumpy(), recon_batch.asnumpy())
            break
        test_iter.reset()

        # Save parameters
        if not os.path.exists(args['param_save_path']):
            os.makedirs(args['param_save_path'])
        net.save_parameters(args['param_save_path'] + prefix)