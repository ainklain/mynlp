from ts_idx.bnn import BNN
from collections import OrderedDict
import tensorflow as tf
from absl import flags
from ts_idx import tf_utils
import math

FLAGS = flags.FLAGS


class BMAML:
    def __init__(self,
                 dim_input,
                 dim_output,
                 dim_hidden=32,
                 num_layers=4,
                 num_particles=2,
                 max_test_step=5):
        # model size
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.num_particles = num_particles

        # learning rate
        self.follow_lr = FLAGS.follow_lr
        self.leader_lr = FLAGS.leader_lr
        self.meta_lr = FLAGS.meta_lr

        # for test time
        self.max_test_step = max_test_step

        # build model
        self.bnn_list = [BNN(dim_input=self.dim_input,
                             dim_output=self.dim_output,
                             dim_hidden=self.dim_hidden,
                             num_layers=self.num_layers,
                             is_bnn=True) for _ in range(num_particles)]

        # init parameters
        self.W_network_particles = None

    def metatrain(self, inputs, follow_step=1, leader_step=1, is_training=True):
        # decompose input data
        [follow_x, leader_x, valid_x,
         follow_y, leader_y, valid_y] = inputs

        if is_training:
            max_follow_step = follow_step
        else:
            max_follow_step = max(follow_step, self.max_test_step)

        with tf.GradientTape() as tape:
            var_lists = []
            for i, bnn_p in enumerate(self.bnn_list):
                # var_lists.append(bnn_p.get_trainable_variables())
                var_lists.append(bnn_p.get_var_lists())
            tape.watch(var_lists)

            [step_follow_weight_var,
             step_follow_data_var,
             step_follow_train_llik,
             step_follow_valid_llik,
             step_follow_train_loss,
             step_follow_valid_loss,
             step_follow_train_pred,
             step_follow_valid_pred,
             step_follow_weight_lprior,
             step_follow_gamma_lprior,
             step_follow_lambda_lprior,
             step_follow_lpost,
             step_follow_kernel_h,
             WW_follow] = self.update_particles(follow_x, follow_y, valid_x, valid_y, follow_step, self.follow_lr)
            # train_x, train_y, valid_x, valid_y, num_updates, lr = follow_x, follow_y, valid_x, valid_y, follow_step, self.follow_lr
            [step_leader_weight_var,
             step_leader_data_var,
             step_leader_train_llik,
             step_leader_valid_llik,
             step_leader_train_loss,
             step_leader_valid_loss,
             step_leader_train_pred,
             step_leader_valid_pred,
             step_leader_weight_lprior,
             step_leader_gamma_lprior,
             step_leader_lambda_lprior,
             step_leader_lpost,
             step_leader_kernel_h,
             WW_leader] = self.update_particles(leader_x, leader_y, valid_x, valid_y, leader_step, self.leader_lr)
            # train_x, train_y, valid_x, valid_y, num_updates, lr = leader_x, leader_y, valid_x, valid_y, leader_step, self.leader_lr
            #############
            # meta loss #
            #############
            meta_loss = []
            # for each particle
            for p_idx in range(self.num_particles):
                # compute follower and leader distance
                p_dist_list = []
                for j in range(len(WW_leader[p_idx])-2):
                    p_dist = tf.square(WW_follow[p_idx][j] - tf.stop_gradient(WW_leader[p_idx][j]))
                    p_dist = tf.reduce_sum(p_dist)
                    p_dist_list.append(p_dist)
                meta_loss.append(tf.reduce_sum(p_dist_list))

            # mean over particles
            meta_loss = tf.reduce_sum(meta_loss)

            if is_training:
                self.total_follow_weight_lprior = [tf.reduce_mean(step_follow_weight_lprior[j])
                                                   for j in range(FLAGS.follow_step + 1)]
                self.total_follow_gamma_lprior = [tf.reduce_mean(step_follow_gamma_lprior[j])
                                                  for j in range(FLAGS.follow_step + 1)]
                self.total_follow_lambda_lprior = [tf.reduce_mean(step_follow_lambda_lprior[j])
                                                   for j in range(FLAGS.follow_step + 1)]
                self.total_follow_train_llik = [tf.reduce_mean(step_follow_train_llik[j])
                                                for j in range(FLAGS.follow_step + 1)]
                self.total_follow_valid_llik = [tf.reduce_mean(step_follow_valid_llik[j])
                                                for j in range(FLAGS.follow_step + 1)]
                self.total_follow_train_loss = [tf.reduce_mean(step_follow_train_loss[j])
                                                for j in range(FLAGS.follow_step + 1)]
                self.total_follow_valid_loss = [tf.reduce_mean(step_follow_valid_loss[j])
                                                for j in range(FLAGS.follow_step + 1)]
                self.total_follow_weight_var = [tf.reduce_mean(step_follow_weight_var[j])
                                                for j in range(FLAGS.follow_step + 1)]
                self.total_follow_data_var = [tf.reduce_mean(step_follow_data_var[j])
                                              for j in range(FLAGS.follow_step + 1)]
                self.total_follow_lpost = [tf.reduce_mean(step_follow_lpost[j])
                                           for j in range(FLAGS.follow_step)]
                self.total_follow_kernel_h = [tf.reduce_mean(step_follow_kernel_h[j])
                                              for j in range(FLAGS.follow_step)]
                self.total_leader_weight_lprior = [tf.reduce_mean(step_leader_weight_lprior[j])
                                                   for j in range(FLAGS.leader_step + 1)]
                self.total_leader_gamma_lprior = [tf.reduce_mean(step_leader_gamma_lprior[j])
                                                  for j in range(FLAGS.leader_step + 1)]
                self.total_leader_lambda_lprior = [tf.reduce_mean(step_leader_lambda_lprior[j])
                                                   for j in range(FLAGS.leader_step + 1)]
                self.total_leader_train_llik = [tf.reduce_mean(step_leader_train_llik[j])
                                                for j in range(FLAGS.leader_step + 1)]
                self.total_leader_valid_llik = [tf.reduce_mean(step_leader_valid_llik[j])
                                                for j in range(FLAGS.leader_step + 1)]
                self.total_leader_train_loss = [tf.reduce_mean(step_leader_train_loss[j])
                                                for j in range(FLAGS.leader_step + 1)]
                self.total_leader_valid_loss = [tf.reduce_mean(step_leader_valid_loss[j])
                                                for j in range(FLAGS.leader_step + 1)]
                self.total_leader_weight_var = [tf.reduce_mean(step_leader_weight_var[j])
                                                for j in range(FLAGS.leader_step + 1)]
                self.total_leader_data_var = [tf.reduce_mean(step_leader_data_var[j])
                                              for j in range(FLAGS.leader_step + 1)]
                self.total_leader_lpost = [tf.reduce_mean(step_leader_lpost[j])
                                           for j in range(FLAGS.leader_step)]
                self.total_leader_kernel_h = [tf.reduce_mean(step_leader_kernel_h[j])
                                              for j in range(FLAGS.leader_step)]
                self.total_meta_loss = tf.reduce_mean(meta_loss)

                # prediction
                self.total_train_z_list = step_follow_train_pred
                self.total_valid_z_list = step_follow_valid_pred

                ###############
                # meta update #
                ###############
                # set optimizer
                optimizer = tf.keras.optimizers.Adam(learning_rate=self.meta_lr)

                # compute gradient
                gv_list = tape.gradient(self.total_meta_loss, var_lists)

                # gradient clipping
                if FLAGS.out_grad_clip > 0:
                    gv_list = [(tf.clip_by_value(grad, -FLAGS.out_grad_clip, FLAGS.out_grad_clip), var)
                               for grad, var in gv_list]

                # apply optimizer
                optimizer.apply_gradients(zip(gv_list, var_lists))
            else:
                # summarize results
                self.eval_train_llik = [tf.reduce_mean(step_follow_train_llik[j])
                                        for j in range(max_follow_step + 1)]
                self.eval_train_loss = [tf.reduce_mean(step_follow_train_loss[j])
                                        for j in range(max_follow_step + 1)]
                self.eval_valid_llik = [tf.reduce_mean(step_follow_valid_llik[j])
                                        for j in range(max_follow_step + 1)]
                self.eval_valid_loss = [tf.reduce_mean(step_follow_valid_loss[j])
                                        for j in range(max_follow_step + 1)]

                # prediction
                self.eval_train_z_list = step_follow_train_pred
                self.eval_valid_z_list = step_follow_valid_pred

    def update_particles(self, train_x, train_y, valid_x, valid_y, num_updates, lr):
        # Stein Variational Gradient Descent
        step_weight_lprior = [None] * (num_updates + 1)
        step_lambda_lprior = [None] * (num_updates + 1)
        step_gamma_lprior = [None] * (num_updates + 1)
        step_train_llik = [None] * (num_updates + 1)
        step_valid_llik = [None] * (num_updates + 1)
        step_train_loss = [None] * (num_updates + 1)
        step_valid_loss = [None] * (num_updates + 1)
        step_train_pred = [None] * (num_updates + 1)
        step_valid_pred = [None] * (num_updates + 1)
        step_weight_var = [None] * (num_updates + 1)
        step_data_var = [None] * (num_updates + 1)
        step_kernel_h = [None] * num_updates
        step_lpost = [None] * num_updates

        WW = [self.bnn_list[p_idx].get_var_lists() for p_idx in range(self.num_particles)]
        for s_idx in range(num_updates + 1):
            # for each particle
            train_z_list = []
            valid_z_list = []
            train_llik_list = []
            valid_llik_list = []
            weight_lprior_list = []
            lambda_lprior_list = []
            gamma_lprior_list = []
            weight_var_list = []
            data_var_list = []

            if s_idx < num_updates:
                # for each particle, compute gradient
                dWW = []

            for p_idx in range(self.num_particles):
                with tf.GradientTape() as tape:
                    bnn_p = self.bnn_list[p_idx]
                    var_lists_p = WW[p_idx]
                    tape.watch(var_lists_p)

                    # compute prediction
                    train_z = bnn_p.forward_network(inputs=train_x)
                    valid_z = bnn_p.forward_network(inputs=valid_x)

                    # compute data log-likelihood
                    train_llik_list.append(bnn_p.log_likelihood_data(predict_y=train_z, target_y=train_y))
                    valid_llik_list.append(bnn_p.log_likelihood_data(predict_y=valid_z, target_y=valid_y))

                    # add to list
                    train_z_list.append(train_z)
                    valid_z_list.append(valid_z)

                    # compute log-posterior
                    weight_lprior, gamma_lprior, lambda_lprior = bnn_p.log_prior_weight()
                    weight_lprior_list.append(weight_lprior)
                    lambda_lprior_list.append(lambda_lprior)
                    gamma_lprior_list.append(gamma_lprior)

                    # get variance
                    weight_var_list.append(tf.math.reciprocal(tf.exp(bnn_p.log_lambda)))
                    data_var_list.append(tf.math.reciprocal(tf.exp(bnn_p.log_gamma)))

                    # update follower
                    if s_idx < num_updates:
                        lpost = weight_lprior + gamma_lprior + lambda_lprior + tf.reduce_sum(train_llik_list[p_idx], axis=[1, 2])

                        dWp = tape.gradient(lpost, var_lists_p)
                        if FLAGS.stop_grad:
                            dWp = tf.stop_gradient(dWp)

                        # save particle loss
                        if p_idx == 0:
                            step_lpost[s_idx] = []
                        step_lpost[s_idx].append(lpost)

                        dWW.append(dWp)


            if s_idx < num_updates:
                # mean over particle loss
                step_lpost[s_idx] = tf.reduce_mean(step_lpost[s_idx])

                # make SVGD gradient and flatten particles and its gradient
                WW_tensor = self.diclist2tensor(WW)
                dWW_tensor = self.diclist2tensor(dWW)

                # compute kernel
                [kernel_mat, grad_kernel, kernel_h] = self.kernel(particle_tensor=WW_tensor)
                dWW_tensor = tf.divide(tf.matmul(kernel_mat, dWW_tensor) + grad_kernel, self.num_particles)
                step_kernel_h[s_idx] = kernel_h

                dWW = self.tensor2diclist(dWW_tensor)
                # update particles
                for p_idx in range(self.num_particles):
                    bnn_p = self.bnn_list[p_idx]
                    WW_p = WW[p_idx]
                    dWW_p = dWW[p_idx]

                    # for each param
                    param_vals = []
                    for j in range(len(WW_p)):
                        if FLAGS.in_grad_clip > 0:
                            grad = tf.clip_by_value(dWW_p[j], -FLAGS.in_grad_clip, FLAGS.in_grad_clip)
                        else:
                            grad = dWW_p[j]
                        # param_names.append(key)
                        # if 'log' in key:
                        if j >= len(WW_p) - 2:
                            # 뒤에 두개는 log_lambda, log_gamma
                            param_vals.append(WW_p[j] + FLAGS.lambda_lr * lr * grad)
                        else:
                            param_vals.append(WW_p[j] + lr * grad)

                    bnn_p.set_var_lists(param_vals)
                    WW[p_idx] = bnn_p.get_var_lists()


            # get mean prediction
            train_z = tf.reduce_mean(train_z_list, 0)
            valid_z = tf.reduce_mean(valid_z_list, 0)

            # aggregate particle results
            step_weight_lprior[s_idx] = tf.reduce_mean(weight_lprior_list)
            step_gamma_lprior[s_idx] = tf.reduce_mean(gamma_lprior_list)
            step_lambda_lprior[s_idx] = tf.reduce_mean(lambda_lprior_list)
            step_train_llik[s_idx] = tf.reduce_mean([tf.reduce_mean(train_llik) for train_llik in train_llik_list])
            step_valid_llik[s_idx] = tf.reduce_mean([tf.reduce_mean(valid_llik) for valid_llik in valid_llik_list])
            step_train_loss[s_idx] = tf.reduce_mean(tf.square(train_z - train_y))
            step_valid_loss[s_idx] = tf.reduce_mean(tf.square(valid_z - valid_y))
            step_train_pred[s_idx] = tf.concat([tf.expand_dims(train_z, 0) for train_z in train_z_list], axis=0)
            step_valid_pred[s_idx] = tf.concat([tf.expand_dims(valid_z, 0) for valid_z in valid_z_list], axis=0)
            step_weight_var[s_idx] = tf.reduce_mean(weight_var_list)
            step_data_var[s_idx] = tf.reduce_mean(data_var_list)

        return [step_weight_var,
                step_data_var,
                step_train_llik,
                step_valid_llik,
                step_train_loss,
                step_valid_loss,
                step_train_pred,
                step_valid_pred,
                step_weight_lprior,
                step_gamma_lprior,
                step_lambda_lprior,
                step_lpost,
                step_kernel_h,
                WW]

    def update_particle_original(self,
                        train_x,
                        train_y,
                        valid_x,
                        valid_y,
                        WW,
                        num_updates,
                        lr):
        # for each step
        step_weight_lprior = [None] * (num_updates + 1)
        step_lambda_lprior = [None] * (num_updates + 1)
        step_gamma_lprior = [None] * (num_updates + 1)
        step_train_llik = [None] * (num_updates + 1)
        step_valid_llik = [None] * (num_updates + 1)
        step_train_loss = [None] * (num_updates + 1)
        step_valid_loss = [None] * (num_updates + 1)
        step_train_pred = [None] * (num_updates + 1)
        step_valid_pred = [None] * (num_updates + 1)
        step_weight_var = [None] * (num_updates + 1)
        step_data_var = [None] * (num_updates + 1)
        step_kernel_h = [None] * num_updates
        step_lpost = [None] * num_updates

        for s_idx in range(num_updates + 1):
            # for each particle
            train_z_list = []
            valid_z_list = []
            train_llik_list = []
            valid_llik_list = []
            weight_lprior_list = []
            lambda_lprior_list = []
            gamma_lprior_list = []
            weight_var_list = []
            data_var_list = []
            for p_idx in range(self.num_particles):
                # compute prediction
                train_z = self.forward_network(x=train_x, W_dict=WW[p_idx])
                valid_z = self.forward_network(x=valid_x, W_dict=WW[p_idx])

                # compute data log-likelihood
                train_llik_list.append(self.bnn.log_likelihood_data(predict_y=train_z,
                                                                    target_y=train_y,
                                                                    log_gamma=WW[p_idx]['log_gamma']))
                valid_llik_list.append(self.bnn.log_likelihood_data(predict_y=valid_z,
                                                                    target_y=valid_y,
                                                                    log_gamma=WW[p_idx]['log_gamma']))

                # add to list
                train_z_list.append(train_z)
                valid_z_list.append(valid_z)

                # compute log-posterior
                weight_lprior, gamma_lprior, lambda_lprior = self.bnn.log_prior_weight(W_dict=WW[p_idx])
                weight_lprior_list.append(weight_lprior)
                lambda_lprior_list.append(lambda_lprior)
                gamma_lprior_list.append(gamma_lprior)

                # get variance
                weight_var_list.append(tf.reciprocal(tf.exp(WW[p_idx]['log_lambda'])))
                data_var_list.append(tf.reciprocal(tf.exp(WW[p_idx]['log_gamma'])))

            # update follower
            if s_idx < num_updates:
                # vectorize
                WW_tensor = self.diclist2tensor(WW=WW)

                # for each particle, compute gradien
                dWW = []
                for p_idx in range(self.num_particles):
                    # compute loss and gradient
                    lpost = weight_lprior_list[p_idx]
                    lpost += lambda_lprior_list[p_idx]
                    lpost += gamma_lprior_list[p_idx]
                    lpost += tf.reduce_sum(train_llik_list[p_idx])
                    dWp = tf.gradients(ys=lpost,
                                       xs=list(WW[p_idx].values()))

                    # save particle loss
                    if p_idx == 0:
                        step_lpost[s_idx] = []
                    step_lpost[s_idx].append(lpost)

                    # stop gradient to avoid second order
                    if FLAGS.stop_grad:
                        dWp = [tf.stop_gradient(grad) for grad in dWp]

                    # re-order
                    dWW.append(OrderedDict(zip(WW[p_idx].keys(), dWp)))

                # mean over particle loss
                step_lpost[s_idx] = tf.reduce_mean(step_lpost[s_idx])

                # make SVGD gradient and flatten particles and its gradient
                dWW_tensor = self.diclist2tensor(WW=dWW)

                # compute kernel
                [kernel_mat,
                 grad_kernel,
                 kernel_h] = self.kernel(particle_tensor=WW_tensor)
                dWW_tensor = tf.divide(tf.matmul(kernel_mat, dWW_tensor) + grad_kernel, self.num_particles)
                step_kernel_h[s_idx] = kernel_h

                # re-order gradient into particle forms
                dWW = self.tensor2diclist(tensor=dWW_tensor)

                # update particles
                for p_idx in range(self.num_particles):
                    # for each param
                    param_names = []
                    param_vals = []
                    for key in list(WW[p_idx].keys()):
                        if FLAGS.in_grad_clip > 0:
                            grad = tf.clip_by_value(dWW[p_idx][key], -FLAGS.in_grad_clip, FLAGS.in_grad_clip)
                        else:
                            grad = dWW[p_idx][key]
                        param_names.append(key)
                        if 'log' in key:
                            param_vals.append(WW[p_idx][key] + FLAGS.lambda_lr * lr * grad)
                        else:
                            param_vals.append(WW[p_idx][key] + lr * grad)
                    WW[p_idx] = OrderedDict(zip(param_names, param_vals))

            # get mean prediction
            train_z = tf.reduce_mean(train_z_list, 0)
            valid_z = tf.reduce_mean(valid_z_list, 0)

            # aggregate particle results
            step_weight_lprior[s_idx] = tf.reduce_mean(weight_lprior_list)
            step_gamma_lprior[s_idx] = tf.reduce_mean(gamma_lprior_list)
            step_lambda_lprior[s_idx] = tf.reduce_mean(lambda_lprior_list)
            step_train_llik[s_idx] = tf.reduce_mean([tf.reduce_mean(train_llik) for train_llik in train_llik_list])
            step_valid_llik[s_idx] = tf.reduce_mean([tf.reduce_mean(valid_llik) for valid_llik in valid_llik_list])
            step_train_loss[s_idx] = tf.reduce_mean(tf.square(train_z - train_y))
            step_valid_loss[s_idx] = tf.reduce_mean(tf.square(valid_z - valid_y))
            step_train_pred[s_idx] = tf.concat([tf.expand_dims(train_z, 0) for train_z in train_z_list], axis=0)
            step_valid_pred[s_idx] = tf.concat([tf.expand_dims(valid_z, 0) for valid_z in valid_z_list], axis=0)
            step_weight_var[s_idx] = tf.reduce_mean(weight_var_list)
            step_data_var[s_idx] = tf.reduce_mean(data_var_list)

        return [step_weight_var,
                step_data_var,
                step_train_llik,
                step_valid_llik,
                step_train_loss,
                step_valid_loss,
                step_train_pred,
                step_valid_pred,
                step_weight_lprior,
                step_gamma_lprior,
                step_lambda_lprior,
                step_lpost,
                step_kernel_h,
                WW]

    # build model
    def construct_model(self, is_training=True):
        print('start model construction')
        # init model
        with tf.variable_scope('model', reuse=None) as training_scope:
            # init parameters
            if is_training or self.W_network_particles is None:
                # network parameters
                self.W_network_particles = [self.construct_network_weights(scope='network{}'.format(p_idx))
                                            for p_idx in range(self.num_particles)]
            else:
                training_scope.reuse_variables()

            # set number of follower steps
            if is_training:
                max_follow_step = FLAGS.follow_step
            else:
                max_follow_step = max(FLAGS.follow_step, self.max_test_step)

            # task-wise inner loop
            def fast_learn_one_task(inputs):
                # decompose input data
                [follow_x, leader_x, valid_x,
                 follow_y, leader_y, valid_y] = inputs

                ############
                # follower #
                ############
                # get the follow particles
                WW_follow = [OrderedDict(zip(W_dic.keys(), W_dic.values()))
                             for W_dic in self.W_network_particles]

                # update
                [step_follow_weight_var,
                 step_follow_data_var,
                 step_follow_train_llik,
                 step_follow_valid_llik,
                 step_follow_train_loss,
                 step_follow_valid_loss,
                 step_follow_train_pred,
                 step_follow_valid_pred,
                 step_follow_weight_lprior,
                 step_follow_gamma_lprior,
                 step_follow_lambda_lprior,
                 step_follow_lpost,
                 step_follow_kernel_h,
                 WW_follow] = self.update_particle(train_x=follow_x,
                                                   train_y=follow_y,
                                                   valid_x=valid_x,
                                                   valid_y=valid_y,
                                                   WW=WW_follow,
                                                   num_updates=max_follow_step,
                                                   lr=self.follow_lr)

                ##########
                # leader #
                ##########
                # get the follow particles
                WW_leader = [OrderedDict(zip(W_dic.keys(), W_dic.values()))
                             for W_dic in WW_follow]

                # update
                [step_leader_weight_var,
                 step_leader_data_var,
                 step_leader_train_llik,
                 step_leader_valid_llik,
                 step_leader_train_loss,
                 step_leader_valid_loss,
                 step_leader_train_pred,
                 step_leader_valid_pred,
                 step_leader_weight_lprior,
                 step_leader_gamma_lprior,
                 step_leader_lambda_lprior,
                 step_leader_lpost,
                 step_leader_kernel_h,
                 WW_leader] = self.update_particle(train_x=leader_x,
                                                   train_y=leader_y,
                                                   valid_x=valid_x,
                                                   valid_y=valid_y,
                                                   WW=WW_leader,
                                                   num_updates=FLAGS.leader_step,
                                                   lr=self.leader_lr)

                #############
                # meta loss #
                #############
                meta_loss = []
                # for each particle
                for p_idx in range(self.num_particles):
                    # compute follower and leader distance
                    p_dist_list = []
                    for name in WW_leader[p_idx].keys():
                        if 'log' in name:
                            continue
                        p_dist = tf.square(WW_follow[p_idx][name] - tf.stop_gradient(WW_leader[p_idx][name]))
                        p_dist = tf.reduce_sum(p_dist)
                        p_dist_list.append(p_dist)
                    meta_loss.append(tf.reduce_sum(p_dist_list))

                # mean over particles
                meta_loss = tf.reduce_sum(meta_loss)

                return [step_follow_weight_lprior,
                        step_follow_gamma_lprior,
                        step_follow_lambda_lprior,
                        step_follow_train_llik,
                        step_follow_valid_llik,
                        step_follow_train_loss,
                        step_follow_valid_loss,
                        step_follow_train_pred,
                        step_follow_valid_pred,
                        step_follow_weight_var,
                        step_follow_data_var,
                        step_follow_lpost,
                        step_follow_kernel_h,
                        step_leader_weight_lprior,
                        step_leader_gamma_lprior,
                        step_leader_lambda_lprior,
                        step_leader_train_llik,
                        step_leader_valid_llik,
                        step_leader_train_loss,
                        step_leader_valid_loss,
                        step_leader_train_pred,
                        step_leader_valid_pred,
                        step_leader_weight_var,
                        step_leader_data_var,
                        step_leader_lpost,
                        step_leader_kernel_h,
                        meta_loss]

            # set output type
            out_dtype = [[tf.float32] * (max_follow_step + 1),
                         [tf.float32] * (max_follow_step + 1),
                         [tf.float32] * (max_follow_step + 1),
                         [tf.float32] * (max_follow_step + 1),
                         [tf.float32] * (max_follow_step + 1),
                         [tf.float32] * (max_follow_step + 1),
                         [tf.float32] * (max_follow_step + 1),
                         [tf.float32] * (max_follow_step + 1),
                         [tf.float32] * (max_follow_step + 1),
                         [tf.float32] * (max_follow_step + 1),
                         [tf.float32] * (max_follow_step + 1),
                         [tf.float32] * max_follow_step,
                         [tf.float32] * max_follow_step,
                         [tf.float32] * (FLAGS.leader_step + 1),
                         [tf.float32] * (FLAGS.leader_step + 1),
                         [tf.float32] * (FLAGS.leader_step + 1),
                         [tf.float32] * (FLAGS.leader_step + 1),
                         [tf.float32] * (FLAGS.leader_step + 1),
                         [tf.float32] * (FLAGS.leader_step + 1),
                         [tf.float32] * (FLAGS.leader_step + 1),
                         [tf.float32] * (FLAGS.leader_step + 1),
                         [tf.float32] * (FLAGS.leader_step + 1),
                         [tf.float32] * (FLAGS.leader_step + 1),
                         [tf.float32] * (FLAGS.leader_step + 1),
                         [tf.float32] * FLAGS.leader_step,
                         [tf.float32] * FLAGS.leader_step,
                         tf.float32]

            # compute over tasks
            result = tf.map_fn(fast_learn_one_task,
                               elems=[self.follow_x, self.leader_x, self.valid_x,
                                      self.follow_y, self.leader_y, self.valid_y],
                               dtype=out_dtype,
                               parallel_iterations=FLAGS.num_tasks)

            # unroll result
            [full_step_follow_weight_lprior,
             full_step_follow_gamma_lprior,
             full_step_follow_lambda_lprior,
             full_step_follow_train_llik,
             full_step_follow_valid_llik,
             full_step_follow_train_loss,
             full_step_follow_valid_loss,
             full_step_follow_train_pred,
             full_step_follow_valid_pred,
             full_step_follow_weight_var,
             full_step_follow_data_var,
             full_step_follow_lpost,
             full_step_follow_kernel_h,
             full_step_leader_weight_lprior,
             full_step_leader_gamma_lprior,
             full_step_leader_lambda_lprior,
             full_step_leader_train_llik,
             full_step_leader_valid_llik,
             full_step_leader_train_loss,
             full_step_leader_valid_loss,
             full_step_leader_train_pred,
             full_step_leader_valid_pred,
             full_step_leader_weight_var,
             full_step_leader_data_var,
             full_step_leader_lpost,
             full_step_leader_kernel_h,
             full_meta_loss] = result

            # for training
            if is_training:
                # summarize results
                self.total_follow_weight_lprior = [tf.reduce_mean(full_step_follow_weight_lprior[j])
                                                   for j in range(FLAGS.follow_step + 1)]
                self.total_follow_gamma_lprior = [tf.reduce_mean(full_step_follow_gamma_lprior[j])
                                                  for j in range(FLAGS.follow_step + 1)]
                self.total_follow_lambda_lprior = [tf.reduce_mean(full_step_follow_lambda_lprior[j])
                                                   for j in range(FLAGS.follow_step + 1)]
                self.total_follow_train_llik = [tf.reduce_mean(full_step_follow_train_llik[j])
                                                for j in range(FLAGS.follow_step + 1)]
                self.total_follow_valid_llik = [tf.reduce_mean(full_step_follow_valid_llik[j])
                                                for j in range(FLAGS.follow_step + 1)]
                self.total_follow_train_loss = [tf.reduce_mean(full_step_follow_train_loss[j])
                                                for j in range(FLAGS.follow_step + 1)]
                self.total_follow_valid_loss = [tf.reduce_mean(full_step_follow_valid_loss[j])
                                                for j in range(FLAGS.follow_step + 1)]
                self.total_follow_weight_var = [tf.reduce_mean(full_step_follow_weight_var[j])
                                                for j in range(FLAGS.follow_step + 1)]
                self.total_follow_data_var = [tf.reduce_mean(full_step_follow_data_var[j])
                                              for j in range(FLAGS.follow_step + 1)]
                self.total_follow_lpost = [tf.reduce_mean(full_step_follow_lpost[j])
                                           for j in range(FLAGS.follow_step)]
                self.total_follow_kernel_h = [tf.reduce_mean(full_step_follow_kernel_h[j])
                                              for j in range(FLAGS.follow_step)]
                self.total_leader_weight_lprior = [tf.reduce_mean(full_step_leader_weight_lprior[j])
                                                   for j in range(FLAGS.leader_step + 1)]
                self.total_leader_gamma_lprior = [tf.reduce_mean(full_step_leader_gamma_lprior[j])
                                                  for j in range(FLAGS.leader_step + 1)]
                self.total_leader_lambda_lprior = [tf.reduce_mean(full_step_leader_lambda_lprior[j])
                                                   for j in range(FLAGS.leader_step + 1)]
                self.total_leader_train_llik = [tf.reduce_mean(full_step_leader_train_llik[j])
                                                for j in range(FLAGS.leader_step + 1)]
                self.total_leader_valid_llik = [tf.reduce_mean(full_step_leader_valid_llik[j])
                                                for j in range(FLAGS.leader_step + 1)]
                self.total_leader_train_loss = [tf.reduce_mean(full_step_leader_train_loss[j])
                                                for j in range(FLAGS.leader_step + 1)]
                self.total_leader_valid_loss = [tf.reduce_mean(full_step_leader_valid_loss[j])
                                                for j in range(FLAGS.leader_step + 1)]
                self.total_leader_weight_var = [tf.reduce_mean(full_step_leader_weight_var[j])
                                                for j in range(FLAGS.leader_step + 1)]
                self.total_leader_data_var = [tf.reduce_mean(full_step_leader_data_var[j])
                                              for j in range(FLAGS.leader_step + 1)]
                self.total_leader_lpost = [tf.reduce_mean(full_step_leader_lpost[j])
                                           for j in range(FLAGS.leader_step)]
                self.total_leader_kernel_h = [tf.reduce_mean(full_step_leader_kernel_h[j])
                                              for j in range(FLAGS.leader_step)]
                self.total_meta_loss = tf.reduce_mean(full_meta_loss)

                # prediction
                self.total_train_z_list = full_step_follow_train_pred
                self.total_valid_z_list = full_step_follow_valid_pred

                ###############
                # meta update #
                ###############
                # get param
                update_params_list = []
                update_params_name = []
                for p_idx in range(self.num_particles):
                    for name in self.W_network_particles[0].keys():
                        update_params_name.append([p_idx, name])
                        update_params_list.append(self.W_network_particles[p_idx][name])

                # set optimizer
                optimizer = tf.train.AdamOptimizer(learning_rate=self.meta_lr)

                # compute gradient
                gv_list = optimizer.compute_gradients(loss=self.total_meta_loss,
                                                      var_list=update_params_list)

                # gradient clipping
                if FLAGS.out_grad_clip > 0:
                    gv_list = [(tf.clip_by_value(grad, -FLAGS.out_grad_clip, FLAGS.out_grad_clip), var)
                               for grad, var in gv_list]

                # apply optimizer
                self.metatrain_op = optimizer.apply_gradients(gv_list)
            else:
                # summarize results
                self.eval_train_llik = [tf.reduce_mean(full_step_follow_train_llik[j])
                                        for j in range(max_follow_step + 1)]
                self.eval_train_loss = [tf.reduce_mean(full_step_follow_train_loss[j])
                                        for j in range(max_follow_step + 1)]
                self.eval_valid_llik = [tf.reduce_mean(full_step_follow_valid_llik[j])
                                        for j in range(max_follow_step + 1)]
                self.eval_valid_loss = [tf.reduce_mean(full_step_follow_valid_loss[j])
                                        for j in range(max_follow_step + 1)]

                # prediction
                self.eval_train_z_list = full_step_follow_train_pred
                self.eval_valid_z_list = full_step_follow_valid_pred

        print('end of model construction')

    def kernel(self, particle_tensor, h=-1):
        euclidean_dists = tf_utils.pdist(particle_tensor)
        pairwise_dists = tf_utils.squareform(euclidean_dists) ** 2

        if h == -1:
            if FLAGS.kernel == 'org':
                mean_dist = tf_utils.median(pairwise_dists)  # tf.reduce_mean(euclidean_dists) ** 2
                h = mean_dist / math.log(self.num_particles)
                h = tf.stop_gradient(h)
            elif FLAGS.kernel == 'med':
                mean_dist = tf_utils.median(euclidean_dists) ** 2
                h = mean_dist / math.log(self.num_particles)
                h = tf.stop_gradient(h)
            else:
                mean_dist = tf.reduce_mean(euclidean_dists) ** 2
                h = mean_dist / math.log(self.num_particles)

        kernel_matrix = tf.exp(-pairwise_dists / h)
        kernel_sum = tf.math.reduce_sum(kernel_matrix, axis=1, keepdims=True)
        grad_kernel = -tf.matmul(kernel_matrix, particle_tensor)
        grad_kernel += particle_tensor * kernel_sum
        grad_kernel /= h
        return kernel_matrix, grad_kernel, h

    def diclist2tensor(self, WW):
        return tf.stack([self.bnn_list[m].list2vec(WW[m]) for m in range(self.num_particles)])

    def tensor2diclist(self, tensor):
        return [self.bnn_list[m].vec2list(tensor[m]) for m in range(self.num_particles)]
