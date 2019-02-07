import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

import numpy as np


class Model():
    def __init__(self, args, training=True):
        self.args = args
        if not training:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        elif args.model == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cells = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)
            if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(cell,
                                          input_keep_prob=args.input_keep_prob,
                                          output_keep_prob=args.output_keep_prob)
            # creating array of cells
            cells.append(cell) 

        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)
        
        #initialising input data
        self.input_data = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])
        
        #initializing target 
        self.targets = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])

        #initializing the state of MultiRNN cell to 0
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w",
                                        [args.rnn_size, args.vocab_size], name="Weights")
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size],name="Bias")

        embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size],name="embedding")
        #dividing input into equal parts ; by default the partition strategy is mod i.e round robin like
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # dropout beta testing: double check which one should affect next line
        # dropout we randomly make some input 0 so that activatoin doesnt depend on few features
        # here we have kept keep prob 1 by default. That means we are not actullaly dropping anything.
        # dropout is done to reqularize when we fear some layers mght overfit; l2 reqularization works just fine.
        # people use it only when the model overfits. Generally keep prob kept close to 1 
        if training and args.output_keep_prob:
            inputs = tf.nn.dropout(inputs, args.output_keep_prob)
        #splits the embedding which is like array of all inputs , splits it into individual arrays to be given as inputs
        inputs = tf.split(inputs, args.seq_length, 1)
        #tf.squeeze squeezes the first argument to make into dimension of second. Also if some dimension is 1 it removes that dimension
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # actual loop that takes output and feeds it as input
        def loop(prev, _,name="loop function of legacy_seq2seq"):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            #prevents argmax's error to be propogated through backpropogation
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            #embedding lookup splits embedding into rows to be given as inputs
            return tf.nn.embedding_lookup(embedding, prev_symbol)
        
        # using lstm encoder decoder module here.
        #it outputs the output note and the hidden layer states of the last node in lstm decoder module
        #seq2seq module is prebuilt by tensorflow , the encoder part , computation happening in both rnn layer
        # the architecture of the layers and everything else is encapsulated for us by tensorflow
        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='rnnlm')
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])


        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        
        # calculates  weighted crossentropy loss by example ; because it is a multi class classification problem
        #because at times we need the output to play 2 notes if necessary
        loss = legacy_seq2seq.sequence_loss_by_example(
                [self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])])
        
        #scopes for tensorboard
        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        #cost passed to grad here so that can be passed to apply gradient function of adam optimizer
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        with tf.name_scope('optimizer'):
            #using adam as optimizer with given learning rate lr
            optimizer = tf.train.AdamOptimizer(self.lr)
            #minimizing cost here
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # instrument tensorboard
        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('loss', loss)
        tf.summary.scalar('train_loss', self.cost)

    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else:  # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret
