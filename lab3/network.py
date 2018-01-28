import numpy as np


def output_softmax(o):
    # o shape = (seq, minibatch, dim)
    res = np.zeros_like(o)
    for seq_idx in range(o.shape[0]):
        for row_idx in range(o.shape[1]):
            row = o[seq_idx][row_idx]
            sm = np.exp(row) / np.sum(np.exp(row), axis=0)
            res[seq_idx][row_idx][:] = sm
    return res


def calculate_loss(yhat, y):
    return np.sum(np.mean(-np.sum(y*np.log(yhat), axis=2), axis=1))


class RNN:
    # ...
    # Code is nested in class definition, indentation is not representative.
    # "np" stands for numpy
    def __init__(self, vocab_size, hidden_size=100, sequence_length=30, learning_rate=1e-1):
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.delta = 1e-7

        dev = 1e-2
        mu = 0.0
        self.U = np.random.normal(mu, dev, size=(self.vocab_size, self.hidden_size))  # ... input projection
        self.W = np.random.normal(mu, dev, size=(self.hidden_size, self.hidden_size))  # ... hidden-to-hidden projection
        self.b = np.zeros(shape=(1, self.hidden_size))  # ... input bias

        self.V = np.random.normal(mu, dev, size=(self.hidden_size, self.vocab_size))  # ... output projection
        self.c = np.zeros(shape=(1, self.vocab_size))  # ... output bias

        # memory of past gradients - rolling sum of squares for Adagrad
        self.memory_U, self.memory_W, self.memory_V = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(
            self.V)
        self.memory_b, self.memory_c = np.zeros_like(self.b), np.zeros_like(self.c)

    @staticmethod
    def rnn_step_forward(x, h_prev, U, W, b):
        # A single time step forward of a recurrent neural network with a
        # hyperbolic tangent nonlinearity.

        # x - input data (minibatch size x input dimension)
        # h_prev - previous hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)

        h_current = np.tanh(np.dot(h_prev, W) + np.dot(x, U) + b)
        cache = (x, h_prev, U, W, h_current)

        # return the new hidden state and a tuple of values needed for the backward step

        return h_current, cache

    @staticmethod
    def rnn_forward(x, h0, U, W, b):
        # Full unroll forward of the recurrent neural network with a
        # hyperbolic tangent nonlinearity

        # x_size = (seq, minibatch, dim)

        # x - input data for the whole time-series (minibatch size x sequence_length x input dimension)
        # h0 - initial hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)

        h_curr = h0
        h_list = []
        cache = []
        for x_t in x:
            h_curr, cache_t = RNN.rnn_step_forward(x_t, h_curr, U, W, b)
            h_list.append(h_curr)
            cache.append(cache_t)

        h = np.stack(h_list)

        # return the hidden states for the whole time series (T+1) and a tuple of values needed for the backward step

        return h, cache

    def rnn_step_backward(self, grad_next, cache):
        # A single time step backward of a recurrent neural network with a
        # hyperbolic tangent nonlinearity.

        # grad_next - upstream gradient of the loss with respect to the next hidden state and current output
        # cache - cached information from the forward pass

        x, h_prev, U, W, h_current = cache
        da_t = grad_next * (1 - h_current*h_current)
        dh_prev = np.dot(da_t, np.transpose(W))
        dU = np.dot(np.transpose(x), da_t)
        dW = np.dot(np.transpose(h_prev), da_t)
        db = da_t.mean(axis=0)

        # compute and return gradients with respect to each parameter
        # HINT: you can use the chain rule to compute the derivative of the
        # hyperbolic tangent function and use it to compute the gradient
        # with respect to the remaining parameters
        dh_prev = np.clip(dh_prev, -5, 5)
        dU = np.clip(dU, -5, 5)
        dW = np.clip(dW, -5, 5)
        db = np.clip(db, -5, 5)
        return dh_prev, dU, dW, db

    def rnn_backward(self, dh, cache):

        dU, dW, db = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.b)
        dh_prev = np.zeros_like(dh[0])
        for t in range(self.sequence_length-1, -1, -1):
            dh_prev, dU_curr, dW_curr, db_curr = self.rnn_step_backward(dh[t] + dh_prev, cache[t])
            dU += dU_curr
            dW += dW_curr
            db += db_curr

        # compute and return gradients with respect to each parameter
        # for the whole time series.
        # Why are we not computing the gradient with respect to inputs (x)?
        dU = np.clip(dU, -5, 5)
        dW = np.clip(dW, -5, 5)
        db = np.clip(db, -5, 5)
        return dU, dW, db

    @staticmethod
    def output(h, V, c):
        # res size = (seq, minibatch, dim)
        return np.dot(h, V) + c

    def output_loss_and_grads(self, h, V, c, y):
        # Calculate the loss of the network for each of the outputs

        # h - hidden states of the network for each timestep.
        #     the dimensionality of h is (batch size x sequence length x hidden size (the initial state is irrelevant for the output)
        # V - the output projection matrix of dimension hidden size x vocabulary size
        # c - the output bias of dimension vocabulary size x 1
        # y - the true class distribution - a tensor of dimension
        #     batch_size x sequence_length x vocabulary size - you need to do this conversion prior to
        #     passing the argument. A fast way to create a one-hot vector from
        #     an id could be something like the following code:

        #   y[batch_id][timestep] = np.zeros((vocabulary_size, 1))
        #   y[batch_id][timestep][batch_y[timestep]] = 1

        #     where y might be a list or a dictionary.

        # calculate the output (o) - unnormalized log probabilities of classes
        o = RNN.output(h, V, c)
        # calculate yhat - softmax of the output
        yhat = output_softmax(o)
        # calculate the cross-entropy loss
        # loss dim: seq x minibatch, redak oznacava za koji sequence gledamo gubitak a stupac za koji podatak iz batcha
        loss = calculate_loss(yhat, y)
        # calculate the derivative of the cross-entropy softmax loss with respect to the output (o)
        do = yhat - y
        # calculate the gradients with respect to the output parameters V and c
        dV = np.zeros_like(V)
        dc = np.zeros_like(c)
        for t in range(self.sequence_length-1, -1, -1):
            do_t = do[t]
            h_t = h[t]
            dV += np.dot(np.transpose(h_t), do_t)
            dc += do_t.mean(axis=0)
        # calculate the gradients with respect to the hidden layer h
        dh = np.zeros_like(h)
        for t in range(self.sequence_length-1, -1, -1):
            dh[t] = np.dot(do[t], np.transpose(V))

        dV = np.clip(dV, -5, 5)
        dc = np.clip(dc, -5, 5)
        return loss, dh, dV, dc

    # The inputs to the function are just indicative since the variables are mostly present as class properties
    def update(self, dU, dW, db, dV, dc):

        # update memory matrices
        # perform the Adagrad update of parameters

        self.memory_U += dU*dU
        self.memory_W += dW*dW
        self.memory_b += db*db
        self.memory_V += dV*dV
        self.memory_c += dc*dc
        self.U += (-self.learning_rate/(self.delta + np.sqrt(self.memory_U))) * dU
        self.W += (-self.learning_rate/(self.delta + np.sqrt(self.memory_W))) * dW
        self.b += (-self.learning_rate/(self.delta + np.sqrt(self.memory_b))) * db
        self.V += (-self.learning_rate/(self.delta + np.sqrt(self.memory_V))) * dV
        self.c += (-self.learning_rate/(self.delta + np.sqrt(self.memory_c))) * dc

    def step(self, h0, x_oh, y_oh, N):
        h, cache = RNN.rnn_forward(x_oh, h0, self.U, self.W, self.b)
        loss, dh, dV, dc = self.output_loss_and_grads(h, self.V, self.c, y_oh)
        dU, dW, db = self.rnn_backward(dh, cache)
        self.update(dU/N, dW/N, db/N, dV/N, dc/N)
        return loss, h[-1]

    def sample(self, seed_oh, n_samples):
        h0_sample = np.zeros(shape=(1, self.hidden_size))
        h, _ = RNN.rnn_forward(seed_oh, h0_sample, self.U, self.W, self.b)
        h_curr = h[-1]
        x_curr = np.asarray([seed_oh[-1]])
        y_list = []
        for i in range(n_samples-len(seed_oh)):
            h, _ = RNN.rnn_forward(x_curr, h_curr, self.U, self.W, self.b)
            h_curr = h[0]
            o = np.asarray([RNN.output(h_curr, self.V, self.c)])
            y = output_softmax(o)[0]
            encoded_y = np.argmax(y, axis=1)[0]
            x_curr = np.zeros_like(x_curr)
            x_curr[0, encoded_y] = 1
            y_list.append(encoded_y)
        return y_list


def run_language_model(seed, dataset, max_epochs, hidden_size=100, sequence_length=30, learning_rate=1e-1, sample_every=100):
    dataset.create_minibatches(100, sequence_length)
    vocab_size = len(dataset.sorted_chars)
    rnn = RNN(
        vocab_size=vocab_size, hidden_size=hidden_size, sequence_length=sequence_length, learning_rate=learning_rate)

    current_epoch = 0
    batch = 0

    h0 = np.zeros((dataset.batch_size, hidden_size))

    seed_encoded = dataset.encode(seed)
    n_samples = 300
    seed_oh = np.zeros(shape=(len(seed), vocab_size))
    for i in range(len(seed_encoded)):
        seed_oh[i][seed_encoded[i]] = 1

    while current_epoch < max_epochs:
        e, x, y = dataset.next_minibatch()

        if e:
            current_epoch += 1
            h0 = np.zeros((dataset.batch_size, hidden_size))

        # One-hot transform the x and y batches
        x_oh = np.zeros(shape=(sequence_length, dataset.batch_size, vocab_size))
        y_oh = np.zeros(shape=(sequence_length, dataset.batch_size, vocab_size))
        for mini_batch_idx in range(x.shape[0]):
            for seq_idx in range(x.shape[1]):
                x_oh[seq_idx][mini_batch_idx][x[mini_batch_idx][seq_idx]] = 1
                y_oh[seq_idx][mini_batch_idx][y[mini_batch_idx][seq_idx]] = 1

        # Run the recurrent network on the current batch
        # Since we are using windows of a short length of characters,
        # the step function should return the hidden state at the end
        # of the unroll. You should then use that hidden state as the
        # input for the next minibatch. In this way, we artificially
        # preserve context between batches.
        loss, h0 = rnn.step(h0, x_oh, y_oh, x.shape[0])

        if batch % sample_every == 0:
            encoded_l = rnn.sample(seed_oh, n_samples)
            output = dataset.decode(encoded_l)
            print('Epoch:', current_epoch, 'Batch:', batch)
            print('Loss:', loss)
            print()
            print('Sample:')
            print(''.join(output))
            print('----------------------------------------------------')
        batch += 1


if __name__ == '__main__':
    from dataset import Data
    input_file = 'data/selected_conversations.txt'
    data = Data()
    data.preprocess(input_file)
    seed = 'STRIKER:\nDon\'t you feel anything for me at all any more?\n\n'
    run_language_model(seed, data, 10)
