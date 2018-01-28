import numpy as np
import operator


class Data:
    def __init__(self):
        self.curr_batch_idx = 0

    def preprocess(self, input_file):
        with open(input_file, "r", encoding='utf-8') as f:
            data = f.read()
        pass
        # count and sort most frequent characters
        char_freq = {}
        for char in data:
            char_freq[char] = 1 if char_freq.get(char, None) is None else (char_freq[char] + 1)
        sorted_tuples = sorted(char_freq.items(), key=operator.itemgetter(1), reverse=True)
        self.sorted_chars = [t[0] for t in sorted_tuples]

        # self.sorted chars contains just the characters ordered descending by frequency
        self.char2id = dict(zip(self.sorted_chars, range(len(self.sorted_chars))))
        # reverse the mapping
        self.id2char = {k: v for v, k in self.char2id.items()}
        # convert the data to ids
        self.x = np.array(list(map(self.char2id.get, data)))

    def encode(self, sequence):
        # returns the sequence encoded as integers
        return np.array(list(map(self.char2id.get, sequence)))

    def decode(self, encoded_sequence):
        # returns the sequence decoded as letters
        return np.array(list(map(self.id2char.get, encoded_sequence)))

    def create_minibatches(self, batch_size, sequence_length):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_batches = int(len(self.x) / (self.batch_size * self.sequence_length))  # calculate the number of batches

        mini_batch_X = []
        mini_batch_Y = []
        self.batches = []
        chars_in_batch = self.batch_size * self.sequence_length
        X = []
        Y = []
        for idx in range(len(self.x)-1):
            X.append(self.x[idx])
            Y.append(self.x[idx + 1])
            if (idx + 1) % self.sequence_length == 0:
                mini_batch_X.append(X)
                mini_batch_Y.append(Y)
                X = []
                Y = []
                if (idx + 1) % chars_in_batch == 0:
                    self.batches.append((np.asarray(mini_batch_X), np.asarray(mini_batch_Y)))
                    mini_batch_X = []
                    mini_batch_Y = []

        if len(X) == 0:
            return
        self.batches.append((np.asarray(mini_batch_X), np.asarray(mini_batch_Y)))

    def next_minibatch(self):
        if self.curr_batch_idx >= len(self.batches):
            self.curr_batch_idx = 0
        new_epoch = True if self.curr_batch_idx == 0 else False
        batch_x, batch_y = self.batches[self.curr_batch_idx]
        self.curr_batch_idx += 1

        return new_epoch, batch_x, batch_y


if __name__ == '__main__':
    input_file = 'data/selected_conversations.txt'
    data = Data()
    data.preprocess(input_file)
    data.create_minibatches(batch_size=100, sequence_length=5)
    last = data.batches[-1]
    n, x, y = data.next_minibatch()
    print(data.decode([0,1,2,3]))
    pass
