# import cv2 as cv
# import pickle
# import gzip
import numpy as np
import mnist_loader
from scipy import signal


class Cnn(object):

    def __init__(self):
        # Network structure
        self.conv1 = ConvLayer((27, 27), np.random.randn(20, 1, 3, 3))
        self.mp1 = MaxPoolingLayer(3)
        self.fc1 = FullyConnected(20 * 9 * 9, 100)
        self.fc2 = FullyConnected(100, 10)
        self.sm = SoftMax()

    # input_data size is 27x27
    def feed_forward(self, input_data):
        temp = self.conv1.feed_forward(input_data)
        temp, _ = self.mp1.feed_forward(temp)
        temp = np.reshape(temp, (20 * 9 * 9, 1))
        temp = self.fc1.feed_forward(temp)
        temp = self.fc2.feed_forward(temp)
        temp = self.sm.feed_forward(temp)
        return temp

    # input data will be like [[image(27x27), result(0~9)], [image(27x27), result(0~9)], ....]
    def evaluate(self, input_data):
        test_results = []
        for index in range(np.size(input_data, 0)):
            x, y = input_data[index]

            test_results.append((np.argmax(self.feed_forward([x])), y))
            # if index % 10 == 0:
            #     print('{} sample accomplished'.format(index))
        return sum(int(x == y) for (x, y) in test_results)

    def back_prop(self, x, y):
        result1 = self.conv1.feed_forward(np.array([x]))
        result2, mpp1 = self.mp1.feed_forward(result1, record=True)
        result3 = np.reshape(result2, (20 * 9 * 9, 1))
        result4 = self.fc1.feed_forward(result3)
        result5 = self.fc2.feed_forward(result4)
        result6 = self.sm.feed_forward(result5)

        # compute the cost
        delta1 = self.sm.cost(result6, y)

        # compute the delta of different layer
        delta2 = self.sm.last_layer_delta(delta1)
        delta3 = self.fc2.last_layer_delta(delta2, result4)
        delta4 = self.fc1.last_layer_delta(delta3, result3)
        delta5 = np.reshape(delta4, [20, 9, 9])
        delta6 = self.mp1.last_layer_delta(delta5, mpp1)

        nabla_w1, nabla_b1 = self.fc2.back_prop(delta2, result4)
        nabla_w2, nabla_b2 = self.fc1.back_prop(delta3, result3)
        nabla_w3, nabla_b3 = self.conv1.back_prop(delta6, np.array([x]))

        return (nabla_w1, nabla_b1), (nabla_w2, nabla_b2), (nabla_w3, nabla_b3)

    def update_mini_batch(self, mini_batch, eta):

        index = 0

        for x, y in mini_batch:
            if index == 0:
                (nabla_w1, nabla_b1), (nabla_w2, nabla_b2), (nabla_w3, nabla_b3) = self.back_prop(x, y)
                index += 1
            else:
                (temp_w1, temp_b1), (temp_w2, temp_b2), (temp_w3, temp_b3) = self.back_prop(x, y)
                nabla_w1 = np.add(nabla_w1, temp_w1)
                nabla_b1 = np.add(nabla_b1, temp_b1)
                nabla_w2 = np.add(nabla_w2, temp_w2)
                nabla_b2 = np.add(nabla_b2, temp_b2)
                nabla_w3 = np.add(nabla_w3, temp_w3)
                nabla_b3 = np.add(nabla_b3, temp_b3)
                index += 1

        self.fc2.weights = np.array([w - (eta / len(mini_batch)) * nw for w, nw in zip(self.fc2.weights, nabla_w1)])
        self.fc2.biases = np.array([b - (eta / len(mini_batch)) * nb for b, nb in zip(self.fc2.biases, nabla_b1)])

        self.fc1.weights = np.array([w - (eta / len(mini_batch)) * nw for w, nw in zip(self.fc1.weights, nabla_w2)])
        self.fc1.biases = np.array([b - (eta / len(mini_batch)) * nb for b, nb in zip(self.fc1.biases, nabla_b2)])

        self.conv1.template = \
            np.array([w - (eta / len(mini_batch)) * nw for w, nw in zip(self.conv1.template, nabla_w3)])
        self.conv1.biases = np.array([b - (eta / len(mini_batch)) * nb for b, nb in zip(self.conv1.biases, nabla_b3)])

    def sgd(self, tr_data, epochs, mini_batch_size, eta, te_data=None):

        n = len(tr_data)

        for j in range(epochs):
            np.random.shuffle(tr_data)
            # randomly sample training data

            # prepare all the data
            mini_batches = [tr_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if te_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(te_data), np.size(te_data, 0)))
            else:
                print("Epoch {0} complete".format(j))


class FullyConnected(object):

    def __init__(self, x, y):
        self.weights = np.random.randn(y, x)
        self.biases = np.zeros((y, 1), dtype=float)

        # when it is feeding forward, all sample is computed
        # input will be like nx1 matrix

    def feed_forward(self, layer_input):

        return self.relu(np.matmul(self.weights, layer_input) + self.biases)

    # compute the delta of last layer
    # delta will be like len(y)x1. output will be like len(x)x1
    def last_layer_delta(self, delta, layer_input):
        return np.matmul(np.transpose(self.weights), delta) * self.relu_diff(layer_input)

    # mini_batch will be like [(delta1,layer_input1), (delta2,layer_input2), (delta3,layer_input3)....]
    # x is the input of this layer and y is the output
    def update_wb(self, mini_batch, eta):
        nabla_w = np.zeros(self.weights.shape)
        nabla_b = np.zeros(self.biases.shape)

        for i in range(len(mini_batch)):
            delta_nabla_w, delta_nabla_b = self.back_prop(mini_batch[i][0], mini_batch[i][1])
            nabla_w = [nabla_w + delta_nabla_w]
            nabla_b = [nabla_b + delta_nabla_b]

        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    # update the weights and biases
    @staticmethod
    def back_prop(delta, layer_input):
        return np.dot(delta, layer_input.transpose()), delta

    @staticmethod
    def relu(z):
        result = np.zeros(shape=[len(z), 1])
        for index in range(len(z)):
            result[index] = max(z[index], 0)
        return result

    @staticmethod
    def relu_diff(z):
        result = np.zeros(shape=[len(z), 1])
        for index in range(len(z)):
            result[index] = (z[index] > 0)
        return result

    @staticmethod
    def cost(output, y):
        return output - y


class ConvLayer(object):

    # x will be like n frame of image
    def __init__(self, im_size, template_input, stride=1):

        # every template has a biases, n frame of x share the same biases
        self.template = template_input
        self.biases = np.zeros(shape=[np.size(self.template, 0), im_size[0], im_size[1]])
        self.stride = stride

    # if input is like n frame of image and template has m tunnel, output will be like nxm frames of image
    # it is worthy to mention that first m frames of output correspond to the first frame of in input
    def feed_forward(self, layer_input):

        return self.relu(self.convolve(layer_input, self.template) + self.biases)

    # compute the delta of last layer
    # delta will be like n_tunnel x y.shape() . output will be like n_tunnel x x.shape().
    # notice that y is multi-layer
    def last_layer_delta(self, delta, layer_input):
        result = np.zeros(np.shape(layer_input))
        for layer in range(np.size(self.template, 1)):
            result[layer] = np.sum(signal.convolve((delta, self.template[:, layer, :, :]))) * self.relu_diff(layer_input)
        return result

    def back_prop(self, delta, layer_input):
        # compute nabla_b
        nabla_b = delta

        # compute nabla_w
        nabla_w = np.zeros(shape=np.shape(self.template))
        # num of bits need to expand
        n_expand = int(np.floor(np.size(self.template, 2) / 2))
        n_start = int(np.floor(np.size(delta, 1) / 2))
        im_expended = np.pad(layer_input, ((0, 0), (n_expand, n_expand), (n_expand, n_expand)), 'constant',
                             constant_values=0)

        for n_tunnel in range(np.size(self.template, 0)):
            for n_layer in range(np.size(self.template, 1)):
                for index_r, anchor_r in enumerate(range(n_start, n_start + np.size(self.template, 2), 1)):
                    for index_c, anchor_c in enumerate(range(n_start, n_start + np.size(self.template, 3), 1)):
                        windows = im_expended[n_layer, anchor_r - n_start:anchor_r + n_start + 1,
                                  anchor_c - n_start:anchor_c + n_start + 1]

                        nabla_w[n_tunnel, n_layer, index_r, index_c] = sum(sum(windows * delta[n_tunnel]))
        return nabla_w, nabla_b

    # mini_batch will be like [(delta1,layer_input1), (delta2,layer_input2), (delta3,layer_input3)....]
    # x is the input of this layer and y is the output
    def update_wb(self, mini_batch, eta):
        nabla_w = np.zeros(shape=self.template)
        nabla_b = np.zeros(shape=self.biases)

        for i in range(len(mini_batch)):
            delta_nabla_w, delta_nabla_b = self.back_prop(mini_batch[i][0], mini_batch[i][1])
            nabla_w = [nabla_w + delta_nabla_w]
            nabla_b = [nabla_b + delta_nabla_b]

        self.template = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.template, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    # template will be like n x n_tunnel_input x template_height x template_width
    # n_tunnel_input is the number of tunnel of input image layer
    # im must be 3-dimensional, even though 1 picture
    @staticmethod
    def convolve(im, template_input):
        template_size = np.size(template_input, 0)
        result = np.zeros([template_size, np.size(template_input, 2), np.size(template_input, 3)])
        for n in range(template_size):
            result = signal.convolve(im, np.flip(template_input[n], (1, 2)), mode='same')

        return result

    def relu(self, z):
        result = np.zeros(shape=np.shape(z))

        for n_tunnel in range(np.size(z, 0)):
            for index_row in range(np.size(z, 1)):
                for index_col in range(np.size(z, 2)):
                    result[n_tunnel, index_row, index_col] = max(z[n_tunnel, index_row, index_col], 0)

        return result

    @staticmethod
    def relu_diff(z):
        result = np.zeros(np.shape(z))

        for n_tunnel in range(np.size(z, 0)):
            for index_row in range(np.size(z, 1)):
                for index_col in range(np.size(z, 2)):
                    result[n_tunnel, index_row, index_col] = z[n_tunnel, index_row, index_col] > 0

        return result


class MaxPoolingLayer(object):
    def __init__(self, stride):
        self.stride = stride

    # the size of input layer must be divisible by stride
    def feed_forward(self, layer_input, record=False):
        output = np.zeros(shape=[np.size(layer_input, 0), np.size(layer_input, 1) // self.stride,
                                 np.size(layer_input, 2) // self.stride])

        if record:
            max_record = np.zeros(np.shape(output))
        else:
            max_record = None

        for index_tunnel in range(np.size(output, 0)):
            for index_row in range(np.size(output, 1)):
                for index_col in range(np.size(output, 2)):
                    select_area = layer_input[index_tunnel,
                                  index_row * self.stride: (index_row + 1) * self.stride,
                                  index_col * self.stride: (index_col + 1) * self.stride]

                    output[index_tunnel, index_row, index_col] = np.max(select_area)
                    if record:
                        max_record[index_tunnel, index_row, index_col] = np.argmax(select_area)
        return output, max_record

    def last_layer_delta(self, delta, max_record):
        delta_last_layer = np.zeros(shape=[np.size(delta, 0), np.size(delta, 1) * self.stride,
                                           np.size(delta, 2) * self.stride])

        for index_tunnel in range(np.size(delta, 0)):
            for index_row in range(np.size(delta, 1)):
                for index_col in range(np.size(delta, 2)):
                    temp_row = int(max_record[index_tunnel, index_row, index_col] // self.stride)
                    temp_col = int(max_record[index_tunnel, index_row, index_col] % self.stride)

                    delta_last_layer[
                        index_tunnel, index_row * self.stride + temp_row, index_col * self.stride + temp_col] = delta[
                        index_tunnel, index_row, index_col]

        return delta_last_layer


class SoftMax(object):
    def __init__(self):
        pass

    @staticmethod
    def feed_forward(layer_input):
        temp = sum(layer_input)
        if temp < 1e-6:
            return np.zeros(np.shape(layer_input))
        else:
            return layer_input / temp

    @staticmethod
    def last_layer_delta(delta):
        return delta  # * sum(np.exp(layer_input))

    @staticmethod
    def cost(output, y):
        return output - y


class AveragePoolingLayer(object):
    def __init__(self, stride):
        self.stride = stride

    # the size of input layer must be divisible by stride
    def feed_forward(self, layer_input):
        output = np.zeros(shape=[np.size(layer_input, 0), np.size(layer_input, 1) / self.stride,
                                 np.size(layer_input, 2) / self.stride])

        for index_tunnel in range(np.size(output, 0)):
            for index_row in range(np.size(output, 1)):
                for index_col in range(np.size(output, 2)):
                    select_area = layer_input[index_tunnel,
                                  index_row * self.stride: (index_row + 1) * self.stride,
                                  index_col * self.stride: (index_col + 1) * self.stride]

                    output[index_tunnel, index_row, index_col] = np.mean(select_area)

        return output

    def last_layer_delta(self, delta):
        delta_last_layer = np.zeros(shape=[np.size(delta, 0), np.size(delta, 1) * self.stride,
                                           np.size(delta, 2) * self.stride])

        for index_tunnel in range(np.size(delta, 0)):
            for index_row in range(np.size(delta, 1)):
                for index_col in range(np.size(delta, 2)):

                    for row_stride in range(self.stride):
                        for col_stride in range(self.stride):
                            delta_last_layer[
                                index_tunnel, index_row * self.stride + row_stride, index_col * self.stride + col_stride] = \
                                delta[index_tunnel, index_row, index_col] / (self.stride ** 2)


if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    my_cnn = Cnn()

    # trimmed_training_data = test_data[0:100]
    # temp_test_data = [(x[0], mnist_loader.vectorized_result(x[1])) for x in trimmed_training_data]

    my_cnn.sgd(training_data[0:6000], 10, 1000, 0.01, test_data[0:100])

    a = my_cnn.evaluate(test_data[0:100])

    c = 1
