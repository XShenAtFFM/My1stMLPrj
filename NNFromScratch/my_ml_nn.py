import my_neurone_from_scratch as my_nn_api
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np

np.random.seed(42)

def create_test_data():
    # Generate a dataset
    x, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size = 0.2,  # 20% for evaluation
        random_state = 42,  # for reproducibility
        stratify = y  # keeps class balance
    )
    return x_train.T, x_test.T, y_train.reshape(1, -1), y_test.reshape(1, -1)

def my_nn_forwards(x, parameters):
    # first dense layer mit activation relu
    a_layer1, z_layer1  = my_nn_api.my_dense_layer(x, parameters['weights_layer1'], parameters['bias_layer1'], 'relu')
    # 2nd dense layer mit activation relu
    a_layer2, z_layer2 = my_nn_api.my_dense_layer(a_layer1, parameters['weights_layer2'], parameters['bias_layer2'], 'relu')
    # output payer mit sigmoid
    y_hat, z_output = my_nn_api.my_dense_layer(a_layer2, parameters['weights_output'], parameters['bias_output'], 'sigmoid')

    catch = [a_layer1, z_layer1, a_layer2, z_layer2]
    return catch, y_hat

def my_nn_backwards(catch, my_nn_parameters, x, y_hat, y_true, optimizer = 'sgd'):
    a_layer1, z_layer1, a_layer2, z_layer2 = catch
    # backward output layer
    #dL_dA = my_nn_api.my_logistic_gradient(y_hat, y_train)
    #dA_dZ = my_nn_api.my_sigmoid_derived(y_hat)
    #dL_dZ = dL_dA * dA_dZ
    # the calculation above can be done easily
    dL_dZ_output = (y_hat - y_true)
    dW_output = np.dot(dL_dZ_output, a_layer2.T) / y_hat.shape[1]
    dB_output = np.sum(dL_dZ_output, axis = 1, keepdims = True) / y_hat.shape[1]

    # backward 2nd dense layer
    dL_dZ_layer2 = np.dot(my_nn_parameters['weights_output'].T, dL_dZ_output) * my_nn_api.my_relu_derived(z_layer2)
    dW_layer2 = np.dot(dL_dZ_layer2, a_layer1.T) / y_hat.shape[1]
    dB_layer2 = np.sum(dL_dZ_layer2, axis = 1, keepdims = True) / y_hat.shape[1]

    # backward 1st dense layer
    dL_dZ_layer1 = np.dot(my_nn_parameters['weights_layer2'].T, dL_dZ_layer2) * my_nn_api.my_relu_derived(z_layer1)
    dW_layer1 = np.dot(dL_dZ_layer1, x.T) / y_hat.shape[1]
    dB_layer1 = np.sum(dL_dZ_layer1, axis=1, keepdims=True) / y_hat.shape[1]

    weights_grads = (dW_output, dW_layer2, dW_layer1)
    bias_grads = (dB_output, dB_layer2, dB_layer1)
    parameter_optimization(weights_grads, bias_grads, my_nn_parameters, optimizer)


def parameter_optimization(weights_grads, bias_grads, my_nn_parameters, optimizer):
    layers = ('output', 'layer2', 'layer1')
    if optimizer == 'sgd':
        for i, layer in enumerate(layers):
            my_nn_parameters[f'weights_{layer}'] -= my_nn_parameters['learning_rate'] * weights_grads[i]
            my_nn_parameters[f'bias_{layer}'] -= my_nn_parameters['learning_rate'] * bias_grads[i]
    elif optimizer == 'adam':
        beta1 = my_nn_parameters['beta1']
        beta2 = my_nn_parameters['beta2']
        learning_rate = my_nn_parameters['learning_rate']
        my_nn_parameters['steps'] +=  1
        t = my_nn_parameters['steps']
        for i, layer in enumerate(layers):
            # ----- weights -----
            m_w = my_nn_parameters[f'm_weights_{layer}']
            v_w = my_nn_parameters[f'v_weights_{layer}']

            # update moments
            m_w = beta1 * m_w + (1. - beta1) * weights_grads[i]
            v_w = beta2 * v_w + (1. - beta2) * (weights_grads[i] ** 2)

            # bias corrected
            m_hat_w = m_w / (1 - beta1 ** t)
            v_hat_w = v_w / (1 - beta2 ** t)

            # update parameters
            my_nn_parameters[f'weights_{layer}'] -= learning_rate * m_hat_w / (np.sqrt(v_hat_w) + 1e-8)
            my_nn_parameters[f'm_weights_{layer}'] = m_w
            my_nn_parameters[f'v_weights_{layer}'] = v_w

            # ----- bias -----
            m_b = my_nn_parameters[f'm_bias_{layer}']
            v_b = my_nn_parameters[f'v_bias_{layer}']

            # update moments
            m_b = beta1 * m_b + (1. - beta1) * bias_grads[i]
            v_b = beta2 * v_b + (1. - beta2) * (bias_grads[i] ** 2)

            # bias corrected
            m_hat_b = m_b / (1 - beta1 ** t)
            v_hat_b = v_b / (1 - beta2 ** t)
            # update parameters
            my_nn_parameters[f'bias_{layer}'] -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + 1e-8)
            my_nn_parameters[f'm_bias_{layer}'] = m_b
            my_nn_parameters[f'v_bias_{layer}'] = v_b

    else:
        raise Exception("Only two optimization methods are supported")


def init_parameters(features):
    my_nn_parameters = {}
    my_nn_parameters['units_layer1'] = 8
    my_nn_parameters['weights_layer1'] = np.random.randn(my_nn_parameters['units_layer1'], features)
    my_nn_parameters['bias_layer1'] = np.zeros((my_nn_parameters['units_layer1'], 1))
    my_nn_parameters['m_weights_layer1'] = np.zeros_like(my_nn_parameters['weights_layer1'] )
    my_nn_parameters['v_weights_layer1'] = np.zeros_like(my_nn_parameters['weights_layer1'])
    my_nn_parameters['m_bias_layer1'] = np.zeros_like(my_nn_parameters['bias_layer1'])
    my_nn_parameters['v_bias_layer1'] = np.zeros_like(my_nn_parameters['bias_layer1'])

    # layer 2
    my_nn_parameters['units_layer2'] = 4
    my_nn_parameters['weights_layer2'] = np.random.randn(my_nn_parameters['units_layer2'], my_nn_parameters['units_layer1'])
    my_nn_parameters['bias_layer2'] = np.zeros((my_nn_parameters['units_layer2'], 1))
    my_nn_parameters['m_weights_layer2'] = np.zeros_like(my_nn_parameters['weights_layer2'] )
    my_nn_parameters['v_weights_layer2'] = np.zeros_like(my_nn_parameters['weights_layer2'])
    my_nn_parameters['m_bias_layer2'] = np.zeros_like(my_nn_parameters['bias_layer2'])
    my_nn_parameters['v_bias_layer2'] = np.zeros_like(my_nn_parameters['bias_layer2'])

    # output layer
    my_nn_parameters['units_output'] = 1
    my_nn_parameters['weights_output'] = np.random.randn(my_nn_parameters['units_output'], my_nn_parameters['units_layer2'])
    my_nn_parameters['bias_output'] = np.zeros((my_nn_parameters['units_output'], 1))
    my_nn_parameters['m_weights_output'] = np.zeros_like(my_nn_parameters['weights_output'] )
    my_nn_parameters['v_weights_output'] = np.zeros_like(my_nn_parameters['weights_output'])
    my_nn_parameters['m_bias_output'] = np.zeros_like(my_nn_parameters['bias_output'])
    my_nn_parameters['v_bias_output'] = np.zeros_like(my_nn_parameters['bias_output'])

    my_nn_parameters['learning_rate'] = 1e-2
    my_nn_parameters['beta1'] = 0.9
    my_nn_parameters['beta2'] = 0.999
    my_nn_parameters['steps'] = 0

    return my_nn_parameters

def train_my_nn(x, y_true, my_nn_parameters, optimizer = 'sgd'):
    # create x_train to mini batches
    batches_size = int(x.shape[1] / 100)
    L = []
    for batch in range(batches_size):
        x_batch = x[:, batch * 100 : (batch + 1) * 100]
        y_batch = y_true[:, batch * 100 : (batch + 1) * 100]
        catch, y_hat = my_nn_forwards(x_batch, my_nn_parameters)
        loss = my_nn_api.my_logistic_cost(y_hat, y_batch)
        L.append(loss)
        my_nn_backwards(catch, my_nn_parameters, x_batch, y_hat, y_batch, optimizer)

    return L

def my_nn_predict(x, my_nn_parameters):
    _, y_hat = my_nn_forwards(x, my_nn_parameters)
    return y_hat

def fitting_my_nn(x_train, y_train, my_nn_parameters, epochs, optimizer = 'sgd'):
    Loss_train = []
    for epoch in range(epochs):
        Loss = train_my_nn(x_train, y_train, my_nn_parameters, optimizer)
        Loss_train.append(np.mean(Loss))

    return Loss_train

def test_my_nn_step1():
    # check output dimension of dense and output layer
    # generate datasets
    data_size = 4
    x = np.random.randn(5, data_size)
    # init parameters
    my_nn_parameters = init_parameters(x.shape[0])
    # forwards propagation
    catch, y_hat = my_nn_forwards(x, my_nn_parameters)
    a_layer1, z_layer1, a_layer2, z_layer2 = catch
    # check dimension
    assert y_hat.shape == (1, data_size), "the y_hat shall be of shape (1, {data_size})"
    assert a_layer1.shape == (8, data_size), "the a_layer1 shall be of shape (8, {data_size})"
    assert a_layer2.shape == (4, data_size), "the a_layer2 shall be of shape (4, {data_size})"
    print('Test my_nn_step1 passed, all shapes are as expected')

def test_my_nn_step2():
    # check gradient descent calculation
    # generate datasets
    data_size = 10
    x = np.random.randn(2, data_size)
    y_true = np.ones((1, data_size))
    # init parameters
    my_nn_parameters = init_parameters(x.shape[0])
    my_nn_parameters['learning_rate'] = 1

    eps_t = 1e-5
    parameters_field = ('weights_layer1', 'bias_layer1', 'weights_layer2', 'bias_layer2', 'weights_output', 'bias_output')
    for parameter in parameters_field:
        for i, j in np.ndindex(my_nn_parameters[parameter].shape):
            # iterate parameters on each layer
            # original parameter value
            parameter_val_orig = my_nn_parameters[parameter][i, j]

            # central derivation approach
            # dL/dwi = L(wi + eps_t) - L(wi - eps_t) / 2 * eps_t
            my_nn_parameters[parameter][i, j] = parameter_val_orig + eps_t
            _, y_hat = my_nn_forwards(x, my_nn_parameters)
            L_plus = my_nn_api.my_logistic_cost(y_hat, y_true)

            my_nn_parameters[parameter][i, j] = parameter_val_orig - eps_t
            _, y_hat = my_nn_forwards(x, my_nn_parameters)
            L_minus = my_nn_api.my_logistic_cost(y_hat, y_true)

            grad = (L_plus - L_minus) / (2 * eps_t)

            # dL/dWi calculated by back propagation
            # the Wi_new = Wi_orig + learning_rat * dL/dWi
            # dL/dWi = Wi_new - Wi_orig, since learning_rat = 1
            my_nn_parameters[parameter][i, j] = parameter_val_orig
            catch, y_hat = my_nn_forwards(x, my_nn_parameters)
            my_nn_backwards(catch, my_nn_parameters, x, y_hat, y_true, 'sgd')
            grad_by_back_propagation = parameter_val_orig - my_nn_parameters[parameter][i, j]

            assert np.isclose(grad, grad_by_back_propagation, atol=1e-05, equal_nan=False), f"derivation {parameter} looks wrong, {grad_by_back_propagation} vs. {grad}"

    print('Test my_nn_step2 passed, the derivations and stochastic gradient descent work proper')

def test_my_nn_step3():
    # check the parameter momentum updates
    my_nn_parameters = init_parameters(2)
    my_nn_parameters['learning_rate'] = 0.01
    layers = ('output', 'layer2', 'layer1')
    target_value = (3.0, 4.0, 5)
    for steps in range(6000):
        weights_grads = [2 * (my_nn_parameters[f'weights_{layer}'] - t_v) for layer, t_v in zip(layers, target_value)]
        bias_grads = [3 * (my_nn_parameters[f'bias_{layer}'] - t_v) for layer, t_v in zip(layers, target_value[::-1])]
        #weights_grads = [2 * (my_nn_parameters['weights_output'] - 3)]
        parameter_optimization(weights_grads, bias_grads, my_nn_parameters, 'adam')

    assert (np.allclose(my_nn_parameters[f'weights_{layer}'], t_v, atol=0.01)
        for layer, t_v in zip(layers, target_value)), f'the weights parameter are not convergent to the target value'

    assert all(np.allclose(my_nn_parameters[f'bias_{layer}'], t_v, atol=0.01)
        for layer, t_v in zip(layers, target_value[::-1])), f'the bias parameter are not convergent to the target value'

    assert all(np.allclose(my_nn_parameters[f'm_weights_{layer}'], 0, atol=0.01)
        for layer in layers), f'the m_weight parameter are not convergent to 0'

    assert all(np.allclose(my_nn_parameters[f'm_bias_{layer}'], 0, atol=0.01)
               for layer in layers), f'the m_bias parameter are not convergent to 0'

    print('Test my_nn_step3 passed, the adam optimization works proper')


if __name__ == '__main__':
    test_my_nn_step1()
    test_my_nn_step2()
    test_my_nn_step3()