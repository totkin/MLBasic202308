import numpy as np
from random import randrange


def eval_numerical_gradient(f, x):
    """
  a naive implementation of numerical gradient of f at x 
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """

    fx = f(x)  # evaluate function value at original point
    grad = np.zeros(x.shape)
    h = 0.00001

    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # evaluate function at x+h
        ix = it.multi_index
        x[ix] += h  # increment by h
        fxh = f(x)  # evalute f(x + h)
        x[ix] -= h  # restore to previous value (very important!)

        # compute the partial derivative
        grad[ix] = (fxh - fx) / h  # the slope
        print(ix, grad[ix])
        it.iternext()  # step to next dimension
    return grad


def eval_numerical_gradient_array(f, x, df, h=10 ** -5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


def eval_numerical_gradient_blobs(f, inputs, output, h=1e-5):
    """
    Compute numeric gradients for a function that operates on input
    and output blobs.

    We assume that f accepts several input blobs as arguments, followed by a blob
    into which outputs will be written. For example, f might be called like this:

    f(x, w, out)

    where x and w are input Blobs, and the result of f will be written to out.

    Inputs:
    - f: function
    - inputs: tuple of input blobs
    - output: output blob
    - h: step size
    """
    numeric_diffs = []
    for input_blob in inputs:
        diff = np.zeros_like(input_blob.diffs)
        it = np.nditer(input_blob.vals, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig = input_blob.vals[idx]

            input_blob.vals[idx] = orig + h
            f(*(inputs + (output,)))
            pos = np.copy(output.vals)
            input_blob.vals[idx] = orig - h
            f(*(inputs + (output,)))
            neg = np.copy(output.vals)
            input_blob.vals[idx] = orig

            diff[idx] = np.sum((pos - neg) * output.diffs) / (2.0 * h)

            it.iternext()
        numeric_diffs.append(diff)
    return numeric_diffs


def eval_numerical_gradient_net(net, inputs, output, h=1e-5):
    return eval_numerical_gradient_blobs(lambda *args: net.forward(),
                                         inputs, output, h=h)


def grad_check_sparse(func, x, analytic_grad,
                      num_checks: int = 10,
                      h: float = 10 ** -5,
                      verbose: bool = True):
    """
  sample a few random elements and only return numerical
  in this dimensions.
  """
    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        x[ix] += h  # increment by h
        fxph = func(x)  # evaluate func(x + h)
        x[ix] -= 2 * h  # increment by h
        fxmh = func(x)  # evaluate func(x - h)
        x[ix] += h  # reset

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))

        if verbose:
            strMessage = f'numerical: {grad_numerical,} analytic: {grad_analytic}, relative error: {rel_error}'
            print(strMessage)
