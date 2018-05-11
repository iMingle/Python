"""tensorflow test

"""
import tensorflow as tf
import matplotlib.pyplot as plt  # 数据集可视化
import numpy as np  # 低级数字 Python 库
import pandas as pd  # 较高级别的数字 Python 库

c = tf.constant('Hello, world!')

with tf.Session() as sess:
    print(sess.run(c))

x = tf.constant([5.2])
y = tf.Variable([4])
y.assign([5])
with tf.Session() as sess:
    initialization = tf.global_variables_initializer()
    sess.run(initialization)
    print(x.eval(), y.eval())

# Create a graph.
g = tf.Graph()

# Establish the graph as the "default" graph.
with g.as_default():
    # Assemble a graph consisting of the following three operations:
    #   * Two tf.constant operations to create the operands.
    #   * One tf.add operation to add the two operands.
    x = tf.constant(8, name="x_const")
    y = tf.constant(5, name="y_const")
    sum = tf.add(x, y, name="x_y_sum")

    # Now create a session.
    # The session will run the default graph.
    with tf.Session() as sess:
        print(sum.eval())

# 矢量加法
with tf.Graph().as_default():
    # Create a six-element vector (1-D tensor).
    primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)

    # Create another six-element vector. Each element in the vector will be
    # initialized to 1. The first argument is the shape of the tensor (more
    # on shapes below).
    ones = tf.ones([6], dtype=tf.int32)

    # Add the two vectors. The resulting tensor is a six-element vector.
    just_beyond_primes = tf.add(primes, ones)

    # Create a session to run the default graph.
    with tf.Session() as sess:
        print(just_beyond_primes.eval())

# 张量形状
with tf.Graph().as_default():
    # A scalar (0-D tensor).
    scalar = tf.zeros([])

    # A vector with 3 elements.
    vector = tf.zeros([3])

    # A matrix with 2 rows and 3 columns.
    matrix = tf.zeros([2, 3])

    with tf.Session() as sess:
        print('scalar has shape', scalar.get_shape(), 'and value:\n', scalar.eval())
        print('vector has shape', vector.get_shape(), 'and value:\n', vector.eval())
        print('matrix has shape', matrix.get_shape(), 'and value:\n', matrix.eval())

# 广播机制
with tf.Graph().as_default():
    # Create a six-element vector (1-D tensor).
    primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)

    # Create a constant scalar with value 1.
    ones = tf.constant(1, dtype=tf.int32)

    # Add the two tensors. The resulting tensor is a six-element vector.
    just_beyond_primes = tf.add(primes, ones)

    with tf.Session() as sess:
        print(just_beyond_primes.eval())

# 矩阵乘法
with tf.Graph().as_default():
    # Create a matrix (2-d tensor) with 3 rows and 4 columns.
    x = tf.constant([[5, 2, 4, 3], [5, 1, 6, -2], [-1, 3, -1, -2]], dtype=tf.int32)

    # Create a matrix with 4 rows and 2 columns.
    y = tf.constant([[2, 2], [3, 5], [4, 5], [1, 6]], dtype=tf.int32)

    # Multiply `x` by `y`.
    # The resulting matrix will have 3 rows and 2 columns.
    matrix_multiply_result = tf.matmul(x, y)

    with tf.Session() as sess:
        print(matrix_multiply_result.eval())

# 张量变形
with tf.Graph().as_default():
    # Create an 8x2 matrix (2-D tensor).
    matrix = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]], dtype=tf.int32)

    # Reshape the 8x2 matrix into a 2x8 matrix.
    reshaped_2x8_matrix = tf.reshape(matrix, [2, 8])

    # Reshape the 8x2 matrix into a 4x4 matrix
    reshaped_4x4_matrix = tf.reshape(matrix, [4, 4])

    with tf.Session() as sess:
        print("Original matrix (8x2):")
        print(matrix.eval())
        print("Reshaped matrix (2x8):")
        print(reshaped_2x8_matrix.eval())
        print("Reshaped matrix (4x4):")
        print(reshaped_4x4_matrix.eval())

with tf.Graph().as_default():
    # Create an 8x2 matrix (2-D tensor).
    matrix = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]], dtype=tf.int32)

    # Reshape the 8x2 matrix into a 3-D 2x2x4 tensor.
    reshaped_2x2x4_tensor = tf.reshape(matrix, [2, 2, 4])

    # Reshape the 8x2 matrix into a 1-D 16-element tensor.
    one_dimensional_vector = tf.reshape(matrix, [16])

    with tf.Session() as sess:
        print("Original matrix (8x2):")
        print(matrix.eval())
        print("Reshaped 3-D tensor (2x2x4):")
        print(reshaped_2x2x4_tensor.eval())
        print("1-D vector:")
        print(one_dimensional_vector.eval())

# 变量、初始化和赋值
g = tf.Graph()
with g.as_default():
    # Create a variable with the initial value 3.
    v = tf.Variable([3])

    # Create a variable of shape [1], with a random initial value,
    # sampled from a normal distribution with mean 1 and standard deviation 0.35.
    w = tf.Variable(tf.random_normal([1], mean=1.0, stddev=0.35))

with g.as_default():
    with tf.Session() as sess:
        try:
            v.eval()
        except tf.errors.FailedPreconditionError as e:
            print("Caught expected error: ", e)

with g.as_default():
    with tf.Session() as sess:
        initialization = tf.global_variables_initializer()
        sess.run(initialization)
        # Now, variables can be accessed normally, and have values assigned to them.
        print(v.eval())
        print(w.eval())

with g.as_default():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # These three prints will print the same value.
        print(w.eval())
        print(w.eval())
        print(w.eval())

with g.as_default():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # This should print the variable's initial value.
        print(v.eval())

        assignment = tf.assign(v, [7])
        # The variable has not been changed yet!
        print(v.eval())

        # Execute the assignment op.
        sess.run(assignment)
        # Now the variable is updated.
        print(v.eval())
