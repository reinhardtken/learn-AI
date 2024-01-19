import tensorflow as tf



import timeit

# 使用cpu运算

def cpu_run():
    with tf.device('/cpu:0'):
        cpu_a = tf.random.normal([100000, 1000])
        cpu_b = tf.random.normal([1000, 1000])
        c = tf.matmul(cpu_a, cpu_b)
        return c

# 使用gpu运算
def gpu_run():
    with tf.device('/gpu:1'):
        gpu_a = tf.random.normal([100000, 1000])
        gpu_b = tf.random.normal([1000, 1000])
        c = tf.matmul(gpu_a, gpu_b)
        return c

# 默认运算
def pu_run():
    gpu_a = tf.random.normal([100000, 1000])
    gpu_b = tf.random.normal([1000, 1000])
    c = tf.matmul(gpu_a, gpu_b)
    return c

cpu_time = timeit.timeit(cpu_run, number=10)
gpu_time = timeit.timeit(gpu_run, number=10)
pu_time=timeit.timeit(pu_run, number=10)
# print("cpu:", cpu_time, " gpu:", gpu_time)
print("cpu:", cpu_time, " gpu:", gpu_time," pu:", pu_time)