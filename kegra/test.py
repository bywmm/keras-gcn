import tensorflow as tf

x = tf.keras.Input(shape=(4,), sparse=True)
y = tf.keras.layers.Dense(4)(x)
model = tf.keras.Model(x, y)

sparse_data = tf.sparse.SparseTensor(
    indices = [(0,0),(0,1),(0,2),
               (4,3),(5,0),(5,1)],
    values = [1,1,1,1,1,1],
    dense_shape = (6,4)
)

model(sparse_data)

print(model.predict(sparse_data))