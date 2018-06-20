from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

from modeler import Modeler

def load_data():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def generate_models():
    model1 = Sequential()
    model1.add(Dense(64, activation='relu', input_dim=784))
    model1.add(Dense(10, activation='softmax'))
    model1.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model2 = Sequential()
    model2.add(Dense(128, activation='relu', input_dim=784))
    model2.add(Dense(10, activation='softmax'))
    model2.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model3 = Sequential()
    model3.add(Dense(256, activation='relu', input_dim=784))
    model3.add(Dense(10, activation='softmax'))
    model3.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return (model1, model2, model3)



if __name__ == "__main__":

    data = load_data()

    (model1, model2, model3)  = generate_models()

    modeler = Modeler()
    modeler.add(model1)
    modeler.add(model2)
    modeler.add(model3)
    modeler.start(data,
                  epochs=12)
    modeler.save(n_save=1)
