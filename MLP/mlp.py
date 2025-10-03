import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0: tudo, 1: remove INFO, 2: remove WARNING, 3: remove WARNING+ERROR leves

from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns

dataset = loadtxt('diabetes-indios-pima.csv', delimiter=',')

x = dataset[:, 0:8]
y = dataset[:, 8]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

mlp = Sequential()
mlp.add(Dense(12, input_shape=(8,), activation='relu'))
# mlp.add(Dense(8, activation='relu'))
mlp.add(Dense(1, activation='sigmoid'))

mlp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

mlp.fit(x_train, y_train, epochs=150, verbose=0)

_, accuracy = mlp.evaluate(x_test, y_test, verbose=0)
print('Acurácia: %.2f' % (accuracy * 100))

predictions = mlp.predict(x_test, verbose=0)
rounded_predictions = [round(x[0]) for x in predictions]

for i in range(len(rounded_predictions)):
    print('%s => %d (esperado %d)' % (x_test[i].tolist(), rounded_predictions[i], y_test[i]))

confusion_matrix = confusion_matrix(y_test, rounded_predictions)

sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão')
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.show()
