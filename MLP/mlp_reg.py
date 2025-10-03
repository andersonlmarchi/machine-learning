from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

moradias = pd.read_csv('moradias.csv')

print(moradias.head())

x = moradias.drop(columns=['rental_id', 'rent', 'neighborhood', 'borough'])
y = moradias['rent']

scaler = MinMaxScaler()
x_minmax = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_minmax, y, test_size=0.3, random_state=42)

mlp = Sequential()
mlp.add(Dense(28, input_shape=(14,), activation='relu'))
mlp.add(Dense(10, activation='relu'))
mlp.add(Dense(1))

mlp.compile(loss='mse', optimizer='adam', metrics=['mae'])

mlp.fit(x_train, y_train, epochs=150, verbose=0, batch_size=10)

y_previsto = mlp.predict(x_test)
# plt.scatter(y_test, y_previsto, color='blue', s=50, alpha=0.5)
# plt.xlabel('Aluguel Real')
# plt.ylabel('Aluguel Previsto')
# plt.title('Alugueis Reais vs Alugueis Previsto')

# plt.show()

_,mae = mlp.evaluate(x_test, y_test, verbose=0)
print('MAE: %.2f' % mae)

rmse = np.sqrt(mean_squared_error(y_test, y_previsto))
print('RMSE: %.2f' % rmse)

mape = mean_absolute_percentage_error(y_test, y_previsto)
print(f"MAPE: {mape:.2%}")



