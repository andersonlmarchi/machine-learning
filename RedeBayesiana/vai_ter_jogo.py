import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

dados = pd.read_csv('dados_jogo.csv')
x = dados[['Tempo', 'Temperatura', 'Umidade', 'Vento']]
y = dados['Jogo']

encoder = OrdinalEncoder()
x_cod = encoder.fit_transform(x)

x_treino, x_teste, y_treino, y_teste = train_test_split(x_cod, y, test_size=0.2, random_state=32)

modelo = CategoricalNB()
modelo.fit(x_treino, y_treino)

y_predito = modelo.predict(x_teste)

novo_caso = encoder.transform([['Chuvoso', 'Quente', 'Alta', 'Forte']])
print("Novo caso:", novo_caso)

predicao = modelo.predict(novo_caso)
probabilidade = modelo.predict_proba(novo_caso)
acuracia = accuracy_score(y_teste, y_predito)

print("Vai ter jogo? ", predicao)
print("Quais as probabilidades?\n", probabilidade)
print("Acurácia: ", acuracia)




