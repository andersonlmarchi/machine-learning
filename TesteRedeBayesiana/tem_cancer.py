import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

df = pd.read_csv('liver_cancer_dataset.csv')
X = df.drop('liver_cancer', axis=1)
y = df['liver_cancer']

label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

test_sizes = [0.2, 0.3, 0.4, 0.5]
accuracies = []

for test_size in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f'Test size: {test_size}, Accuracy: {acc:.4f}')

plt.figure(figsize=(8,5))
plt.plot(test_sizes, accuracies, marker='o')
plt.title('Acurácia vs. Proporção de Teste')
plt.xlabel('Proporção do conjunto de teste')
plt.ylabel('Acurácia')
plt.grid(True)
plt.show()

def encode_case(case_dict):
    case = case_dict.copy()
    for col, le in label_encoders.items():
        case[col] = le.transform([case[col]])[0]
    return case

new_cases = [
    {'age': 60, 'gender': 'Female', 'bmi': 25.0, 'alcohol_consumption': 'Regular', 'smoking_status': 'Never', 'hepatitis_b': 0, 'hepatitis_c': 0, 'liver_function_score': 70.0, 'alpha_fetoprotein_level': 10.0, 'cirrhosis_history': 0, 'family_history_cancer': 0, 'physical_activity_level': 'Moderate', 'diabetes': 0},
    {'age': 45, 'gender': 'Male', 'bmi': 30.0, 'alcohol_consumption': 'Occasional', 'smoking_status': 'Current', 'hepatitis_b': 1, 'hepatitis_c': 0, 'liver_function_score': 55.0, 'alpha_fetoprotein_level': 20.0, 'cirrhosis_history': 1, 'family_history_cancer': 1, 'physical_activity_level': 'Low', 'diabetes': 1},
    {'age': 70, 'gender': 'Female', 'bmi': 22.0, 'alcohol_consumption': 'Never', 'smoking_status': 'Former', 'hepatitis_b': 0, 'hepatitis_c': 1, 'liver_function_score': 80.0, 'alpha_fetoprotein_level': 5.0, 'cirrhosis_history': 0, 'family_history_cancer': 0, 'physical_activity_level': 'High', 'diabetes': 0}
]
new_cases_encoded = pd.DataFrame([encode_case(c) for c in new_cases])

probs = model.predict_proba(new_cases_encoded)
preds = model.predict(new_cases_encoded)

for i, (p, pred) in enumerate(zip(probs, preds)):
    print(f'Caso {i+1}: Probabilidades {p}, Predição {"Sim" if pred == 1 else "Não"}')

plt.figure(figsize=(8,5))
labels = ['Não', 'Sim']
for i, p in enumerate(probs):
    plt.bar([f'Caso {i+1} - {l}' for l in labels], p, alpha=0.7)

plt.title('Probabilidades a posteriori dos novos casos')
plt.ylabel('Probabilidade')
plt.show()