import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support,
                             classification_report, roc_curve, auc)

from keras.models import Sequential
from keras.layers import Dense, Dropout

# link: https://www.kaggle.com/datasets/khushikyad001/mental-health-and-burnout-in-the-workplace
def load_data(path="mental_health_workplace_survey.csv"):
    return pd.read_csv(path)

def preprocess(df):
    df = df.copy()
    if "EmployeeID" in df.columns:
        df = df.drop(columns=["EmployeeID"])

    if "BurnoutRisk" not in df.columns:
        raise ValueError("Coluna alvo 'BurnoutRisk' não encontrada no dataset")

    X = df.drop(columns=["BurnoutRisk"])
    y = df["BurnoutRisk"].astype(int)

    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values


def build_mlp(input_dim, hidden_layers=[64, 32], dropout=0.2):
    model = Sequential()
    model.add(Dense(hidden_layers[0], activation='relu', input_dim=input_dim))
    if dropout and dropout > 0:
        model.add(Dropout(dropout))
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
        if dropout and dropout > 0:
            model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_eval_config(X_train, X_test, y_train, y_test, config):
    model = build_mlp(input_dim=X_train.shape[1], hidden_layers=config['layers'], dropout=config['dropout'])
    history = model.fit(X_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'],
                        verbose=0, validation_data=(X_test, y_test))

    y_proba = model.predict(X_test).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[0,1])
    report = classification_report(y_test, y_pred, digits=4)

    try:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
    except Exception:
        fpr, tpr, roc_auc = None, None, None

    metrics = {
        'accuracy': acc,
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': report,
        'roc': (fpr, tpr, roc_auc),
        'history': history.history
    }
    return metrics

def main():
    st.set_page_config(layout='wide')
    st.title("Projeto Prático MLP — Burnout Risk (Classificação)")

    test_size = 30
    random_state = 42
    configs = [
        {"name": "MLP-1", "layers": [32, 16], "dropout": 0.2, "epochs": 30, "batch_size": 32},
        {"name": "MLP-2", "layers": [64, 32], "dropout": 0.3, "epochs": 40, "batch_size": 64},
        {"name": "MLP-3", "layers": [128, 64, 32], "dropout": 0.4, "epochs": 50, "batch_size": 128},
    ]

    df = load_data()

    st.markdown("---")
    st.subheader("Amostra do dataset")
    st.dataframe(df.head())

    st.markdown("---")
    st.subheader("Sumário estatístico")
    st.write(df.describe(include='all'))

    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=int(random_state))

    st.write(f"Treinando {len(configs)} configurações com {test_size}% de dados de teste!")

    results = []

    prog = st.progress(0)
    for i, cfg in enumerate(configs):
        st.markdown("---")
        st.write(f"### Treinando: {cfg.get('name', str(cfg))}")
        metrics = train_eval_config(X_train, X_test, y_train, y_test, cfg)
        results.append({
            'name': cfg.get('name', f'cfg_{i+1}'),
            'layers': cfg['layers'],
            'dropout': cfg['dropout'],
            'epochs': cfg['epochs'],
            'batch_size': cfg['batch_size'],
            'accuracy': metrics['accuracy'],
            'precision_class0': metrics['precision'][0],
            'precision_class1': metrics['precision'][1],
            'recall_class0': metrics['recall'][0],
            'recall_class1': metrics['recall'][1],
            'f1_class0': metrics['f1'][0],
            'f1_class1': metrics['f1'][1],
            'confusion_matrix': metrics['confusion_matrix'],
            'roc_auc': metrics['roc'][2],
            'report': metrics['classification_report']
        })
        prog.progress(int((i+1)/len(configs)*100))

    st.markdown("---")
    results_df = pd.DataFrame(results)
    st.subheader("Resultados comparativos")
    st.dataframe(results_df.drop(columns=['confusion_matrix','report']))

    for r in results:
        st.markdown("---")
        st.subheader(f"# Resultado: {r['name']}")
        st.write(f"Acurácia: {r['accuracy']:.4f}")
        st.write(f"Precisão (classe 0 / 1): {r['precision_class0']:.4f} / {r['precision_class1']:.4f}")
        st.write(f"Revocação (classe 0 / 1): {r['recall_class0']:.4f} / {r['recall_class1']:.4f}")
        st.write(f"F1-score (classe 0 / 1): {r['f1_class0']:.4f} / {r['f1_class1']:.4f}")
        st.write(f"ROC AUC: {r['roc_auc']}")

        cm = r['confusion_matrix']
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Previsão')
        ax.set_ylabel('Real')
        ax.set_title(f"Matriz de Confusão - {r['name']}")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Melhor configuração (por F1 da classe 1):")
    best = max(results, key=lambda x: x['f1_class1'])
    st.success(f"{best['name']} — f1 = {best['f1_class1']:.4f}")

    st.markdown("---")
    st.subheader("Gráficos da melhor rede")
    best_cfg = [c for c in configs if c.get('name') == best['name']][0]
    model = build_mlp(input_dim=X_train.shape[1], hidden_layers=best_cfg['layers'], dropout=best_cfg['dropout'])
    history = model.fit(X_train, y_train, epochs=best_cfg['epochs'], batch_size=best_cfg['batch_size'],
                        verbose=0, validation_data=(X_test, y_test))

    fig, ax = plt.subplots(1, 2, figsize=(12,4))
    ax[0].plot(history.history['loss'], label='Treinamento')
    ax[0].plot(history.history['val_loss'], label='Validação')
    ax[0].set_title('Perda')
    ax[0].legend()

    ax[1].plot(history.history['accuracy'], label='Treinamento')
    ax[1].plot(history.history['val_accuracy'], label='Validação')
    ax[1].set_title('Acurácia')
    ax[1].legend()
    st.pyplot(fig)

if __name__ == '__main__':
    main()
