import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
# Modelos
from sklearn.linear_model import LogisticRegression
# Split
from sklearn.model_selection import train_test_split


def algoritmo():
    # Opcion
    df = pd.read_csv(
        "C:/Users/Acer/Documents/Repositorios_Github/SmartRisk_NC/data/csv/df_concat_a.csv")
    # Cargamos el datasetpip
    # df = pd.read_csv("../../data/csv/df_concat_a.csv")
    # df = df.drop(columns="Unnamed: 0")

    # Oversampling
    smote = SMOTE(random_state=16)

    # Definimos target y predictoras
    X = df.drop(columns=["sk_id_curr", "target"])
    y = df["target"]

    # Balanceamos las clases
    X_balanced, y_balanced = smote.fit_resample(X, y)

    # Hacemos el train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.20, random_state=16)

    # Instanciamos la regresion logística y la ajustamos
    reg = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    # Hacemos predicciones en train y test
    y_pred_train = reg.predict(X_train)
    y_pred_test = reg.predict(X_test)

    return reg
