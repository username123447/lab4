"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

from grader_contracts.ensembles import EnsembleInput, EnsembleResult
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    df = pd.read_csv('german.csv', sep=';')
    df = pd.get_dummies(df, drop_first=True)
    y = df['Creditability']
    X = df.drop(columns=['Creditability'])
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state= data.random_seed
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)    

    models_to_train = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, 
            max_depth=9, 
            min_samples_split=10,
            random_state=data.random_seed
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.05, 
            max_iter=150, 
            max_depth=5,
            random_state=data.random_seed
        ),
        "MLP (Neural Network)": MLPClassifier(
            hidden_layer_sizes=(32, 16), 
            activation='relu',
            alpha=0.01, 
            random_state=data.random_seed, 
            max_iter=1000
        )
    }

    y_probs = {}
    for name, model in models_to_train.items():
        model.fit(X_train_scaled, y_train)

        y_prob = model.predict_proba(X_test_scaled)[:,1]
        y_probs[name] = y_prob

    y_prob_blend = (
        y_probs["Random Forest"] * 0.8
        + y_probs["HistGradientBoosting"] * 0.1
        + y_probs["MLP (Neural Network)"] * 0.1
    )
    y_pred_blend = (y_prob_blend >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_prob_blend)
    acc = accuracy_score(y_test, y_pred_blend)
    prec = precision_score(y_test, y_pred_blend)
    rec = recall_score(y_test, y_pred_blend)
    f1 = f1_score(y_test, y_pred_blend)
    
    return EnsembleResult(
      model_name="Weighted Ensemble",
      probabilities=y_prob_blend,
      predictions=y_pred_blend,
      roc_auc=float(roc_auc),
      accuracy=float(acc),
      precision=float(prec),
      recall=float(rec),
      f1=float(f1),
  )
        
