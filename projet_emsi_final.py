import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml, fetch_20newsgroups
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, KBinsDiscretizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings('ignore')

# =================================================================
# I. ÉTUDE DE CAS SUR DONNÉES TABULAIRES (TITANIC)
# =================================================================
print("--- DÉMARRAGE PARTIE I : DONNÉES TABULAIRES ---")

# 1. Inspection et Chargement
titanic = fetch_openml('titanic', version=1, as_frame=True, parser='auto')
df = titanic.frame
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
X_t = df[features]
y_t = df['survived'].astype(int)

# 2. Stratégie de Prétraitement (ColumnTransformer)
# Distinction explicite : Numérique, Ordinal, Nominal
numeric_cols = ['age', 'sibsp', 'parch']
nominal_cols = ['sex', 'embarked']
ordinal_cols = ['pclass']
fare_col = ['fare']

# Transformers
num_pipe = Pipeline([('imputer', KNNImputer(n_neighbors=5)), ('scaler', StandardScaler())])
nom_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))])
ord_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ord', OrdinalEncoder())])
fare_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), 
                      ('bin', KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')),
                      ('scaler', StandardScaler())])

preprocessor_t = ColumnTransformer([
    ('num', num_pipe, numeric_cols),
    ('nom', nom_pipe, nominal_cols),
    ('ord', ord_pipe, ordinal_cols),
    ('fare', fare_pipe, fare_col)
])

# 3. Pipelines et Comparaison
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_t, y_t, test_size=0.2, random_state=42)

# Recherche d'hyperparamètres (GridSearchCV) pour RandomForest
rf_pipe = Pipeline([('prep', preprocessor_t), ('clf', RandomForestClassifier(random_state=42))])
param_grid = {'clf__n_estimators': [50, 100], 'clf__max_depth': [None, 10]}
grid_t = GridSearchCV(rf_pipe, param_grid, cv=5, scoring='f1')
grid_t.fit(X_train_t, y_train_t)

print(f"Meilleurs paramètres Titanic : {grid_t.best_params_}")
print(f"Meilleur score F1 : {grid_t.best_score_:.4f}")

# Visualisation Matrix et Importance
ConfusionMatrixDisplay.from_estimator(grid_t, X_test_t, y_test_t, cmap='Blues')
plt.title("Titanic : Matrice de Confusion (Optimisée)")
plt.show()

# =================================================================
# II. ÉTUDE DE CAS SUR DONNÉES TEXTUELLES (20 NEWSGROUPS)
# =================================================================
print("\n--- DÉMARRAGE PARTIE II : DONNÉES TEXTUELLES ---")

categories = ['sci.space', 'rec.sport.baseball', 'talk.politics.mideast']
data_train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
data_test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))

# Comparaison Count vs TF-IDF vs Hashing
pipes_text = {
    'Count': Pipeline([('vect', CountVectorizer(stop_words='english', ngram_range=(1,2))), ('clf', MultinomialNB())]),
    'TF-IDF': Pipeline([('vect', TfidfVectorizer(stop_words='english', ngram_range=(1,2))), ('clf', MultinomialNB())]),
    'Hashing': Pipeline([('vect', HashingVectorizer(stop_words='english', n_features=2**16, alternate_sign=False)), ('clf', MultinomialNB())])
}

for name, p in pipes_text.items():
    score = cross_val_score(p, data_train.data, data_train.target, cv=5).mean()
    print(f"Performance {name} : {score:.4f}")

# Évaluation finale sur TF-IDF
final_text_pipe = pipes_text['TF-IDF']
final_text_pipe.fit(data_train.data, data_train.target)
y_pred_text = final_text_pipe.predict(data_test.data)

print("\nClassification Report (Texte) :")
print(classification_report(data_test.target, y_pred_text, target_names=categories))
