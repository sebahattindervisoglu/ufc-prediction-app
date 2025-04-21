############################################################
# UFC Bahis Tahmin Modeli
# - ufcstats.com üzerinden veri toplanarak oluşturulmuştur.
# - Amaç: Bahis tahmini için makine öğrenmesi modeli geliştirmek.
# - Ek olarak sıklet, dövüşçü, yıl gibi boyutlarda analizler ve görselleştirme.
############################################################

# Gerekli Kütüphaneler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score

# Modeller
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Görselleştirme ve gösterim ayarları
def set_pandas_display():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

set_pandas_display()

# Veri Setinin Okunması
df_ = pd.read_csv('Datasets/large_dataset.csv')
df = df_.copy()

# Kullanılmayacak sütunların silinmesi
df.drop(columns=[
    'method', 'finish_round', 'time_sec',
    'r_sig_str', 'r_sig_str_att', 'r_sig_str_acc', 'r_str', 'r_str_att', 'r_str_acc',
    'r_td', 'r_td_att', 'r_td_acc', 'r_sub_att', 'r_rev', 'r_ctrl_sec',
    'b_sig_str', 'b_sig_str_att', 'b_sig_str_acc', 'b_str', 'b_str_att', 'b_str_acc',
    'b_td', 'b_td_att', 'b_td_acc', 'b_sub_att', 'b_rev', 'b_ctrl_sec',
    'kd_diff', 'sig_str_diff', 'sig_str_att_diff', 'sig_str_acc_diff',
    'str_diff', 'str_att_diff', 'str_acc_diff',
    'td_diff', 'td_att_diff', 'td_acc_diff',
    'sub_att_diff', 'rev_diff', 'ctrl_sec_diff',
    'wins_total_diff', 'losses_total_diff', 'age_diff', 'height_diff',
    'weight_diff', 'reach_diff', 'SLpM_total_diff', 'SApM_total_diff',
    'sig_str_acc_total_diff', 'td_acc_total_diff', 'str_def_total_diff',
    'td_def_total_diff', 'sub_avg_diff', 'td_avg_diff', 'event_name'
], inplace=True)

############################################################
# EDA Fonksiyonları
############################################################

def check_df(dataframe):
    print("##### Shape #####")
    print(dataframe.shape)
    print("##### Types #####")
    print(dataframe.dtypes)
    print("##### Head #####")
    print(dataframe.head(3))
    print("##### Tail #####")
    print(dataframe.tail(3))
    print("##### NA #####")
    print(dataframe.isnull().sum())
    print("##### Quantiles #####")
    print(dataframe.select_dtypes(include=["int64", "float64"]).quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Kategorik, sayısal ve yüksek kardinaliteli değişkenleri ayırır.
    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, cat_but_car, num_cols

cat_cols, cat_but_car, num_cols = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({
        col_name: dataframe[col_name].value_counts(),
        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
    }))
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.title(col_name)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)

def num_summary(dataframe, numerical_col, plot=True):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()
    print("#####################################")

for col in num_cols:
    num_summary(df, col)

# Label Encoding: Red (1), Blue (0)
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

label_encoder(df, 'winner')

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({
        "TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()
    }), end="\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "winner", col)

# Korelasyon Analizi
corr = df[num_cols].corr()
sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

def high_correlated_cols(dataframe, plot=False, corr_th=0.70, verbose=False):
    corr = dataframe.corr().abs()
    upper_triangle_matrix = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]

    if verbose:
        print(f"Yüksek Korelasyonlu Değişken Çiftleri (>{corr_th}):")
        for i in range(len(upper_triangle_matrix.columns)):
            for j in range(i):
                if upper_triangle_matrix.iloc[j, i] > corr_th:
                    print(f"{upper_triangle_matrix.columns[i]} & {upper_triangle_matrix.index[j]} => "
                          f"Korelasyon: {upper_triangle_matrix.iloc[j, i]:.2f}")

    if plot:
        sns.heatmap(corr, cmap="RdBu", annot=True)
        plt.show()

    return drop_list

high_correlated_cols(df[num_cols], plot=True, verbose=True)

######################################
# FEATURE ENGINEERING
######################################

# Body Mass Index (BMI)
df["r_body_mass_index"] = df["r_weight"] / ((df["r_height"] / 100) ** 2)
df["b_body_mass_index"] = df["b_weight"] / ((df["b_height"] / 100) ** 2)

# Temel farklar
df["wins_total_diff"] = df["r_wins_total"] - df["b_wins_total"]
df["losses_total_diff"] = df["r_losses_total"] - df["b_losses_total"]
df["age_diff"] = df["r_age"] - df["b_age"]
df["height_diff"] = df["r_height"] - df["b_height"]
df["weight_diff"] = df["r_weight"] - df["b_weight"]
df["reach_diff"] = df["r_reach"] - df["b_reach"]

# Performans farkları
df["SLpM_total_diff"] = df["r_SLpM_total"] - df["b_SLpM_total"]
df["SApM_total_diff"] = df["r_SApM_total"] - df["b_SApM_total"]
df["sig_str_acc_total_diff"] = df["r_sig_str_acc_total"] - df["b_sig_str_acc_total"]
df["td_acc_total_diff"] = df["r_td_acc_total"] - df["b_td_acc_total"]
df["str_def_total_diff"] = df["r_str_def_total"] - df["b_str_def_total"]
df["td_def_total_diff"] = df["r_td_def_total"] - df["b_td_def_total"]
df["sub_avg_diff"] = df["r_sub_avg"] - df["b_sub_avg"]
df["td_avg_diff"] = df["r_td_avg"] - df["b_td_avg"]
df["kd_diff"] = df["r_kd"] - df["b_kd"]

# Kategorik kıyaslamalar (binary featurelar)
df["is_r_younger"] = (df["r_age"] < df["b_age"]).astype(int)
df["is_r_taller"] = (df["r_height"] > df["b_height"]).astype(int)
df["is_r_more_experienced"] = (df["r_wins_total"] > df["b_wins_total"]).astype(int)
df["is_r_better_td_acc"] = (df["r_td_acc_total"] > df["b_td_acc_total"]).astype(int)
df["same_stance"] = (df["r_stance"] == df["b_stance"]).astype(int)

# Güç ve savunma skorları
df["r_power_striking_score"] = df["r_SLpM_total"] * df["r_sig_str_acc_total"]
df["b_power_striking_score"] = df["b_SLpM_total"] * df["b_sig_str_acc_total"]
df["r_defense_score"] = df["r_str_def_total"] + df["r_td_def_total"]
df["b_defense_score"] = df["b_str_def_total"] + df["b_td_def_total"]

df["overall_score_diff"] = (
    (df["r_SLpM_total"] - df["b_SLpM_total"]) * 0.4 +
    (df["r_str_def_total"] - df["b_str_def_total"]) * 0.3 +
    (df["r_td_def_total"] - df["b_td_def_total"]) * 0.3
)

# Fazla korelasyonlu kolonları düşür
df.drop(columns=["r_height", "r_weight", "b_height", "b_weight", "r_reach", "b_reach"], inplace=True)


######################################
# OUTLIER ANALYSIS
######################################

def outlier_thresholds(dataframe, variable, low_quantile=0.05, up_quantile=0.95):
    q1 = dataframe[variable].quantile(low_quantile)
    q3 = dataframe[variable].quantile(up_quantile)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper

def check_outlier(dataframe, col_name):
    lower, upper = outlier_thresholds(dataframe, col_name)
    return dataframe[(dataframe[col_name] < lower) | (dataframe[col_name] > upper)].any(axis=None)

def replace_with_thresholds(dataframe, variable):
    lower, upper = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < lower, variable] = lower
    dataframe.loc[dataframe[variable] > upper, variable] = upper

cat_cols, cat_but_car, num_cols = grab_col_names(df)

# Aykırı değerleri baskıla
for col in num_cols:
    if col != "winner":
        replace_with_thresholds(df, col)


######################################
# MISSING VALUE ANALYSIS
######################################

def missing_values_table(dataframe, na_name=False):
    na_cols = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_cols].isnull().sum().sort_values(ascending=False)
    ratio = (n_miss / len(dataframe) * 100).round(2)
    missing_df = pd.concat([n_miss, ratio], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n\n")
    if na_name:
        return na_cols

missing_values_table(df)

# Eksik değerleri median/mode ile doldur
def quick_missing_imp(data, num_method="median", cat_length=20, target="winner"):
    variables_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]
    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n")

    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis=0)

    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER")
    print(data[variables_with_na].isnull().sum(), "\n")
    return data

df = quick_missing_imp(df, num_method="median", cat_length=17)

# Gereksiz kolon
df.drop(columns=["referee"], inplace=True)


######################################
# RARE ENCODING
######################################

# Kategorik değişkenlerde nadir sınıfların analiz edilmesi
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(f"{col} : {dataframe[col].nunique()}")
        print(pd.DataFrame({
            "COUNT": dataframe[col].value_counts(),
            "RATIO": dataframe[col].value_counts() / len(dataframe),
            "TARGET_MEAN": dataframe.groupby(col)[target].mean()
        }), end="\n\n\n")


rare_analyser(df, "winner", cat_cols)


# Rare encoder: Nadir sınıfları "Rare" etiketiyle grupla
def rare_encoder(dataframe, rare_perc=0.01):
    temp_df = dataframe.copy()
    rare_cols = [col for col in temp_df.columns if temp_df[col].dtype == 'O' and
                 (temp_df[col].value_counts() / len(temp_df) < rare_perc).any()]

    for col in rare_cols:
        freqs = temp_df[col].value_counts() / len(temp_df)
        rare_labels = freqs[freqs < rare_perc].index
        temp_df[col] = np.where(temp_df[col].isin(rare_labels), "Rare", temp_df[col])

    return temp_df


df = rare_encoder(df, rare_perc=0.1)

######################################
# ENCODING
######################################

# Güncel kategorik ve nümerik kolonları tekrar alalım
cat_cols, cat_but_car, num_cols = grab_col_names(df)


# Label Encoding: Binary kategorik kolonlar
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtype == "O" and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

# One-Hot Encoding: Diğer kategorik değişkenler
df = one_hot_encoder(df, cat_cols, drop_first=True)

######################################
# MODELLEME: BASELINE + TUNING
######################################

# Hedef ve bağımsız değişkenler
y = df["winner_1"]
X = df.drop(["winner_1"], axis=1)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

# Ölçekleme
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model listesi
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(verbose=False, random_state=42),
    "SVC": SVC(probability=True, random_state=42)
}

# Modellerin değerlendirilmesi
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred, average='weighted')
    }

# Sonuçların yazdırılması
for name, score in results.items():
    print(f"{name}: Accuracy = {score['Accuracy']:.4f}, F1 Score = {score['F1 Score']:.4f}")

# Logistic Regression: Accuracy = 0.7870, F1 Score = 0.7807
# KNN: Accuracy = 0.6989, F1 Score = 0.6905
# Decision Tree: Accuracy = 0.7171, F1 Score = 0.7172
# Random Forest: Accuracy = 0.7769, F1 Score = 0.7689
# Gradient Boosting: Accuracy = 0.7876, F1 Score = 0.7824
# XGBoost: Accuracy = 0.7823, F1 Score = 0.7789
# LightGBM: Accuracy = 0.7903, F1 Score = 0.7869
# CatBoost: Accuracy = 0.7997, F1 Score = 0.7960 ****
# SVC: Accuracy = 0.7923, F1 Score = 0.7847

######################################
# CATBOOST TUNING
######################################

# Feature importance çizdirme fonksiyonu
def plot_importance(model, features, num=20, save=False):
    feature_imp = pd.DataFrame({
        "Value": model.feature_importances_,
        "Feature": features.columns
    })
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[:num])
    plt.title("Feature Importances")
    plt.tight_layout()
    if save:
        plt.savefig("importances.png")
    plt.show()


# CatBoost modeli eğit ve importance göster
cat_model = CatBoostClassifier(verbose=False, random_state=42)
cat_model.fit(X_train_scaled, y_train)
plot_importance(cat_model, X)

# Cross-validation (base model)
cv_results = cross_validate(cat_model, X_train_scaled, y_train, cv=5, scoring=["accuracy", "f1"], n_jobs=-1)
print(f"CV Accuracy (mean): {cv_results['test_accuracy'].mean():.4f}")
print(f"CV F1 Score (mean): {cv_results['test_f1'].mean():.4f}")

# CV Accuracy (mean): 0.8118
# CV F1 Score (mean): 0.8621

# Hiperparametre arama (RandomizedSearchCV)
param_grid = {
    'iterations': [100, 200, 300],
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'l2_leaf_reg': [1, 3, 5, 7, 9]
}

# Randomized Search
random_search = RandomizedSearchCV(estimator=cat_model,
                                   param_distributions=param_grid,
                                   n_iter=20,             # İstersen artırılabilir
                                   scoring='f1',
                                   cv=5,
                                   verbose=1,
                                   random_state=42,
                                   n_jobs=-1)
random_search.fit(X_train_scaled, y_train)

# En iyi model sonuçları
print("Best Params:", random_search.best_params_)
print("Best F1 Score:", random_search.best_score_)

# En iyi model ile final CV
best_catboost_model = random_search.best_estimator_

### PKL dosyaları ###########
import joblib

# Model, scaler ve kolon isimlerini pkl olarak kaydet
joblib.dump(best_catboost_model, "best_catboost_model.pkl")
joblib.dump(scaler, "robust_scaler.pkl")
joblib.dump(X_train.columns.tolist(), "model_columns.pkl")
#####################################################################

cv_results = cross_validate(best_catboost_model, X_train, y_train, cv=5,
                            scoring=["accuracy", "f1"], return_train_score=True, n_jobs=-1)

print(f"Accuracy (CV Test): {cv_results['test_accuracy'].mean():.4f}")
print(f"F1 Score (CV Test): {cv_results['test_f1'].mean():.4f}")
print(f"Accuracy (CV Train): {cv_results['train_accuracy'].mean():.4f}")
print(f"F1 Score (CV Train): {cv_results['train_f1'].mean():.4f}")

# Best Params: {'learning_rate': 0.05, 'l2_leaf_reg': 5, 'iterations': 300, 'depth': 6}
# Best F1 Score: 0.8635304578854637
# Accuracy (CV Test): 0.8118
# F1 Score (CV Test): 0.8624
# Accuracy (CV Train): 0.9020
# F1 Score (CV Train): 0.9276


# Feature importance tekrar çiz
plot_importance(best_catboost_model, X_train, num=20)


#############################################################
# Test verisi için kazanma olasılığı (Red = 1)
y_probs = best_catboost_model.predict_proba(X_test_scaled)[:, 1]  # 1: red kazanır

# Olasılıkların ilk 5 tanesi
print(y_probs[:5])

from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_test, y_probs)
print(f"ROC AUC: {roc_auc:.4f}")

for i, prob in enumerate(y_probs):
    winner = "Red" if prob >= 0.5 else "Blue"
    confidence = prob if prob >= 0.5 else 1 - prob
    print(f"{i+1}. dövüş: %{confidence*100:.1f} ihtimalle {winner} kazanır.")


import shap
explainer = shap.TreeExplainer(best_catboost_model)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train)