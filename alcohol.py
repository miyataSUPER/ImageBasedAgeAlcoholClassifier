import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# データの読み込み
train_folder = "train/"
test_folder = "Alcohol_test/"

train_labels = []
train_images = []
test_labels = []
test_images = []

# ラベル辞書の作成
label_dict = {"adult": "normal", "no": "drunkard"}

# 読み込むフォルダの指定
folders_to_load = ["adult", "no"]

# ラベルエンコーダーの初期化
label_encoder = LabelEncoder()

# 訓練データの読み込み
for folder_name in folders_to_load:
    age_label = label_dict[folder_name]  # フォルダ名から正解ラベルを抽出
    for file_name in sorted(os.listdir(train_folder + folder_name)):  # ソートを追加
        if file_name == ".DS_Store":
            continue

        img = cv2.imread(train_folder + folder_name + "/" + file_name, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (100, 100))  # 画像サイズをリサイズ
        train_images.append(img.flatten())  # 画像データをフラット化
        train_labels.append(age_label)

# ラベルのエンコーディング
train_labels_encoded = label_encoder.fit_transform(train_labels)

# テストデータの読み込み
for file_name in sorted(os.listdir(test_folder)):  # ソートを追加
    if file_name == ".DS_Store":
        continue

    img = cv2.imread(test_folder + file_name, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (100, 100))  # 画像サイズをリサイズ
    test_images.append(img.flatten())  # 画像データをフラット化
    age_label = file_name.split("_")[0]  # ファイル名から正解ラベルを抽出
    test_labels.append(age_label)

# ラベルのエンコーディング
test_labels_encoded = label_encoder.transform(test_labels)

# ベース学習器の定義
# Random ForestとSVMのパラメータは手動で設定
# Random Forest
rf_n_estimators = 15
rf_max_depth = None
rf_min_samples_split = 3
rf_min_samples_leaf = 3

# SVM
svm_C = 5
svm_kernel = 'rbf'
svm_degree = 1
svm_gamma = 'scale'

base_learners = [
                 ('rf', RandomForestClassifier(n_estimators=rf_n_estimators, 
                                               max_depth=rf_max_depth,
                                               min_samples_split=rf_min_samples_split,
                                               min_samples_leaf=rf_min_samples_leaf,
                                               random_state=42)),
                 ('svc', svm.SVC(C=svm_C,
                                 kernel=svm_kernel,
                                 degree=svm_degree,
                                 gamma=svm_gamma,
                                 probability=True))
                ]

# スタッキング分類器（メタ学習器としてロジスティック回帰を使用）
stacking_model = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression())

# モデルの学習
stacking_model.fit(np.array(train_images), train_labels_encoded)

# モデルを使って予測
predictions = stacking_model.predict(np.array(test_images))

# 結果の出力
correct_predictions = 0
for i in range(len(predictions)):
    result = label_encoder.inverse_transform([predictions[i]])[0] + " : " + test_labels[i]
    if label_encoder.inverse_transform([predictions[i]])[0] == test_labels[i]:
        result += " True"
        correct_predictions += 1
    else:
        result += " False"
    print(result)

# 正答率の計算と出力
accuracy = correct_predictions / len(predictions)
print("正答率: ", accuracy)

# テストデータに対する予測確率を計算
probs = stacking_model.predict_proba(np.array(test_images))
probs = probs[:, 1]

# ROC曲線の計算
fpr, tpr, thresholds = roc_curve(test_labels_encoded, probs)
roc_auc = auc(fpr, tpr)

# ROC曲線の描画
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc, color='blue')
plt.plot([0, 1], [0, 1], 'k--', color='red')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
