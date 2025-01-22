import subprocess

# Age_Estimation.pyの実行
proc_age = subprocess.Popen(["python", "Age_Estimation.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
stdout_age, stderr_age = proc_age.communicate()
output_age = stdout_age.decode().split('\n')

# alcohol.pyの実行
proc_alcohol = subprocess.Popen(["python", "alcohol.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
stdout_alcohol, stderr_alcohol = proc_alcohol.communicate()
output_alcohol = stdout_alcohol.decode().split('\n')

# 正答率を取り出す
accuracy_age = float(output_age[-2].split(": ")[1])  # Age_Estimation.pyの正答率
accuracy_alcohol = float(output_alcohol[-2].split(": ")[1])  # alcohol.pyの正答率

# 判定結果を表示
print("Age Estimation Results:")
print("\n".join(output_age[:-2]))  # 最後の2行（空行と正答率行）を除く
print("\nAlcohol Detection Results:")
print("\n".join(output_alcohol[:-2]))  # 最後の2行（空行と正答率行）を除く

# 各々の正答率を表示
print("\n年齢推定の正答率: ", accuracy_age)
print("酔っ払い推定の正答率: ", accuracy_alcohol)

# 総合正答率を計算し表示
accuracy_overall = (accuracy_age + accuracy_alcohol) / 2
print("総合正答率: ", accuracy_overall)
