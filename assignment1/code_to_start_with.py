# -*- coding: utf-8 -*-
# %% [markdown]

"""
Homework:

The folder '~//data//homework' contains data of Titanic with various features and survivals.

Try to use what you have learnt today to predict whether the passenger shall survive or not.

Evaluate your model.
# -*- coding: utf-8 -*- 
# %% [markdown] 

""" 
Homework: 

The folder '~//data//homework' contains data of Titanic with various features and survivals. 

Try to use what you have learnt today to predict whether the passenger shall survive or not. 

Evaluate your model. 
""" 
# %% 
# load data 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载数据
# 注意：根据实际数据路径调整
data = pd.read_csv('"C:\Users\30358\Desktop\aiSummerCamp2025-master\day1\assignment1\data\train.csv"')  
if data.empty:
    raise ValueError("数据加载失败，请检查文件路径是否正确")

df = data.copy() 
print("数据集样本:\n", df.sample(10)) 
# %% 
# delete some features that are not useful for prediction 
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True) 
print("数据信息:\n")
df.info() 
# %% 
# check if there is any NaN in the dataset 
print('\n处理前是否存在缺失值: {}'.format(df.isnull().values.any())) 
# 对于缺失值，除了直接删除，也可以考虑使用均值/中位数填充
# 这里使用简单删除法
original_rows = df.shape[0]
df.dropna(inplace=True)
remaining_rows = df.shape[0]
print(f'删除缺失值后剩余数据比例: {remaining_rows/original_rows:.2%}')
print('处理后是否存在缺失值: {}'.format(df.isnull().values.any())) 
# %% 
# convert categorical data into numerical data using one-hot encoding 
# For example, a feature like sex with categories ['male', 'female'] would be transformed into two new binary features, sex_male and sex_female, represented by 0 and 1. 
df = pd.get_dummies(df) 
print("\n独热编码后的数据集样本:\n", df.sample(10)) 
# %% 
# separate the features and labels 
X = df.drop('Survived', axis=1)  # 特征集
y = df['Survived']               # 标签集
print('\n特征集形状:', X.shape)
print('标签集形状:', y.shape) 
# %% 
# train-test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # 使用固定的随机种子以确保结果可复现
)
print('\n训练集大小:', X_train.shape)
print('测试集大小:', X_test.shape) 
# %% 
# build model 
# build three classification models 
# SVM, KNN, Random Forest 

def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """训练模型并评估性能"""
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    print(f'\n{model_name} 模型准确率: {accuracy:.4f}')
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print(f'混淆矩阵:\n{cm}')
    
    # 分类报告
    print('分类报告:\n', classification_report(y_test, y_pred))
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name} 混淆矩阵')
    plt.colorbar()
    classes = ['未存活', '存活']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # 在混淆矩阵上标注数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.show()
    
    return model, accuracy

# 初始化模型
models = {
    'SVM': SVC(kernel='rbf', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),  # 通常从5开始尝试
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# 存储各模型准确率用于比较
model_accuracies = {}

# %% 
# predict and evaluate 
print("\n======= 模型训练与评估 =======")
for name, model in models.items():
    trained_model, acc = train_and_evaluate_model(model, name, X_train, X_test, y_train, y_test)
    model_accuracies[name] = acc

# %% 
# 比较不同模型性能
print("\n======= 模型性能比较 =======")
best_model = max(model_accuracies, key=model_accuracies.get)
for name, acc in model_accuracies.items():
    print(f'{name}: {acc:.4f}' + (' 👈 最佳模型' if name == best_model else ''))

# 绘制模型准确率比较图
plt.figure(figsize=(10, 6))
colors = ['lightblue' if name != best_model else 'green' for name in model_accuracies.keys()]
plt.bar(model_accuracies.keys(), model_accuracies.values(), color=colors)
plt.title('不同模型准确率比较')
plt.ylabel('准确率')
plt.ylim(0, 1.0)

# 在柱状图上标注准确率
for i, v in enumerate(model_accuracies.values()):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center')

plt.tight_layout()
plt.show()

print("\n结论: 在当前数据集上，{}模型表现最佳，准确率为{:.4f}".format(best_model, model_accuracies[best_model]))
