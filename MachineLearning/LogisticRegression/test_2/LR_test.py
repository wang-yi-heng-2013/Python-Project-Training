import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 读取数据
wine_data=pd.read_csv('WineData/wine.data', header=None,
                       names=['TYPE','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium',
                              'Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins',
                              'Color intensity','Hue','OD280/OD315 of diluted wines','Proline'])

# 统计每个类别各有多少条数据
print(wine_data['TYPE'].value_counts())

# 选择要进行训练的特征
feature_choose=['Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols']
X=wine_data[feature_choose].values
Y=wine_data['TYPE'].values

# 划分训练集和测试集
x_train, x_test, y_train, y_test=train_test_split(X,Y,test_size=0.3,stratify=Y)

# 标准化
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# 调用逻辑回归模型进行训练
wine_classification = LogisticRegression(max_iter=1000)
wine_classification.fit(x_train,y_train)

# 评估结果
y_prediction=wine_classification.predict(x_test)
print("\n准确率：",accuracy_score(y_test,y_prediction))

