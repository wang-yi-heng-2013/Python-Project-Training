import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 读取数据
glass_data=pd.read_csv('GlassData/glass.data', header=None,
                       names=['ID','RI','NA2O','MGO','AL2O3','SIO2','K2O','CAO','BAO','FE2O3','TYPE'])

# 使用的是逻辑回归，将类别进行二值化处理，即类别分为Window和Non-window两种,并统计各有多少条数据
glass_data['TYPE']=np.where(glass_data['TYPE'].isin([1,2,3,4]),1,2)
print(glass_data['TYPE'].value_counts())

# 选择特征
feature_choose=['RI','NA2O','MGO','AL2O3','SIO2','K2O']
X=glass_data[feature_choose].values
Y=glass_data['TYPE'].values

# 划分训练集和测试集
x_train, x_test, y_train, y_test=train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)

# 标准化
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# 调用逻辑回归模型进行训练
glass_identification = LogisticRegression(max_iter=1000)
glass_identification.fit(x_train,y_train)

# 评估结果
y_prediction=glass_identification.predict(x_test)
print("\n准确率：",accuracy_score(y_test,y_prediction))

