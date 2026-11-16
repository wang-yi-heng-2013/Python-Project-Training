import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 读取数据
glass_data=pd.read_csv('GlassData/glass.data', header=None,
                       names=['ID','RI','NA2O','MGO','AL2O3','SIO2','K2O','CAO','BAO','FE2O3','TYPE'])

# 统计每个类别各有多少条数据
glass_data['TYPE']=np.where(glass_data['TYPE'].isin([1,2,3,4]),1,2)
print(glass_data['TYPE'].value_counts())

# 选择特征
feature_choose=['RI','NA2O','MGO','AL2O3','SIO2','K2O','CAO','BAO','FE2O3']
X=glass_data[feature_choose].values
Y=glass_data['TYPE'].values

# 标准化处理，为降维做准备
X_processed=StandardScaler().fit_transform(X)

# 降到二维
pca=PCA(n_components=2)
X_pca=pca.fit_transform(X_processed)

# 绘制散点图
for label in np.unique(Y):
    mask= Y==label
    plt.scatter(X_pca[mask,0],X_pca[mask,1],label=str(label),alpha=0.5)

plt.xlabel("")
plt.ylabel("")
plt.legend(title='TYPE')
plt.show()


# 划分训练集和测试集
x_train,x_test,y_train,y_test=train_test_split(X_pca,Y,test_size=0.3,stratify=Y)

# 进行标准化操作
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# 调用KNN模型进行训练
wine_classification=KNeighborsClassifier(n_neighbors=5)
wine_classification.fit(x_train,y_train)

# 评估结果
y_prediction=wine_classification.predict(x_test)
print("\n准确率：",accuracy_score(y_test,y_prediction))

