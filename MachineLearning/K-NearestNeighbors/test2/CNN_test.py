import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 让图表输出中文
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

# 读取数据
wine_data=pd.read_csv('WineData/wine.data', header=None,
                       names=['TYPE','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium',
                              'Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins',
                              'Color intensity','Hue','OD280/OD315 of diluted wines','Proline'])

# 统计每个类别各有多少条数据
print(wine_data['TYPE'].value_counts())

# 选取两个类别
feature_1='Alcohol'
feature_2='Total phenols'

# 绘制散点图
for label in wine_data['TYPE'].unique():
    subset=wine_data[ wine_data['TYPE']==label ]
    plt.scatter(subset[feature_1],subset[feature_2],label=str(label))

plt.xlabel(feature_1)
plt.ylabel(feature_2)
plt.title("葡萄酒在部分特征下的分布散点图")
plt.legend(title='类别')
plt.savefig('chart.png')
plt.show()

# 划分训练集和测试集
feature_choose=['Alcohol','Malic acid','Ash','Magnesium','Total phenols','Flavanoids']
X=wine_data[feature_choose].values
Y=wine_data['TYPE'].values
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,stratify=Y)

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

