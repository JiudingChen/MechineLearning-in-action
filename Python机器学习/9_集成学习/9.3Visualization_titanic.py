import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据集
file_path = 'D:/python/Python机器学习/9_集成学习/titanic-data.xlsx'  # 请根据您的文件位置进行调整
titanic_data = pd.read_excel(file_path)

# 数据可视化
plt.figure(figsize=(12, 8))

# 年龄分布图
plt.subplot(2, 3, 1)
sns.histplot(titanic_data['Age'].dropna(), kde=True, bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

# 船舱等级分布图
plt.subplot(2, 3, 2)
sns.countplot(x='Pclass', data=titanic_data)
plt.title('Pclass Distribution')
plt.xlabel('Pclass')
plt.ylabel('Count')

# 性别分布图
plt.subplot(2, 3, 3)
sns.countplot(x='Sex', data=titanic_data)
plt.title('Sex Distribution')
plt.xlabel('Sex')
plt.ylabel('Count')

# 登船地点分布图
plt.subplot(2, 3, 4)
sns.countplot(x='Embarked', data=titanic_data)
plt.title('Embarked Distribution')
plt.xlabel('Embarked')
plt.ylabel('Count')

# 票价分布图
plt.subplot(2, 3, 5)
sns.histplot(titanic_data['Fare'].dropna(), kde=True, bins=30)
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Frequency')

# 生存与年龄的关系图
plt.subplot(2, 3, 6)
sns.boxplot(x='Survived', y='Age', data=titanic_data)
plt.title('Survival by Age')
plt.xlabel('Survived')
plt.ylabel('Age')

plt.tight_layout()
plt.show()

# 生存与船舱等级的关系图
sns.catplot(x='Pclass', y='Survived', kind='bar', data=titanic_data)
plt.title('Survival by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Survival Probability')

# 生存与性别的关系图
sns.catplot(x='Sex', y='Survived', kind='bar', data=titanic_data)
plt.title('Survival by Sex')
plt.xlabel('Sex')
plt.ylabel('Survival Probability')

plt.show()
