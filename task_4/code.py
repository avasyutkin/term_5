import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import numpy as np
from mlxtend.plotting import plot_decision_regions


"""df = pd.read_csv('circles.csv')
my = ListedColormap(['lightgray', 'gray', 'darkgray'])
plt.figure()
plt.scatter(df['x1'], df['x2'], c=df['label'], cmap=my)
plt.axis('equal')
plt.show()"""

data = pd.read_csv("D:/tasks/task_1/creditcard.csv")

"""
plt.figure()
g = sns.jointplot(x=data['V2'], y=data['V26'], kind='scatter', color='red', edgecolor="blue", alpha=0.6, space=0)
g.ax_joint.set_aspect('equal')
plt.title("Model data")
plt.show()
"""
data = data.drop(['Amount'], axis=1)  #удаляем столб суммы
#print(data.isnull().sum())  #проверка строк с пустыми ячейками
data = data.dropna()  #содержащих NaN

data = data[(data.Class == 1) | (data.Class == 0).sample(n=492)]
data_without_class = data.iloc[:, :29]
confidence_interval_min = np.mean(data_without_class)-3*np.std(data_without_class)
confidence_interval_max = np.mean(data_without_class)+3*np.std(data_without_class)



data = data.loc[(confidence_interval_min[0] < data.Time) & (confidence_interval_max[0] > data.Time)]
data = data.loc[(confidence_interval_min[1] < data.V1) & (confidence_interval_max[1] > data.V1)]
data = data.loc[(confidence_interval_min[2] < data.V2) & (confidence_interval_max[2] > data.V2)]
data = data.loc[(confidence_interval_min[3] < data.V3) & (confidence_interval_max[3] > data.V3)]
data = data.loc[(confidence_interval_min[4] < data.V4) & (confidence_interval_max[4] > data.V4)]
data = data.loc[(confidence_interval_min[5] < data.V5) & (confidence_interval_max[5] > data.V5)]
data = data.loc[(confidence_interval_min[6] < data.V6) & (confidence_interval_max[6] > data.V6)]
data = data.loc[(confidence_interval_min[7] < data.V7) & (confidence_interval_max[7] > data.V7)]
data = data.loc[(confidence_interval_min[8] < data.V8) & (confidence_interval_max[8] > data.V8)]
data = data.loc[(confidence_interval_min[9] < data.V9) & (confidence_interval_max[9] > data.V9)]
data = data.loc[(confidence_interval_min[10] < data.V10) & (confidence_interval_max[10] > data.V10)]
data = data.loc[(confidence_interval_min[11] < data.V11) & (confidence_interval_max[11] > data.V11)]
data = data.loc[(confidence_interval_min[12] < data.V12) & (confidence_interval_max[12] > data.V12)]
data = data.loc[(confidence_interval_min[13] < data.V13) & (confidence_interval_max[13] > data.V13)]
data = data.loc[(confidence_interval_min[14] < data.V14) & (confidence_interval_max[14] > data.V14)]
data = data.loc[(confidence_interval_min[15] < data.V15) & (confidence_interval_max[15] > data.V15)]
data = data.loc[(confidence_interval_min[16] < data.V16) & (confidence_interval_max[16] > data.V16)]
data = data.loc[(confidence_interval_min[17] < data.V17) & (confidence_interval_max[17] > data.V17)]
data = data.loc[(confidence_interval_min[18] < data.V18) & (confidence_interval_max[18] > data.V18)]
data = data.loc[(confidence_interval_min[19] < data.V19) & (confidence_interval_max[19] > data.V19)]
data = data.loc[(confidence_interval_min[20] < data.V20) & (confidence_interval_max[20] > data.V20)]
data = data.loc[(confidence_interval_min[21] < data.V21) & (confidence_interval_max[21] > data.V21)]
data = data.loc[(confidence_interval_min[22] < data.V22) & (confidence_interval_max[22] > data.V22)]
data = data.loc[(confidence_interval_min[23] < data.V23) & (confidence_interval_max[23] > data.V23)]
data = data.loc[(confidence_interval_min[24] < data.V24) & (confidence_interval_max[24] > data.V24)]
data = data.loc[(confidence_interval_min[25] < data.V25) & (confidence_interval_max[25] > data.V25)]
data = data.loc[(confidence_interval_min[26] < data.V26) & (confidence_interval_max[26] > data.V26)]
data = data.loc[(confidence_interval_min[27] < data.V27) & (confidence_interval_max[27] > data.V27)]
data = data.loc[(confidence_interval_min[28] < data.V28) & (confidence_interval_max[28] > data.V28)]






points_train, points_test, labels_train, labels_test = train_test_split(data.iloc[:, :-1], data['Class'], test_size=0.25, random_state=0)


dt = DecisionTreeClassifier(criterion='entropy', max_depth=3)
dt.fit(points_train, labels_train)

prediction = dt.predict(points_test)

print('Оценка качества работы классификатора для масштабированных данных -', format(dt.score(points_test, labels_test)))



points_new = pd.DataFrame({'Time':[171088, 61642, 17065465464565], 'V1':[-0.20903, -1.523, 165654654656.332], 'V2':[0.92065, 1.50554845121, 0.332], 'V3':[0.03362, 0.3723656454, -0.332],
                           'V4':[-0.8386, 2.286821561564, 16456456546.332], 'V5':[0.56709, -0.526518212, 0.332], 'V6':[-0.613278, 0.99855, 0.332],
                           'V7':[0.85105, -1.08752401, 6546456546.0332], 'V8':[0.006625, -0.0272656, -6546456546.372], 'V9':[-0.0017107, -0.53302154, -2.332],
                           'V10':[0.083472, 0.16985211, -35354353454351.022], 'V11':[0.35119, 2.79035558, -53454354354350.002987], 'V12':[0.41123, -2.31655432, -0.0332],
                           'V13':[-0.5612, -0.78253176765, -4353454354351.403232], 'V14':[0.325176, -3.43175401, -5454354351.332], 'V15':[-0.99818, -0.52770160001, 2.043243],
                           'V16':[0.22293, -3.2935404, 2.354332], 'V17':[-0.778826, -3.90150043401, -53453453453450.343232], 'V18':[-0.073345, -2.525440, -1.334322],
                           'V19':[0.327765, 0.79514421, -2.534], 'V20':[0.07312, 0.412802564, -54353454354352.00032], 'V21':[-0.2727, 0.332216401, 1.332],
                           'V22':[-0.5965, 0.493943401, 5435345451.332], 'V23':[0.041588, -0.0801943401, 5454354354354353450.332], 'V24':[-0.432856, -0.25333401, 0.332],
                           'V25':[-0.430749, 0.47779811345301, -4353454351.332], 'V26':[0.1522, 0.991739601, -53453454353452.53532], 'V27':[0.3477, -0.952663401, -2.332],
                           'V28':[0.1412, -0.3903646401, 53453454352.000332]})

#print(data.isnull().sum())  #проверка строк с пустыми ячейками
data = data.dropna()  #содержащих NaN

points_new_after = points_new.loc[(confidence_interval_min[0] < points_new.Time) & (confidence_interval_max[0] > points_new.Time)]
points_new_after = points_new_after.loc[(confidence_interval_min[1] < points_new_after.V1) & (confidence_interval_max[1] > points_new_after.V1)]
points_new_after = points_new_after.loc[(confidence_interval_min[2] < points_new_after.V2) & (confidence_interval_max[2] > points_new_after.V2)]
points_new_after = points_new_after.loc[(confidence_interval_min[3] < points_new_after.V3) & (confidence_interval_max[3] > points_new_after.V3)]
points_new_after = points_new_after.loc[(confidence_interval_min[4] < points_new_after.V4) & (confidence_interval_max[4] > points_new_after.V4)]
points_new_after = points_new_after.loc[(confidence_interval_min[5] < points_new_after.V5) & (confidence_interval_max[5] > points_new_after.V5)]
points_new_after = points_new_after.loc[(confidence_interval_min[6] < points_new_after.V6) & (confidence_interval_max[6] > points_new_after.V6)]
points_new_after = points_new_after.loc[(confidence_interval_min[7] < points_new_after.V7) & (confidence_interval_max[7] > points_new_after.V7)]
points_new_after = points_new_after.loc[(confidence_interval_min[8] < points_new_after.V8) & (confidence_interval_max[8] > points_new_after.V8)]
points_new_after = points_new_after.loc[(confidence_interval_min[9] < points_new_after.V9) & (confidence_interval_max[9] > points_new_after.V9)]
points_new_after = points_new_after.loc[(confidence_interval_min[10] < points_new_after.V10) & (confidence_interval_max[10] > points_new_after.V10)]
points_new_after = points_new_after.loc[(confidence_interval_min[11] < points_new_after.V11) & (confidence_interval_max[11] > points_new_after.V11)]
points_new_after = points_new_after.loc[(confidence_interval_min[12] < points_new_after.V12) & (confidence_interval_max[12] > points_new_after.V12)]
points_new_after = points_new_after.loc[(confidence_interval_min[13] < points_new_after.V13) & (confidence_interval_max[13] > points_new_after.V13)]
points_new_after = points_new_after.loc[(confidence_interval_min[14] < points_new_after.V14) & (confidence_interval_max[14] > points_new_after.V14)]
points_new_after = points_new_after.loc[(confidence_interval_min[15] < points_new_after.V15) & (confidence_interval_max[15] > points_new_after.V15)]
points_new_after = points_new_after.loc[(confidence_interval_min[16] < points_new_after.V16) & (confidence_interval_max[16] > points_new_after.V16)]
points_new_after = points_new_after.loc[(confidence_interval_min[17] < points_new_after.V17) & (confidence_interval_max[17] > points_new_after.V17)]
points_new_after = points_new_after.loc[(confidence_interval_min[18] < points_new_after.V18) & (confidence_interval_max[18] > points_new_after.V18)]
points_new_after = points_new_after.loc[(confidence_interval_min[19] < points_new_after.V19) & (confidence_interval_max[19] > points_new_after.V19)]
points_new_after = points_new_after.loc[(confidence_interval_min[20] < points_new_after.V20) & (confidence_interval_max[20] > points_new_after.V20)]
points_new_after = points_new_after.loc[(confidence_interval_min[21] < points_new_after.V21) & (confidence_interval_max[21] > points_new_after.V21)]
points_new_after = points_new_after.loc[(confidence_interval_min[22] < points_new_after.V22) & (confidence_interval_max[22] > points_new_after.V22)]
points_new_after = points_new_after.loc[(confidence_interval_min[23] < points_new_after.V23) & (confidence_interval_max[23] > points_new_after.V23)]
points_new_after = points_new_after.loc[(confidence_interval_min[24] < points_new_after.V24) & (confidence_interval_max[24] > points_new_after.V24)]
points_new_after = points_new_after.loc[(confidence_interval_min[25] < points_new_after.V25) & (confidence_interval_max[25] > points_new_after.V25)]
points_new_after = points_new_after.loc[(confidence_interval_min[26] < points_new_after.V26) & (confidence_interval_max[26] > points_new_after.V26)]
points_new_after = points_new_after.loc[(confidence_interval_min[27] < points_new_after.V27) & (confidence_interval_max[27] > points_new_after.V27)]
points_new_after = points_new_after.loc[(confidence_interval_min[28] < points_new_after.V28) & (confidence_interval_max[28] > points_new_after.V28)]

abnormal_points_new = points_new[~points_new.apply(tuple, 1).isin(points_new_after.apply(tuple, 1))]
print('Aномалия!\n', abnormal_points_new)


prediction = dt.predict(points_new_after)
print('Результат классификации новых меток:\n', points_new_after.assign(predict=prediction))


plt.figure()
tree.plot_tree(dt)
plt.show()