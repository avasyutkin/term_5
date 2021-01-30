import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

df0_1 = pd.read_csv('creditcard.csv')

df0 = df0_1[(df0_1.Class == 1) | (df0_1.Class == 0).sample(n=492)]

print(df0, len(df0[(df0.Class == 1)]))

features = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"]
df0_x = df0.loc[:, features].values
df0_y = df0.loc[:, ["Class"]].values
#print(df0_x, df0_y)

ss = StandardScaler()
df1 = df0_x.copy()
df1 = ss.fit_transform(df1) #масштабируем без метки

pca = PCA(n_components=2)
df2 = df1.copy()
df2 = pca.fit_transform(df2)
#df2.columns = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7', 'pc8', 'pc9', 'pc10', 'pc11', 'pc12', 'pc13', 'pc14', 'pc15', 'pc16', 'pc17', 'pc18', 'pc19', 'pc20', 'pc21', 'pc22', 'pc23', 'pc24', 'pc25', 'pc26', 'pc27', 'pc28', 'pc29', 'pc30', 'Class']
#df2.columns = ['pc1', 'pc2']
pcdf = pd.DataFrame(data=df2, columns=['pc1', 'pc2'])
pcdf1 = pd.concat([pcdf, df0[['Class']]], axis=1)
print(pcdf.values, 'values')

plt.figure()
plt.grid()
plt.scatter(pcdf['pc1'], pcdf['pc2'], c=df0_y, edgecolor='black', lw=.4, cmap='jet', alpha=.1)
plt.title("Rotated projected points [2D]")
plt.xlabel("pc1")
plt.ylabel("pc2")
plt.axis('equal')
plt.show()

df3 = pcdf.copy()
#df3['pc2'] = 0 #!
df3 = pca.inverse_transform(df3)

df3 = pd.DataFrame(data=df3, columns=["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"])

plt.matshow(pca.components_[0:2, 0:30])
cb=plt.colorbar()
plt.gca().xaxis.tick_bottom()
plt.yticks(range(len(pcdf1.columns) -1), pcdf1.iloc[:, :-1].columns)
plt.xticks(range(len(df0.columns) - 1), df0.iloc[:, :-1].columns)
plt.title("Main features")
plt.show()

print(pca.components_, 'components') #!

pcadf = pd.DataFrame(data=pca.components_, columns=["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11", "V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"])

#print(pcadf)

for i in pcadf:
    if (-0.05 < pcadf[i][0] < 0.05 and -0.03 < pcadf[i][1] < 0.054):
        print(i)
        print(pcadf[i])

















"""
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)


targets = ['0', '1']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = pcdf['Class'] == target
    ax.scatter(pcdf.loc[indicesToKeep, 'pc1']
               , pcdf.loc[indicesToKeep, 'pc2']
               , c = color
               , s = 50)

ax.legend(targets)
ax.grid()
plt.show()




"""














"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df0 = pd.read_csv('creditcard.csv')
print(df0)

plt.figure()
plt.grid()
plt.scatter(df0['x1'], df0['x2'], c=df0['label'], edgecolor='black', lw=.6, cmap='jet')
plt.title("Input points [2D]")
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis('equal')
plt.show()


ss = StandardScaler()
df1 = df0.copy()
df1.iloc[:, :-1] = ss.fit_transform(df1.iloc[:, :-1])
plt.figure()
plt.grid()
plt.scatter(df1['x1'], df1['x2'], c=df1['label'], edgecolor='black', cmap='jet')
plt.title("Normalized input points [2D]")
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis('equal')
plt.show()


pca = PCA(svd_solver='full') #n_components=2
df2 = df1.copy()
df2.iloc[:, :-1] = pca.fit_transform(df2.iloc[:, :-1])
df2.columns = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7', 'pc8', 'pc9', 'pc10', 'pc11', 'pc12', 'pc13', 'pc14', 'pc15', 'pc16', 'pc17', 'pc18', 'pc19', 'pc20', 'pc21', 'pc22', 'pc23', 'pc24', 'pc25', 'pc26', 'pc27', 'pc28', 'pc29', 'pc30', 'Class']
print(pca.components_) #!

plt.figure()
plt.grid()
plt.scatter(df2['pc1'], df2['pc2'], c=df2['label'], edgecolor='black', lw=.6, cmap='jet')
plt.title("Rotated projected points [2D]")
plt.xlabel("pc1")
plt.ylabel("pc2")
plt.axis('equal')
plt.show()


df3 = df2.copy()
df3['pc2'] = 0 #!
df3.iloc[:, :-1] = pca.inverse_transform(df3.iloc[:, :-1])
df3.columns = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7', 'pc8', 'pc9', 'pc10', 'pc11', 'pc12', 'pc13', 'pc14', 'pc15', 'pc16', 'pc17', 'pc18', 'pc19', 'pc20', 'pc21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'Class']
plt.figure()
plt.grid()
plt.scatter(df3['x1'], df3['x2'], c=df3['label'], edgecolor='black', lw=.6, cmap='jet')
plt.title("PC1 in the original feature space [2D]")
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis('equal')
plt.show()

plt.matshow(pca.components_)
plt.matshow(pca.components_[0:2, 0:2])
plt.colorbar()
plt.gca().xaxis.tick_bottom()
plt.xticks(range(len(df0.columns) - 1), df0.iloc[:, :-1].columns, rotation=90)
plt.yticks(range(len(df2.columns) - 1), df2.iloc[:, :-1].columns)
plt.title("Main features")
i, k = plt.ylim() #
plt.ylim(i+0.5, k-0.5) #
plt.show()
"""