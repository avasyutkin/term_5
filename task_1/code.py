"""import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from array import *
from scipy.signal import argrelextrema
import math

data = pd.read_csv("D:/tasks/task_1/creditcard.csv")
data = data.loc[(data.V1 > -6)]
data = data.loc[(-5 < data.V2)]
data = data.loc[(data.V2 < 5)]
data = data.loc[(-5 < data.V3)]
data = data.loc[(data.V3 < 5)]

print(data['V4'].min(), data['V4'].max(), data['V4'].max()-data['V4'].min())

k = int((1.25 * len(data) ** 0.4 - 0.55 * len(data) ** 0.4) / 4 + 0.55 * len(data) ** 0.4)
if k % 2 == 0:
    k = k + 1
print('k', k, 1.25 * len(data) ** 0.4, 0.55 * len(data) ** 0.4)

(n, bins, patches) = plt.hist(data["V4"], bins=k)
plt.show()

#s = pd.Series(data['Amount']).sort_values()
#bins = np.concatenate(([-np.inf], np.arange(data['Amount'].min(), data['Amount'].max(), (data['Amount'].max()-data['Amount'].min())/k), [np.inf]))
#print(s.groupby(pd.cut(data['Amount'], bins), observed=True).apply(lambda x: x.to_list()))

array_local_minimum_index = argrelextrema(n, np.less)
print("Индексы локальных минимумов -", array_local_minimum_index[0])

array_math_expectation = array('f', [])
array_local_minimum_index_counter = array_local_minimum_index[0]
sum = 0
count_bins = 0
counter = 0

for i in n:
    if counter < len(array_local_minimum_index_counter):
        if i != n[array_local_minimum_index_counter[counter]]:
            sum = sum + (bins[count_bins + 1] - bins[count_bins]) * i / (len(data["Amount"]) * 2)
            count_bins = count_bins + 1
        else:
            counter = counter + 1
            count_bins = count_bins + 1
            array_math_expectation.append(sum)
            sum = 0
    else:
        sum = sum + (bins[count_bins + 1] - bins[count_bins]) * i / (len(data["Amount"]) * 2)

array_math_expectation.append(sum)

print(n[14], len(bins))
print(array_math_expectation, len(array_math_expectation))


array_disp = array('f', [])
array_3sigm = array('f', [])
counter = 0
count_bins = 0
disp = 0

for i in n:
    if counter < len(array_local_minimum_index_counter):
        if i != n[array_local_minimum_index_counter[counter]]:
            disp = disp + (i / len(data["Amount"])) * ((((bins[count_bins + 1] - bins[count_bins]) / 2) - array_math_expectation[counter]) ** 2)
            count_bins = count_bins + 1
        else:
            array_disp.append(disp)
            array_3sigm.append(array_math_expectation[counter] - math.sqrt(array_disp[counter]) * 3)
            array_3sigm.append(array_math_expectation[counter] + math.sqrt(array_disp[counter]) * 3)
            disp = 0
            counter = counter + 1
            count_bins = count_bins + 1
    else:
        disp = disp + (i / len(data["Amount"])) * ((((bins[count_bins + 1] - bins[count_bins]) / 2) - array_math_expectation[counter]) ** 2)

array_disp.append(disp)
array_3sigm.append(array_math_expectation[counter] - math.sqrt(array_disp[counter]) * 3)
array_3sigm.append(array_math_expectation[counter] + math.sqrt(array_disp[counter]) * 3)



#print("Дисперсия для каждой области однородности -", array_disp)
print("Индексы локальных минимумов -", array_local_minimum_index[0])
print("Математическое ожидание для каждой области однородности -", array_math_expectation)
print("Доверительный интервал для каждой области однородности -", array_3sigm)

"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from array import *
from scipy.signal import argrelextrema
import math
import xlrd

data_t = xlrd.open_workbook('D:/tasks/task_1/histogram_test.xlsx')
sheet = data_t.sheet_by_index(0)
data = [sheet.col_values(11)][0][1:]
#print(data)

#print(len(data))

k = int((1.25 * len(data) ** 0.4 - 0.55 * len(data) ** 0.4) / 2.5 + 0.55 * len(data) ** 0.4)
if k % 2 == 0:
    k = k + 1
print('k', k, 1.25 * len(data) ** 0.4, 0.55 * len(data) ** 0.4)
k=3
(n, bins, patches) = plt.hist(data, bins=k, density=True)
plt.show()

#s = pd.Series(data['Amount']).sort_values()
#bins = np.concatenate(([-np.inf], np.arange(data['Amount'].min(), data['Amount'].max(), (data['Amount'].max()-data['Amount'].min())/k), [np.inf]))
#print(s.groupby(pd.cut(data['Amount'], bins), observed=True).apply(lambda x: x.to_list()))


array_local_minimum_index = argrelextrema(n, np.less)
print("локальные минимумы -",bins[array_local_minimum_index[0]] ,"середина -", (bins[array_local_minimum_index[0]+1]-bins[array_local_minimum_index[0]])/2+bins[array_local_minimum_index[0]])
"""
array_math_expectation = array('f', [])
array_local_minimum_index_counter = array_local_minimum_index[0]
sum = 0
count_bins = 0
counter = 0

for i in n:
    if counter < len(array_local_minimum_index_counter):
        if i != n[array_local_minimum_index_counter[counter]]:
            sum = sum + (bins[count_bins + 1] - bins[count_bins]) * i / (len(data) * 2)
            count_bins = count_bins + 1
        else:
            counter = counter + 1
            count_bins = count_bins + 1
            array_math_expectation.append(sum)
            sum = 0
    else:
        sum = sum + (bins[count_bins + 1] - bins[count_bins]) * i / (len(data) * 2)

array_math_expectation.append(sum)

print(len(bins))
print(array_math_expectation, len(array_math_expectation))


array_disp = array('f', [])
array_3sigm = array('f', [])
counter = 0
count_bins = 0
disp = 0

for i in n:
    if counter < len(array_local_minimum_index_counter):
        if i != n[array_local_minimum_index_counter[counter]]:
            disp = disp + (i / len(data)) * ((((bins[count_bins + 1] - bins[count_bins]) / 2) - array_math_expectation[counter]) ** 2)
            count_bins = count_bins + 1
        else:
            array_disp.append(disp)
            array_3sigm.append(array_math_expectation[counter] - math.sqrt(array_disp[counter]) * 3)
            array_3sigm.append(array_math_expectation[counter] + math.sqrt(array_disp[counter]) * 3)
            disp = 0
            counter = counter + 1
            count_bins = count_bins + 1
    else:
        disp = disp + (i / len(data)) * ((((bins[count_bins + 1] - bins[count_bins]) / 2) - array_math_expectation[counter]) ** 2)

array_disp.append(disp)
array_3sigm.append(array_math_expectation[counter] - math.sqrt(array_disp[counter]) * 3)
array_3sigm.append(array_math_expectation[counter] + math.sqrt(array_disp[counter]) * 3)



print("Дисперсия для каждой области однородности -", array_disp)
print("Индексы локальных минимумов -", array_local_minimum_index[0])
print("Математическое ожидание для каждой области однородности -", array_math_expectation)
print("Доверительный интервал для каждой области однородности -", array_3sigm)
"""