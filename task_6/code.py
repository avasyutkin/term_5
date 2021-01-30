from PIL import ImageColor, Image, ImageDraw
from array import *
from scipy.signal import argrelextrema
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import matplotlib.pyplot as plt
from colorutils import Color
import random
from sklearn.model_selection import cross_val_score

"""
def get_label_preprocessing(df_color, file_img):
    array_gray_pix = array('f', [])
    with open(file_img) as file:
        for line in file:
            if "TD  BGCOLOR" in line:
                color_hex = line[19:26]
                color_rgb = ImageColor.getrgb(color_hex)
                gray = 0.299 * color_rgb[0] + 0.587 * color_rgb[1] + 0.114 * color_rgb[2]
                array_gray_pix.append(gray)

    file.close()

    k = int((1.25 * len(array_gray_pix) ** 0.4 - 0.55 * len(array_gray_pix) ** 0.4) / 2 + 0.55 * len(array_gray_pix) ** 0.4)   # высчитываем оптимальное число интервалов
    if k % 2 == 0:
        k = k + 1

    plt.figure()
    (n, bins, patches) = plt.hist(array_gray_pix, bins=k, density=True)
    #plt.show()

    array_local_minimum_index = argrelextrema(n, np.less)

    array_local_minimum_index_counter = array_local_minimum_index[0]
    array_sum_of_densities = array('f', [])
    counter = 0
    sum_of_one_area = 0

    for i in n:
        if counter < len(array_local_minimum_index_counter):
            if i != n[array_local_minimum_index_counter[counter]]:
                sum_of_one_area = sum_of_one_area + i
            else:
                counter = counter + 1
                array_sum_of_densities.append(sum_of_one_area)
                sum_of_one_area = 0
        else:
            sum_of_one_area = sum_of_one_area + i
            array_sum_of_densities.append(sum_of_one_area)

    array_sum_of_densities_without_first_max = array_sum_of_densities[:]
    array_sum_of_densities_without_first_max.remove(max(array_sum_of_densities_without_first_max))
    array_sum_of_densities_without_first_max.remove(max(array_sum_of_densities_without_first_max))

    array_bins_index_of_second_area = array('i', [])
    index_of_second_largest_area = array_sum_of_densities.index(max(array_sum_of_densities_without_first_max))

    n = array('f', n)
    counter = 0
    for i in n:
        if counter < len(array_local_minimum_index_counter):
            if i != n[array_local_minimum_index_counter[counter]]:
                if counter == index_of_second_largest_area:
                    array_bins_index_of_second_area.append(n.index(i))
            else:
                counter = counter + 1

    array_bins_of_second_area = bins[array_bins_index_of_second_area]
    bins_of_second_area_min = min(array_bins_of_second_area)
    bins_of_second_area_max = max(array_bins_of_second_area)

    labels = []
    file_finished = open('D:/tasks/task_6/' + str(random.getrandbits(3)) + 'finished.html', 'w')
    with open(file_img) as file:
        for line in file:
            if "TD  BGCOLOR" in line:
                color_hex = line[19:26]
                color_rgb = ImageColor.getrgb(color_hex)
                gray = 0.299 * color_rgb[0] + 0.587 * color_rgb[1] + 0.114 * color_rgb[2]
                if gray >= bins_of_second_area_min and gray <= bins_of_second_area_max:
                    line = '<td bgcolor="#ff0000">'
                    file_finished.write(line)
                    labels.append(1)
                else:
                    file_finished.write(line)
                    labels.append(0)
            else:
                file_finished.write(line)
    file.close()
    file_finished.close()
    df_color['label'] = labels

    return df_color
"""
def html_to_df_color(df_color, file, mode):
    img = Image.open(file)
    pix = img.load()
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            color = pix[x, y][0], pix[x, y][1], pix[x, y][2]
            if mode == 'hsv':
                color = Color(color).hsv
            df_color.loc[len(df_color)] = color
            print(color)

    print(df_color)
    return df_color

def preprocessing(df_color, mode):
    df_color.drop_duplicates()
    df_color = df_color[(df_color.label == 1) | (df_color.label == 0).sample(n=len(df_color[(df_color.label == 1)]))]

    df_color_without_class = df_color.iloc[:, :3]

    confidence_interval_min = np.mean(df_color_without_class)-3*np.std(df_color_without_class)
    confidence_interval_max = np.mean(df_color_without_class)+3*np.std(df_color_without_class)

    if mode == 'rgb':
        df_color = df_color.loc[(confidence_interval_min[0] < df_color.r) & (confidence_interval_max[0] > df_color.r)]
        df_color = df_color.loc[(confidence_interval_min[1] < df_color.g) & (confidence_interval_max[1] > df_color.g)]
        df_color = df_color.loc[(confidence_interval_min[2] < df_color.b) & (confidence_interval_max[2] > df_color.b)]
    if mode == 'hsv':
        df_color = df_color.loc[(confidence_interval_min[0] < df_color.h) & (confidence_interval_max[0] > df_color.h)]
        df_color = df_color.loc[(confidence_interval_min[1] < df_color.s) & (confidence_interval_max[1] > df_color.s)]
        df_color = df_color.loc[(confidence_interval_min[2] < df_color.v) & (confidence_interval_max[2] > df_color.v)]

    return df_color, confidence_interval_min, confidence_interval_max

def train_classifier(df_color, df_color_new):
    points_train, points_test, labels_train, labels_test = train_test_split(df_color.iloc[:, :-1], df_color['label'], test_size=0.25, random_state=0)
    nbcla = GaussianNB()
    nbcla.fit(points_train, labels_train)

    prediction = nbcla.predict(points_test)

    print('Оценка качества работы классификатора -', format(nbcla.score(points_test, labels_test)))
    print(df_color_new)
    prediction = nbcla.predict(df_color_new)
    print(df_color_new, prediction)
    df_color_new['label'] = prediction
    print(df_color_new)
    return df_color_new

def search_anomaly(df_color, confidence_interval_min, confidence_interval_max, mode):
    if mode == 'rgb':
        df_color_after = df_color.loc[(confidence_interval_min[0] < df_color.r) & (confidence_interval_max[0] > df_color.r)]
        df_color_after = df_color_after.loc[(confidence_interval_min[1] < df_color_after.g) & (confidence_interval_max[1] > df_color_after.g)]
        df_color_after = df_color_after.loc[(confidence_interval_min[2] < df_color_after.b) & (confidence_interval_max[2] > df_color_after.b)]
    if mode == 'hsv':
        df_color_after = df_color.loc[(confidence_interval_min[0] < df_color.h) & (confidence_interval_max[0] > df_color.h)]
        df_color_after = df_color_after.loc[(confidence_interval_min[1] < df_color_after.s) & (confidence_interval_max[1] > df_color_after.s)]
        df_color_after = df_color_after.loc[(confidence_interval_min[2] < df_color_after.v) & (confidence_interval_max[2] > df_color_after.v)]

    abnormal_points = df_color[~df_color.apply(tuple, 1).isin(df_color_after.apply(tuple, 1))]
    print('Aномалия!\n', abnormal_points)

    return df_color

def create_file_after_recognition(prediction, file, mode):
    img = Image.open(file)
    draw = ImageDraw.Draw(img)
    i = 0
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            if prediction[i] == 1:
                draw.point((x, y), (255, 0, 0))
                i = i + 1
            else:
                i = i + 1
    img.save("result_" + mode + ".jpg", "JPEG")


def cross_validation(df_color):
    nbcla = GaussianNB()
    scores = cross_val_score(nbcla, df_color.iloc[:, :-1], df_color['label'], cv=10)
    print(scores)
    print(scores.mean())


def get_label_preprocessing_img(file):
    img = Image.open(file)
    pix = img.load()
    array_pix_label = array('i', [])
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            if pix[x, y][0] == 255 & pix[x, y][1] == 255 & pix[x, y][2] == 255:
                array_pix_label.append(0)
            else:
                array_pix_label.append(1)

    return array_pix_label








file_1 = 'D:/tasks/task_6/1000s_d_850_whte.jpg'
file_2 = 'D:/tasks/task_6/755869560832595_white.jpg'
file_3 = 'D:/tasks/task_6/755895443908228_white.jpg'
file_4 = 'D:/tasks/task_6/thumb_Smirnova_Marina_white.jpg'
file_5 = 'D:/tasks/task_6/teen2.jpg'


"""

### создание DataFrame из html ###

df_color_rgb = pd.DataFrame(columns=['r', 'g', 'b'])
df_color_rgb = html_to_df_color(df_color_rgb, file_1, 'rgb')
df_color_rgb = html_to_df_color(df_color_rgb, file_2, 'rgb')
df_color_rgb = html_to_df_color(df_color_rgb, file_3, 'rgb')
df_color_rgb = html_to_df_color(df_color_rgb, file_4, 'rgb')
df_color_rgb.to_csv("D:/tasks/task_6/color_rgb_white.csv", columns=['r', 'g', 'b'])

df_color_hsv = pd.DataFrame(columns=['h', 's', 'v'])
df_color_hsv = html_to_df_color(df_color_hsv, file_1, 'hsv')
df_color_hsv = html_to_df_color(df_color_hsv, file_2, 'hsv')
df_color_hsv = html_to_df_color(df_color_hsv, file_3, 'hsv')
df_color_hsv = html_to_df_color(df_color_hsv, file_4, 'hsv')
df_color_hsv.to_csv("D:/tasks/task_6/color_hsv_white.csv", columns=['h', 's', 'v'])

df_color_rgb = pd.read_csv("D:/tasks/task_6/color_rgb_white.csv")
df_color_hsv = pd.read_csv("D:/tasks/task_6/color_hsv_white.csv")


### добавление label к созданному ранее DataFrame (task_2) и запись в csv ###

array_label = get_label_preprocessing_img(file_1)
array_label = [*array_label, *(get_label_preprocessing_img(file_2))]
array_label = [*array_label, *(get_label_preprocessing_img(file_3))]
array_label = [*array_label, *(get_label_preprocessing_img(file_4))]



df_color_hsv['label'] = array_label
df_color_rgb['label'] = array_label


df_color_rgb.to_csv("D:/tasks/task_6/color_rgb_white_with_label.csv", columns=['r', 'g', 'b', 'label'])
df_color_hsv.to_csv("D:/tasks/task_6/color_hsv_white_with_label.csv", columns=['h', 's', 'v', 'label'])
"""
### считываение записанного выше csv ###
df_color_hsv_with_label = pd.read_csv("D:/tasks/task_6/color_hsv_white_with_label.csv")
df_color_hsv_with_label = df_color_hsv_with_label.iloc[:, 1:5]

df_color_rgb_with_label = pd.read_csv("D:/tasks/task_6/color_rgb_white_with_label.csv")
df_color_rgb_with_label = df_color_rgb_with_label.iloc[:, 1:5]


df_color_hsv_with_label_before_preprocessing = pd.read_csv("D:/tasks/task_6/color_hsv.csv")
df_color_hsv_with_label_before_preprocessing = df_color_hsv_with_label_before_preprocessing.iloc[:, 1:5]

df_color_rgb_with_label_before_preprocessing = pd.read_csv("D:/tasks/task_6/color_rgb.csv")
df_color_rgb_with_label_before_preprocessing = df_color_rgb_with_label_before_preprocessing.iloc[:, 1:5]

df_color_rgb_with_label_before_preprocessing['label'] = df_color_rgb_with_label['label']
df_color_hsv_with_label_before_preprocessing['label'] = df_color_hsv_with_label['label']

df_color_rgb_with_label_before_preprocessing.to_csv("D:/tasks/task_6/color_rgb_with_label.csv", columns=['r', 'g', 'b', 'label'])
df_color_hsv_with_label_before_preprocessing.to_csv("D:/tasks/task_6/color_hsv_with_label.csv", columns=['h', 's', 'v', 'label'])

df_color_hsv_with_label_before_preprocessing = pd.read_csv("D:/tasks/task_6/color_hsv_with_label.csv")
df_color_rgb_with_label_before_preprocessing = pd.read_csv("D:/tasks/task_6/color_rgb_with_label.csv")

### предобработка и определение доверительного интервала ###
df_color_hsv_with_label_after_preprocessing = preprocessing(df_color_hsv_with_label_before_preprocessing, 'hsv')  #[0], min[1], max[2]
df_color_rgb_with_label_after_preprocessing = preprocessing(df_color_rgb_with_label_before_preprocessing, 'rgb')  #[0], min[1], max[2]
"""
### создание DataFrame для нового html ###
df_color_rgb_new = pd.DataFrame(columns=['r', 'g', 'b'])
df_color_rgb_new = html_to_df_color(df_color_rgb_new, file_5, 'rgb')

df_color_hsv_new = pd.DataFrame(columns=['h', 's', 'v'])
df_color_hsv_new = html_to_df_color(df_color_hsv_new, file_5, 'hsv')

### создание csv для созданного выше DataFrame ###
df_color_rgb_new.to_csv("D:/tasks/task_6/color_rgb_new_teenager2.csv", columns=['r', 'g', 'b'])
df_color_hsv_new.to_csv("D:/tasks/task_6/color_hsv_new_teenager2.csv", columns=['h', 's', 'v'])
"""
### считывание созданного выше csv ###
df_color_hsv_new = pd.read_csv("D:/tasks/task_6/color_hsv_new_teenager2.csv")
df_color_hsv_new = df_color_hsv_new.iloc[:, 1:4]

df_color_rgb_new = pd.read_csv("D:/tasks/task_6/color_rgb_new_teenager2.csv")
df_color_rgb_new = df_color_rgb_new.iloc[:, 1:4]

print(df_color_rgb_new, df_color_hsv_new)


### поиск аномалий среди новых данных ###
df_color_hsv_new_after_search_anomaly = search_anomaly(df_color_hsv_new, df_color_hsv_with_label_after_preprocessing[1], df_color_hsv_with_label_after_preprocessing[2], 'hsv')
df_color_rgb_new_after_search_anomaly = search_anomaly(df_color_rgb_new, df_color_rgb_with_label_after_preprocessing[1], df_color_rgb_with_label_after_preprocessing[2], 'rgb')

### классификация новых объектов (создание, обучение классификатора и классификация) ###
df_color_hsv_new_after_classification = train_classifier(df_color_hsv_with_label_after_preprocessing[0].iloc[:, 1:5], df_color_hsv_new_after_search_anomaly)
df_color_rgb_new_after_classification = train_classifier(df_color_rgb_with_label_after_preprocessing[0].iloc[:, 1:5], df_color_rgb_new_after_search_anomaly)

### запись в html классифицированных пикселей ###
create_file_after_recognition(df_color_hsv_new_after_classification['label'].to_list(), file_5, 'hsv')
create_file_after_recognition(df_color_rgb_new_after_classification['label'].to_list(), file_5, 'rgb')

### кросс-валидация ###
df_color_hsv_with_label_before_preprocessing = pd.read_csv("D:/tasks/task_6/color_hsv_with_label.csv")
df_color_rgb_with_label_before_preprocessing = pd.read_csv("D:/tasks/task_6/color_rgb_with_label.csv")

df_color_hsv_with_label_after_preprocessing = preprocessing(df_color_hsv_with_label_before_preprocessing, 'hsv')
df_color_rgb_with_label_after_preprocessing = preprocessing(df_color_rgb_with_label_before_preprocessing, 'rgb')

cross_validation(df_color_hsv_with_label_after_preprocessing[0])
cross_validation(df_color_rgb_with_label_after_preprocessing[0])


