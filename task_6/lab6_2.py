from PIL import Image
import numpy as np
import pandas as pd
from colorutils import Color
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

dictionaryRGB = {}  # будем собирать сюда границы доверительных интервалов
dictionaryHSV = {}  # будем собирать сюда границы доверительных интервалов

def confInterval(nameSign, _data, dictionary):
    mean = np.mean(_data[nameSign])
    std = np.std(_data[nameSign])

    confIntervalMin = mean - (3 * std)
    confIntervalMax = mean + (3 * std)

    dictionary[nameSign] = [confIntervalMin, confIntervalMax]
    _data = _data[(_data[nameSign] >= confIntervalMin) & (_data[nameSign] <= confIntervalMax)]

    return _data


def preprocessing(_data, col1, col2, col3, dictionary):
    # уравниваем число представителей классов
    variousValues = _data["label"].unique().tolist()
    myList = list()

    for value in variousValues:
        myList.append(len(_data[_data.label == value]))

    minValue = min(myList)
    data = pd.DataFrame()

    for value in variousValues:
        print(_data[_data.label == value].sample(minValue))
        data = pd.concat([data, _data[_data.label == value].sample(minValue)])
        print(len(data[data.label == value]))

    # доверительный интервал
    data = confInterval(col1, data, dictionary)
    data = confInterval(col2, data, dictionary)
    data = confInterval(col3, data, dictionary)
    return data


def classifier(data):
    points_train, points_test, labels_train, labels_test = \
        train_test_split(data.iloc[:, :-1],
                         data['label'], test_size=0.25, random_state=0)

    gnb = GaussianNB()
    gnb.fit(points_train, labels_train)  # учим классификатор

    prediction = gnb.predict(points_test)
    # print(prediction)
    print(points_test.assign(predict=prediction))

    print(format(gnb.score(points_test, labels_test)))

    return gnb


def getNewPhoto(photo):
    _dfRGB = pd.DataFrame(columns=['R', 'G', 'B'])
    _dfHSV = pd.DataFrame(columns=['H', 'S', 'V'])
    picture = Image.open(photo)

    width, height = picture.size
    print(width, height)

    for x in range(width):
        dictsRGB = []
        dictsHSV = []
        for y in range(height):
            pixelPicture = picture.getpixel((x, y))

            R, G, B = pixelPicture
            c = Color((R, G, B))
            H, S, V = c.hsv

            dictsRGB = dictsRGB + [{'R': R, 'G': G, 'B': B}]
            dictsHSV = dictsHSV + [{'H': H, 'S': S, 'V': V}]

        # print(x)
        _dfRGB = _dfRGB.append(dictsRGB, ignore_index=True, sort=False)
        _dfHSV = _dfHSV.append(dictsHSV, ignore_index=True, sort=False)

    return _dfRGB, _dfHSV


def showNewPhoto(data, photo, RGB):
    picture = Image.open(photo)

    width, height = picture.size
    print(width, height)

    i = 0
    for x in range(width):
        for y in range(height):

            if not RGB:
                H, S, V = data.iloc[i]["H"], data.iloc[i]["S"], data.iloc[i]["V"]
                R, G, B = Color(hsv=(H, S, V)).rgb
                newColor = int(R), int(G), int(B)

            else:
                newColor = data.iloc[i]["R"], data.iloc[i]["G"], data.iloc[i]["B"]

            i += 1
            picture.putpixel((x, y), newColor)

    picture.show()

def crossValidation(_data):
    gnb = GaussianNB()
    scores = cross_val_score(gnb, _data.iloc[:, :-1], _data['label'], cv=10)

    print(scores)
    print(scores.mean())
    print(scores.std())


###########################################################################################

myPhotos = [["1000s_d_850.jpg", "1000s_d_850_whte.jpg"],
            ["755869560832595.jpg", "755869560832595_white.jpg"],
            ["755895443908228.jpg", "755895443908228_white.jpg"], ["thumb_Smirnova_Marina.jpg", "thumb_Smirnova_Marina_white.jpg"]]


dfRGB = pd.DataFrame(columns=['R', 'G', 'B', 'label'])
dfHSV = pd.DataFrame(columns=['H', 'S', 'V', 'label'])

for photo in myPhotos:
    picture = Image.open(photo[0])
    pictureWhite = Image.open(photo[1])

    width, height = picture.size
    print(width, height)

    for x in range(width):
        dictsRGB = []
        dictsHSV = []
        for y in range(height):
            pixelPicture = picture.getpixel((x, y))
            pixelPictureWhite = pictureWhite.getpixel((x, y))

            # print(pixelPicture)

            R, G, B = pixelPicture
            c = Color((R, G, B))
            H, S, V = c.hsv

            if pixelPictureWhite == (255, 255, 255):
                dictsRGB = dictsRGB + [{'R': R, 'G': G, 'B': B, 'label': '0'}]
                dictsHSV = dictsHSV + [{'H': H, 'S': S, 'V': V, 'label': '0'}]

            else:
                dictsRGB = dictsRGB + [{'R': R, 'G': G, 'B': B, 'label': '1'}]
                dictsHSV = dictsHSV + [{'H': H, 'S': S, 'V': V, 'label': '1'}]

        # print(x)
        dfRGB = dfRGB.append(dictsRGB, ignore_index=True, sort=False)
        dfHSV = dfHSV.append(dictsHSV, ignore_index=True, sort=False)

# удаляем повторы
dfRGB = dfRGB.drop_duplicates()
dfHSV = dfHSV.drop_duplicates()

print(dfRGB)
print(dfHSV)

dfRGB = preprocessing(dfRGB, "R", "G", "B", dictionaryRGB)  # предобработка
gnbRGB = classifier(dfRGB)

dfHSV = preprocessing(dfHSV, "H", "S", "V", dictionaryHSV)  # предобработка
gnbHSV = classifier(dfHSV)

newDataRGB, newDataHSV = getNewPhoto("гудков.jpg")

#################################

prediction = gnbRGB.predict(newDataRGB)
newDataRGB = newDataRGB.assign(predict=prediction)

df = pd.DataFrame()
# выявляем аномальные объекты и отбрасываем их
for d in dictionaryRGB:
    df = pd.concat([df, newDataRGB[(newDataRGB[d] < dictionaryRGB[d][0]) | (newDataRGB[d] > dictionaryRGB[d][1])]])

if not df.empty:
    print("Аномальные объекты:")
    print(df)
else:
    print("Аномальные объекты не выявлены")


newDataRGB.loc[newDataRGB['predict'] == "1", 'R'] = 255
newDataRGB.loc[newDataRGB['predict'] == "1", ['G', 'B']] = 0

print(newDataRGB)

showNewPhoto(newDataRGB, "гудков.jpg", True)

##################################

prediction = gnbHSV.predict(newDataHSV)
newDataHSV = newDataHSV.assign(predict=prediction)

df = pd.DataFrame()
# выявляем аномальные объекты и отбрасываем их
for d in dictionaryHSV:
    df = pd.concat([df, newDataHSV[(newDataHSV[d] < dictionaryHSV[d][0]) | (newDataHSV[d] > dictionaryHSV[d][1])]])

if not df.empty:
    print("Аномальные объекты:")
    print(df)
else:
    print("Аномальные объекты не выявлены")


newDataHSV.loc[newDataHSV['predict'] == "1", 'H'] = 0.0
newDataHSV.loc[newDataHSV['predict'] == "1", ['S', 'V']] = 1.0

print(newDataHSV)

showNewPhoto(newDataHSV, "гудков.jpg", False)

##################################

crossValidation(dfRGB)
crossValidation(dfHSV)
