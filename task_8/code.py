import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import math

words = ['aes', 'encryption', 'standard', 'cmp', 'des', 'certificate', 'management', 'protocol',
         'digital', 'diffie-hellman', 'dns', 'domain', 'DSA', 'digital', 'ecdh', 'elliptic', 'algorithm',
         'hash', 'authentication', 'internet', 'ip', 'ipsec', 'md2', 'md4', 'md5', 'identifier', 'object',
         'privacy', 'cryptography', 'public', 'key', 'rsa', 'secure', 'electronic', 'transaction',
         'sha', 'http', 'https', 'ssl', 'абонент', 'информационной', 'сети', 'автоматизированная', 'система',
         'авторизованный', 'субъект', 'доступа', 'авторство', 'информации', 'администратор', 'ас', 'защиты',
         'безопасности', 'информация', 'сертификат', 'аннулирование', 'аутентификация', 'аутсорсинг', 'безопасность',
         'технология', 'биометрическая', 'браузер', 'верификация', 'цп', 'эп', 'цифровая', 'электронная', 'подпись',
         'сертификация', 'владелец', 'сертификат', 'ключ', 'ключа', 'подписи', 'средства', 'выпуск', 'аудит',
         'доверяющая', 'документ', 'домен', 'доверия', 'доступ', 'ресурсу', 'доступность', 'закладочное', 'устройство',
         'pki', 'нсд', 'несанкционированного', 'санкционированного', 'идентификация', 'сети', 'инфраструктура',
         'компрометация', 'ключей', 'контроль', 'управление', 'конфиденциальная', 'конфиденциальность', 'целостность',
         'доступность', 'корпоративная', 'ис', 'криптографическая', 'ключ', 'преобразование', 'открым', 'закрытым',
         'локальная', 'межсетевой', 'экран', 'документа', 'открытый', 'закрытый', 'средства', 'пара', 'ключей',
         'подписчик', 'владелец', 'сертификата', 'подлинность', 'политика', 'разграничение', 'провайдер', 'профиль',
         'ресурасам', 'аннулированных', 'расшифрование', 'дешифрование', 'регистрационный', 'регламенту', 'риски',
         'модель', 'рисков', 'секретный', 'сервер', 'сервис', 'сертификация', 'сзи', 'ксзи', 'отозванных',
         'техническая', 'канал', 'утечки', 'токен', 'угроза', 'уязвимость', 'хеш-код', 'хеш-функция', 'функция',
         'хеширования', 'эцп', 'шифровальный', 'шифртекст', 'кузнечик', 'магма', 'эллиптических', 'кривых',
         'криптостойкость', 'анализ', 'параметры', 'блочные', 'поточные', 'гаммы', 'генератов', 'гост', 'пдн',
         'персональные', 'данные', 'обпеспечения', 'блокчейн', 'эцп', 'симметричные', 'асимметричные', 'механизм',
         'отправитель', 'получатель', 'сеансовый', 'сообщение', 'распределение', 'технических', 'мер', 'односторонняя',
         'блочное', 'поточное', 'шифрование', 'шифр', 'dss', 'логи', 'взлом', 'операционная', 'проникновение',
         'оценка', 'взаимная', 'методы', 'пароль', 'компьютер', 'форензика', 'помена', 'сертифицицированный',
         'код', 'системная', 'моделирование', 'антивирус', 'субд', 'атака']

df = pd.DataFrame(columns=words)
"""
#print(len(words))
k = 0
i = 1
arr = []
for word in words:
    while i < 66:
        with open('D:/tasks/task_8/docs/' + str(i) + '.txt', encoding='utf-8') as file:
            for line in file:
                if word in line:
                    k = k+1
                    print(k, line)
        file.close()
        print('closed', i)
        i = i + 1
        arr.append(k)
        k = 0
    i = 1
    print(arr)
    df[word] = arr
    arr = []
"""

df.to_csv("D:/tasks/task_8/df.csv", columns=words)


df = pd.read_csv('D:/tasks/task_8/df.csv')
#df = df.T

#ss = StandardScaler()
#df_scal = df.copy()
#df_scal = ss.fit_transform(df_scal)
df_scal = df.copy()
pca = PCA(n_components=2)
df2 = df_scal.copy()
df2 = pca.fit_transform(df2)

pcdf = pd.DataFrame(data=df2, columns=['pc1', 'pc2'])
#print(pca.components_, 'components') #!

plt.figure()
plt.grid()
plt.scatter(pcdf['pc1'], pcdf['pc2'], edgecolor='black', lw=.4, cmap='jet')
for i in range(len(pcdf)):
    plt.annotate(i, (pcdf['pc1'][i], pcdf['pc2'][i]))
    plt.arrow(0, 0, pcdf['pc1'][i], pcdf['pc2'][i])
plt.xlabel("pc1")
plt.ylabel("pc2")
plt.show()

def distance(df, num1, num2):
    distance = (df['pc1'][num1-1] * df['pc1'][num2-1] + df['pc2'][num1-1] * df['pc2'][num2-1]) /\
               math.sqrt((df['pc1'][num1-1] ** 2 + df['pc2'][num1-1] ** 2) * (df['pc1'][num2-1] ** 2 + df['pc2'][num2-1] ** 2))
    print("Расстояние между документами", num1, "и", num2, "равно", distance)

    return distance

distance(pcdf, 57, 6)