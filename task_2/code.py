### для HTML
from PIL import ImageColor
import matplotlib.pyplot as plt
from array import *
import numpy as np
from scipy.signal import argrelextrema
"""
array_gray_pix = array('f', [])   # создаем массив, в который будем записывать "серые пиксели"
with open('D:/tasks/task_2/mask2_294x182.html') as file:
    for line in file:   # построчно считываем файл
        if "TD  BGCOLOR" in line:
            color_hex = line[19:26]   # указываем, где находится нужная комбинация символов (цвет в формате hex)
            color_rgb = ImageColor.getrgb(color_hex)   # конвертируем hex в rgb
            gray = 0.299 * color_rgb[0] + 0.587 * color_rgb[1] + 0.114 * color_rgb[2]   # высчитываем яркость пикселя
            array_gray_pix.append(gray)   # записываем в массив полученное значение ярковсти

file.close()

k = int((1.25 * len(array_gray_pix) ** 0.4 - 0.55 * len(array_gray_pix) ** 0.4) / 2 + 0.55 * len(array_gray_pix) ** 0.4)   # высчитываем оптимальное число интервалов
if k % 2 == 0:   # так как нужно брать нечетное значение, мы проверяем k на четность
    k = k + 1   # если четно, прибавляем единицу

plt.figure()
(n, bins, patches) = plt.hist(array_gray_pix, bins=k, density=True)
plt.show()

array_local_minimum_index = argrelextrema(n, np.less)   # ищем локальные минимумы и получаем их индексы

array_local_minimum_index_counter = array_local_minimum_index[0]   # объявляем переменную, в которой будем потом перебирать индексы локальных минимумов
array_sum_of_densities = array('f', [])   # создаем массив, в который будем записывать сумму значений h для каждой области между локальными минимумами
counter = 0   # переменная, которая будет использоваться для перебора массива с индексами локальных минимумов
sum_of_one_area = 0   # сумма значений h в пределах одной однородной области

for i in n:   # пробегаем все значения h
    if counter < len(array_local_minimum_index_counter):   # условие, необходимое для для корреектной работы, после обработки последнего локального минимума
        if i != n[array_local_minimum_index_counter[counter]]:   # пока не встретится значение h, равное значению h в локальном минимуме
            sum_of_one_area = sum_of_one_area + i   # суммируем значения h
        else:   # когда встречается значение h, равное значению h в локальном минимуме
            counter = counter + 1   # начниаем работу с значением h следующего локального минимума
            array_sum_of_densities.append(sum_of_one_area)   # записываем полученную сумму значений h в ячейку массива
            sum_of_one_area = 0   # обнуляем переменную, чтобы считать сумму для следующей однородной области
    else:   # когда мы дошли до конца массива с минимуми, и оастлись значения в массиве со всеми h
        sum_of_one_area = sum_of_one_area + i   # суммируем значения h
        array_sum_of_densities.append(sum_of_one_area)   # записываем полученную сумму значений h в ячейку массива

array_sum_of_densities_without_first_max = array_sum_of_densities[:] # создали массив из сумм h без учета наибольшей суммы
array_sum_of_densities_without_first_max.remove(max(array_sum_of_densities_without_first_max))
array_sum_of_densities_without_first_max.remove(max(array_sum_of_densities_without_first_max))


array_bins_index_of_second_area = array('i', [])   # создаем массив для индексов значений вариационного ряда, входящих во вторую по величине область однородности


index_of_second_largest_area = array_sum_of_densities.index(max(array_sum_of_densities_without_first_max))   # индекс второй по величине (среди сумм h) области однородности в массиве со всеми суммами h (array_sum_of_densities)

   # создаем массив для индексов значений вариационного ряда, входящих во вторую по величине область однородности



n = array('f', n)
counter = 0   # переменная, которая будет использоваться для перебора массива с индексами локальных минимумов
for i in n:   # пробегаем все значения h
    if counter < len(array_local_minimum_index_counter):   # условие, необходимое для для корреектной работы, после обработки последнего локального минимума
        if i != n[array_local_minimum_index_counter[counter]]:   # пока не встретится значение h, равное значению h в локальном минимуме
            if counter == index_of_second_largest_area:   # если текущее значение массива с индексами локальных минимумов равно индексу второй по величине области однородности
                array_bins_index_of_second_area.append(n.index(i))   # записываем в массив индексы значений вариационного ряда, входящих во вторую по величине область однородности
        else:   # если текущее значение массива с индексами локальных минимумов не равно индексу второй по величине области однородности
            counter = counter + 1   # рассматриваем следующее массива с индексами локальных минимумов

array_bins_of_second_area = bins[array_bins_index_of_second_area]   # массив значений вариационного ряда, входящих во вторую по величине область однородности
bins_of_second_area_min = min(array_bins_of_second_area)   # минимальное значение вариационного ряда, вхоядщее во вторую по величине область однородности
bins_of_second_area_max = max(array_bins_of_second_area)   # максимальное значение вариационного ряда, вхоядщее во вторую по величине область однородности

file_finished = open('D:/tasks/task_2/plane_294x182_finished.html', 'w')   # создаем файл, в который будем осуществлять запись измененного файла
with open('D:/tasks/task_2/mask2_294x182.html') as file:
    for line in file:   # построчно считываем файл
        if "TD  BGCOLOR" in line:
            color_hex = line[19:26]   # указываем, где находится нужная комбинация символов (цвет в формате hex)
            color_rgb = ImageColor.getrgb(color_hex)   # конвертируем hex в rgb
            gray = 0.299 * color_rgb[0] + 0.587 * color_rgb[1] + 0.114 * color_rgb[2]   # высчитываем яркость пикселя
            if gray >= bins_of_second_area_min and gray <= bins_of_second_area_max:   # если значение яркости пикселя принадлежит промежутку значений вариационного ряда, входящих во вторую по величине область однородности
                line = '<td bgcolor="#ff0000">'   # заменяем строку с цветом, яркость в серых тонах которого принадлежит принадлежит промежутку значений вариационного ряда, входящих во вторую по величине область однородности, на строку, с красным цветом
                file_finished.write(line)   # записываем в новый файл эту строку
            else:   # если значение яркости пикселя не принадлежит промежутку значений вариационного ряда, входящих во вторую по величине область однородности
                file_finished.write(line)   # записываем в новый файл строку без изменений
        else:   # если обрабатываемая строка не содержит ключевого слова, по которому мы искали строки, содержащие цвет
            file_finished.write(line)   # записываем в новый файл строку без изменений
file.close()   # закрываем исходный файл
file_finished.close()   # закрываем файл, в который записали новые цвета

"""








### для изображения
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from array import *
img = Image.open('D:/tasks/task_2/plane_294x182.html')
draw = ImageDraw.Draw(img)
pix = img.load()
array_pix = array('i', [])
for x in range(img.size[0]):
    for y in range(img.size[1]):
        gray = int(0.299 * pix[x, y][0] + 0.587 * pix[x, y][1] + 0.114 * pix[x, y][2])
        draw.point((x, y), (gray, gray, gray))
        array_pix.append(gray)

#img.save("result.jpg", "JPEG")
print(array_pix)
#n=len(array_pix)

k = 256
plt.figure()
(n, bins, patches) = plt.hist(array_pix, bins=k, density=True)
plt.show()

for x in range(img.size[0]):
    for y in range(img.size[1]):
        if (pix[x, y][1] < 55):
            draw.point((x, y), (255, 0, 0))
img.save("result.jpg", "JPEG")