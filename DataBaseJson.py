import json
import numpy as np
import dlib, cv2
import os, shutil
import ast
from scipy.spatial import distance

import datetime

#Чтение файла
def readJSON(filename='inroom.json'):
    if not os.path.isfile(filename): #если его нет -> создаем пустой
        with open(filename, 'w') as outfile:
            json.dump([], outfile)
        return []
    else:
        with open(filename, 'r') as f:
            _json = ast.literal_eval(str(json.load(f)))
        _json = dlibVectorFormating(data=_json, tolist=False)
        return _json

def sample_readJSON(filename='inroom.json'):
    if not os.path.isfile(filename): #если его нет -> создаем пустой
        with open(filename, 'w') as outfile:
            json.dump([], outfile)
        return []
    else:
        with open(filename, 'r') as f:
            _json = ast.literal_eval(str(json.load(f)))
        return _json

#Добавление данных в файл
def addToJSON(data, filename = 'inroom.json'):
    _json = readJSON(filename)
    data = ast.literal_eval(str(data))
    _json.append(data)
    writeToJSON(_json, filename)
#форматирование дескрипторов для чтения и записи
def dlibVectorFormating(data, tolist=True, key_descr = 'descr'):
    for d in data:
        if tolist:
            d[key_descr]=[list(dd) for dd in d[key_descr]]
        else:
            d[key_descr]=[dlib.vector(dd) for dd in d[key_descr]]
    return data
#Перезапись файла
def writeToJSON(data, filename = 'inroom.json'):
    _ = readJSON(filename)
    data = dlibVectorFormating(data)
    data = ast.literal_eval(str(data))
    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True)
#Удаление из файла по ключу (дефолтно - id)
def deleteFromJSON(value, key = 'id', filename='inroom.json'):
    _json = readJSON()
    _json[:] = [d for d in _json if d.get(key) != value]
    writeToJSON(_json, filename)

#Почти взвешанное решение (возвращает минимальное расстояние)
def weighted_decision_ish(face_descriptor, descriptors):
    return min([distance.euclidean(descr, face_descriptor) for descr in descriptors])
#Поиск лица в базе данных
def searchInJSON(descriptor, key='descr', filename='inroom.json'):
    thresold, predict_name, predict_id, url = 0.55, '', 0, ''
    data = readJSON(filename)
    for d in data:
        dist = weighted_decision_ish(face_descriptor=descriptor, descriptors=d[key])
        if dist < thresold:
            return d
    return None
#Возвращает максимальное вхождение по ключу
def getMaxValue(key='id', filename='inroom.json'):
    _json = readJSON(filename)
    maxValue = 0
    try:
        return max([d[key] for d in _json])
    except:
        return 0
#Выделяет папку для новой персоны, возвращает путь до неё
def getNewURL(path):
    if os.path.isdir(path):
        return path
    else:
        abspath = os.path.abspath(path)
        os.mkdir(abspath)
        return abspath
#Переиминовываем файлы для очереди (1->0, 2->1, 3->2 и т.д.)
def renameFiles(path):
    filenames = os.listdir(path)
    filenames.sort()
    for fn in filenames:
        os.rename('{path}/{fn}'.format(path=path, fn=fn), '{path}/{new_name}.png'.format(path=path, new_name=int(fn[:-4])-1))
#Создает очередь из фотографий (держит в себе только fdb_size последних)
def queue(path, data, fdb_size):
    num_files = len(os.listdir(os.path.abspath(path)))
    if not os.path.isfile('{path}/0.png'.format(path=path)): #если его нет -> создаем пустой
        renameFiles(path)
        cv2.imwrite('{path}/{fdb_size}.png'.format(fdb_size=fdb_size, path=path), data)
    elif num_files < fdb_size:
        cv2.imwrite('{path}/{num_files}.png'.format(num_files=num_files,path=path), data)
    else:
        os.remove('{path}/0.png'.format(path=path))
        renameFiles(path)
        cv2.imwrite('{path}/{fdb_size}.png'.format(fdb_size=fdb_size, path=path), data)
#Увеличение выделенной области лица на коэффициент deltha и вырезаем
def framesSizeIncreasing(image, coordinates, deltha):
    shape = image.shape
    alpha, betta = int(shape[0]*deltha), int(shape[1]*deltha)
    t, b, l, r = max(coordinates.top() - alpha, 0), min(coordinates.bottom() + alpha, shape[0]), max(coordinates.left() - betta, 0), min(coordinates.right() + betta, shape[1])
    return image[t:b, l:r]


def age_label(ages):

    predicted_ages = np.mean(ages)
    print(ages)
    print(predicted_ages)

    if predicted_ages < 18:
        return 'C'

    elif 18 <= predicted_ages < 30:
        return 'Y'

    elif 30 <= predicted_ages < 55:
        return 'A'

    else:
        return 'O'


def gender_label(genders):

    female_prob = np.mean([g[0] for g in genders])
    male_prob = np.mean([g[1] for g in genders])

    return "F" if female_prob >= male_prob else "M"

#Сохраняем вескрипторы в файл в виде очереди
def dump(meta, fdb_size=10, key_id='id', filename='inroom.json'):
    db = readJSON(filename)
    isexists = False
    for d in db:
        if d[key_id] == meta[key_id]:
            if len(d['descr'])>=fdb_size: #хранятся всегда fdb_size актуальных дескрипторов
                d['descr'] = d['descr'][len(d['descr'])-fdb_size+1:]

            d['descr'].append(meta['descr'][0])
            d['age'] = meta['age']
            d['gender'] = meta['gender']

            d['period'] = meta['period']
            d['checked'] = meta['checked']

            d['age_label'] = meta['age_label']
            d['gender_label'] = meta['gender_label']
            isexists = True
            break

    if not isexists:
        db.append(meta)
    writeToJSON(db, filename)

def stupid_queue(mass, value, fdb_size):
    if len(mass) > fdb_size:
        mass = mass[1:]
    mass.append(value)
    return mass

def ageAndGenderRecogn(metas, model, faces, key_gender='gender', key_age='age', fdb_size = 10):
    results = model.predict(faces)
    predicted_genders = results[0]
    ages = np.arange(0, 101).reshape(101, 1)
    predicted_ages = results[1].dot(ages).flatten()
    for i in range(faces.shape[0]):
        metas[i][key_age] = stupid_queue(metas[i][key_age], float(predicted_ages[i]), fdb_size)
        metas[i][key_gender] = stupid_queue(metas[i][key_gender], predicted_genders[i].tolist(),fdb_size)
    return metas

#TODO NEED Refactoring
def WoodenSaveHist(metas, period = 24):

    male = sample_readJSON('male.json')

    if not male:
        male = {"C" : [0 for i in range(period)],
                "Y" : [0 for i in range(period)],
                "A" : [0 for i in range(period)],
                "O" : [0 for i in range(period)],
                "total": [0 for i in range(period)],
                "percentage": 0,
                "nframe" : period}

    female = sample_readJSON('female.json')

    if not female:
        female = {"C" : [0 for i in range(period)],
                  "Y" : [0 for i in range(period)],
                  "A" : [0 for i in range(period)],
                  "O" : [0 for i in range(period)],
                  "total": [0 for i in range(period)],
                  "percentage": 0,
                  "nframe" : period}


    for i in range(len(metas)):

        if metas[i]['checked']:
            continue

        metas[i]['checked'] = True

        print(metas[i]['gender_label'],metas[i]['age_label'])

        if metas[i]['gender_label'] == 'M':
            male[metas[i]['age_label']][metas[i]['period']] += 1
            male['total'][metas[i]['period']] += 1

        elif metas[i]['gender_label'] == 'F':
            female[metas[i]['age_label']][metas[i]['period']] += 1
            female['total'][metas[i]['period']] += 1

    totalMale = sum(male['total'])
    totalFemale = sum(female['total'])
    totalTotal = (totalMale + totalFemale)

    female["percentage"] = (totalFemale / totalTotal) * 100
    male["percentage"] = (totalMale / totalTotal) * 100

    with open('male.json', 'w') as outfile:
        json.dump(male, outfile, indent=4, sort_keys=True)

    with open('female.json', 'w') as outfile:
        json.dump(female, outfile, indent=4, sort_keys=True)

    return metas

#Распознавалка по лицу
def faceRecogn(shape_predictor, recognizer, coordinates, gray, frame, AGSmodel, acc = 'm', img_size=64, key_id='id', key_name='name', key_descr='descr', key_gender='gender', key_age='age', filename='inroom.json', path_with_faces = './faces/', fdb_size=10, deltha = .2):
    faces = np.empty((len(coordinates), img_size, img_size, 3))
    metas = []

    time = datetime.datetime.now().minute
    period = 60

    if acc == 'h':
        time = datetime.datetime.now().hour
        period = 24

    elif acc == 'w':
        time = datetime.datetime.now().weekday()
        period = 7

    for k, oneface in enumerate(coordinates):
        shape = shape_predictor(gray, oneface)
        face_descriptor = recognizer.compute_face_descriptor(gray, shape)#получаем дескриптор лица
        faces[k,:,:,:] = cv2.resize(framesSizeIncreasing(frame,oneface, deltha), (img_size,img_size))
        meta = searchInJSON(face_descriptor, key=key_descr, filename=filename)#ищем его в базе
        if not meta:#если ни на кого не похож
            new_id=getMaxValue(key=key_id, filename=filename)+1#создаем нового с самым большим индексом в базе
            meta = {'gender_label': '', 'age_label': '', 'checked':False,  key_id:new_id, key_name:'NoName{new_id}'.format(new_id=new_id), key_descr:[], key_gender:[], key_age:[], 'key_url':getNewURL('{path_with_faces}{new_id}'.format(path_with_faces=path_with_faces, new_id=new_id)), 'period' : time}
        meta[key_descr] = [face_descriptor]
        face = framesSizeIncreasing(frame, oneface, deltha)
        queue(path=meta['key_url'], data=face, fdb_size=fdb_size)#сохраняем лицо в нужную папочку

        if not meta['period'] == time:
            meta['checked'] = False
            meta['period'] = time

        meta['age_label'] = age_label(meta['age'])
        meta['gender_label'] = gender_label(meta['gender'])

        metas.append(meta)

    metas = ageAndGenderRecogn(metas=metas, model=AGSmodel, faces=faces, fdb_size = fdb_size)

    metas = WoodenSaveHist(metas, period = period)

    for i in range(len(metas)):
        dump(meta=metas[i],fdb_size=fdb_size, filename=filename)
