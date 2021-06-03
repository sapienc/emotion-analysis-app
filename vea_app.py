# ===== [IMPORT: Importing standard libraries] =====
import os.path
import pafy
import shutil
import json
import random
import datetime

# ===== [IMPORT: Importing external libraries] =====
from cv2 import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
import moviepy.editor

#_______________________________________________________________________________________________________________
class VideoEmotionAnalyzer:
    def __init__(self) -> None:
        self.SETTINGS_PROJECT_PREVIEW_IMG_WIDTH = 60
        self.SETTINGS_PROJECT_PREVIEW_IMG_HEIGHT = 60
        
        self.FOLDER_ROOT = './' # '/'
        self.FOLDER_STATIC = self.FOLDER_ROOT + 'static/'
        self.FOLDER_MODEL = self.FOLDER_STATIC + 'model/'
        self.FOLDER_FRAMES = 'frames/'
        self.FOLDER_PROJECTS = 'projects/'
        self.FOLDER_PREDICTED_FRAMES = 'predicted_frames/'
        
        
        self.PATH_PROJECTS = self.FOLDER_STATIC + self.FOLDER_PROJECTS

        self.FILE_NAME_IMG_PREVIEW = 'preview.jpg'
        self.FILE_NAME_PROJECT_INFO = 'project_info.json'
        self.FILE_NAME_DATA = 'data.csv'
        self.FILE_NAME_EMOJIS_DATA = 'emojisData.json'
        self.FILE_NAME_AUDIO = 'extracted_audio.mp3'

        self.FILE_PREFIX_ANALYZED = 'analyzed_'
        self.FILE_PREFIX_FRAME = 'frame_'

        self.AVAILABLE_VDIDEO_EXTENSIONS = ['.mp4'] #, '.avi']

        self.CURRENT_PROJECT_name = None
        self.CURRENT_PROJECT_path = None
        self.CURRENT_PROJECT_original_video_file = None
        self.CURRENT_PROJECT_analyzed_video_file = None
        self.CURRENT_PROJECT_data_file = None
        self.CURRENT_PROJECT_emojis_data_file = None
        self.CURRENT_PROJECT_meta_data = {}
    
    # ========== [Проверка на существование соответствующей директории для проекта] ==========
    def get_path_by_project_name(self, project_name):
        return self.PATH_PROJECTS + project_name + '/'
    
    # ========== [Проверка на существование соответствующей директории для проекта] ==========
    def is_project_has_own_folder(self, project_name):
        path = self.get_path_by_project_name(project_name)
        if os.path.exists(path):
            return True, path
        return False, path
    
    # ========== [Проверка на существование директории frames для проекта] ==========
    def is_project_has_frames_folder(self, project_name):
        path = self.get_path_by_project_name(project_name) + self.FOLDER_FRAMES
        if os.path.exists(path):
            return True, path
        return False, path

    # ========== [Проверка на существование директории frames для проекта] ==========
    def is_project_has_predicted_frames_folder(self, project_name):
        path = self.get_path_by_project_name(project_name) + self.FOLDER_PREDICTED_FRAMES
        if os.path.exists(path):
            return True, path
        return False, path

    # ========== [Проверка на существование директории frames для проекта] ==========
    def create_folder_for_project(self, project_name, folder_name):
        folder_path = self.get_path_by_project_name(project_name) + folder_name

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        return folder_path

    # ========== [Получение данных проекта (также своего рода проверка на существование данных в проекте)] ==========
    def get_project_data_info(self, project_name) -> object:
        if not self.is_project_has_own_folder(project_name)[0]: # если как минимум нету папки проекта
            return {} # False # то данных проекта не существует
        
        project_info = {
            'original_video_filename': None,
            'analyzed_video_filename': None,
            'prediction_data': None,
            'emojis_data': None,
            # 'original_video_duration': None, # TODO
            # 'analyzed_video_duration': None # TODO
        }

        for current_filename in os.listdir(self.get_path_by_project_name(project_name)): # пройдемся по файлам
            file_name, file_extension = os.path.splitext(current_filename) # извлечем имя и расширение файла


            if file_extension in self.AVAILABLE_VDIDEO_EXTENSIONS: # если такой формат файла можно рассмотреть
                if file_name.startswith('analyzed'): # если название файла начинается со слова analyzed
                    project_info.update({ 'analyzed_video_filename': current_filename })
                
                if file_name == project_name: # если название файла совпадает с названием проекта
                    project_info.update({ 'original_video_filename': current_filename })
            
            elif current_filename == self.FILE_NAME_DATA:
                project_info.update({'prediction_data': current_filename})
                # with open(self.get_path_by_project_name(project_name) + current_filename, "r") as file:
                    # project_info.update({'prediction_data': json.load(file)})

            elif current_filename == self.FILE_NAME_EMOJIS_DATA:
                project_info.update({'emojis_data': current_filename})

        return project_info
    
    # ========== [Загрузка метаданных проекта] ==========
    def get_project_meta_data_info(self, project_path):
        print(project_path)
        result = {}
        try:
            with open(project_path + '/' + self.FILE_NAME_PROJECT_INFO) as json_file: # self.CURRENT_PROJECT_path + '/' + self.FILE_NAME_PROJECT_INFO
                result = json.load(json_file)
        except:
            pass

        return result

    # ========== [Создать файл с информацией о проекте] ==========
    def make_project_info(self, video_file_path, project_path):
        video = moviepy.editor.VideoFileClip(video_file_path)
        seconds_duration = video.duration

        normalized_duration = str(datetime.timedelta(seconds=seconds_duration))
        corrected_duration = normalized_duration.split(':')
        
        for i, time_value in enumerate(corrected_duration):
            print("\ttime_value:", time_value)
            if len(time_value) == 1 and int(time_value) < 10:
                corrected_duration[i] = '0' + time_value
                
            elif '.' in time_value:
                corrected_duration[i] = str(round(float(time_value)))

        corrected_duration = ':'.join(corrected_duration)
        print("[DURATION 2]:corrected_duration", corrected_duration)
        
        now = datetime.datetime.now()
        current_date_and_time = now.strftime("%d.%m.%Y %H:%M")

        self.CURRENT_PROJECT_meta_data.update({
            "created_date": current_date_and_time,
            "video_duration": corrected_duration
        })

        with open(project_path + '/' + self.FILE_NAME_PROJECT_INFO, "w", encoding="utf-8") as file:
            json.dump(self.CURRENT_PROJECT_meta_data, file, indent = 4, ensure_ascii = False)

        print("Made project info")
        return True


    # ========== [Выбор текущего проекта] ==========
    def select_project(self, project_name):
        project_info = self.get_project_data_info(project_name)
        
        if not project_info:
            return False

        self.CURRENT_PROJECT_name = project_name
        self.CURRENT_PROJECT_path = self.get_path_by_project_name(project_name)
        self.CURRENT_PROJECT_original_video_file = project_info['original_video_filename']
        self.CURRENT_PROJECT_analyzed_video_file = project_info['analyzed_video_filename']
        self.CURRENT_PROJECT_data_file = project_info['prediction_data']
        self.CURRENT_PROJECT_emojis_data_file = project_info['emojis_data']

        self.CURRENT_PROJECT_meta_data = self.get_project_meta_data_info(self.CURRENT_PROJECT_path)
        return True

    # ========== [Создать дефолтное Изображение "превью проекта"] ==========
    def make_default_project_preview_by_video(self, video_file_path, project_path):
        # Картинка превью проекта
        capture = cv2.VideoCapture(video_file_path)
        frames_amount = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        chosen_frame = random.randint(24, frames_amount)

        capture.set(cv2.CAP_PROP_POS_FRAMES, chosen_frame) # возьмем рандомный chosen_frame кадр
        ret, frame = capture.read()
        
        if ret:
            # img = cv2.imread(frame)
            frame = cv2.resize(frame, (self.SETTINGS_PROJECT_PREVIEW_IMG_WIDTH, self.SETTINGS_PROJECT_PREVIEW_IMG_HEIGHT))
            cv2.imwrite(project_path + '/' + self.FILE_NAME_IMG_PREVIEW, frame) # создадим изображение кадра
            print("Preview for the project was successfully created")
        
        capture.release()
        cv2.destroyAllWindows()

    # ========== [Создание проекта] ==========
    def create_project(self, project_name):
        project_path = self.get_path_by_project_name(project_name)

        if not self.is_project_has_own_folder(project_name)[0]: # если у проекта нету папки    
            os.makedirs( project_path ) # создадим папку
        
        return project_path

    # ========== [Создание проекта по скачиванию видео из YouTube] ==========
    def create_project_from_youtube(self, url):
        vPafy = pafy.new(url)
        video = vPafy.getbest(preftype="mp4")
        project_name = video.title

        project_info = self.get_project_data_info(project_name) # извлекаем информацию о проекте

        if not project_info or project_info['original_video_filename'] is None: # если информации нет, либо видео для проекта не загружено
            pjct_path = self.create_project(project_name) # создаем проект
            # print("DOWNLOAD INTO:", pjct_path)

            video_file_path = pjct_path + video.title + "." + video.extension
            # print("video_file_path:", video_file_path)
            video.download(filepath = video_file_path) # загружаем видео
            # return video_file_path

            self.make_default_project_preview_by_video(video_file_path, pjct_path) # Картинка превью проекта
            self.make_project_info(video_file_path, pjct_path)
        
        return project_name # возвращаем название проекта

    # ========== [Создание проекта по скачиванию видео из YouTube] ==========
    def create_project_local_storage(self, file):
        project_name, file_extension = os.path.splitext(file.filename)
        
        if file_extension in self.AVAILABLE_VDIDEO_EXTENSIONS: # если такой формат файла можно рассмотреть
            project_info = self.get_project_data_info(project_name) # извлекаем информацию о проекте

            if not project_info or project_info['original_video_filename'] is None: # если информации нет, либо видео для проекта не загружено
                pjct_path = self.create_project(project_name) # создаем проект
                video_file_path = pjct_path + file.filename
                file.save(video_file_path) # сохраняем видеофайл

                self.make_default_project_preview_by_video(video_file_path, pjct_path) # Картинка превью проекта
                self.make_project_info(video_file_path, pjct_path)
        
        return project_name # возвращаем название проекта

    # ========== [Обзор имеющихся проектов (папок)] ==========
    def browse_all_projects(self):
        projects_list = []
        projects_paths = []
        if os.path.exists(self.PATH_PROJECTS):
            projects_list = os.listdir(self.PATH_PROJECTS) #TODO what if there is no video file

            for project in projects_list:
                projects_paths.append(
                    os.path.join(self.PATH_PROJECTS, project)
                )
            print("path_concatenation has been done:", projects_paths)
        
        return projects_list, projects_paths

    # ========== [Разбиение видео на кадры] ==========
    def extract_audio_of_video(self, video_file_path = None):
        video_file_path = video_file_path or self.CURRENT_PROJECT_path + '/' + self.CURRENT_PROJECT_original_video_file
        new_audio_file_path = self.get_path_by_project_name(self.CURRENT_PROJECT_name) + self.FILE_NAME_AUDIO
        video_file_audio = moviepy.editor.AudioFileClip(video_file_path)
        video_file_audio.write_audiofile(new_audio_file_path)
        return True, new_audio_file_path

    # ========== [Разбиение видео на кадры] ==========
    def cut_video_into_frames(self, video_file_path = None):
        frames_folder_exists, frames_folder_path = self.is_project_has_frames_folder(self.CURRENT_PROJECT_name)
        video_file_path = video_file_path or self.CURRENT_PROJECT_path + '/' + self.CURRENT_PROJECT_original_video_file

        capture = cv2.VideoCapture(video_file_path)
        total_frames_amount = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        try:
            if frames_folder_exists:
                # shutil.rmtree(frames_folder_path) # очищаем кадры, если они были там до этого
                # os.makedirs(frames_folder_path)
                pass
            else:
                os.makedirs(frames_folder_path) # создаем папку, если она не была ранее создана

        except OSError:
            print ('Error: Creating directory of data')

        print("\tCutting video into frames...")
        print("\tTotal amount of frames:", total_frames_amount)
        created_frame_counter = 0

        while(True):
            # Читаем видеофайл покадроово
            print("\t\tcurrent frame is:", created_frame_counter, "/", total_frames_amount, end='\r')
            ret, frame = capture.read()
        
            # если кадр еще имеется, то создадим для него изображение:
            if ret:
                frame_file_name = frames_folder_path + "frame_" + str(created_frame_counter) + '.jpg'
                cv2.imwrite(frame_file_name, frame) # создадим изображение кадра
                created_frame_counter += 1 # инкрементируем счетчик действительно созданных кадров
            # иначе
            else:
                break # прекратим цикл
        
        print("\tFrames were successfully created:", created_frame_counter, "/", total_frames_amount)
        capture.release()
        cv2.destroyAllWindows() 

    # ========== [Сборка видео из кадров] ==========
    def build_video_from_frames(self, video_file_path):
        print("\tBuilding analyzed video from frames...")
        _, predicted_frames_folder_path = self.is_project_has_predicted_frames_folder(self.CURRENT_PROJECT_name)
        video_file_path = video_file_path or self.CURRENT_PROJECT_path + '/' + self.CURRENT_PROJECT_original_video_file

        capture = cv2.VideoCapture(video_file_path) 
        video_file_fps = capture.get(cv2.CAP_PROP_FPS)

        predicted_images = []

        print("predicted_frames_folder_path:", predicted_frames_folder_path)
        for frame_number in range( len(os.listdir( predicted_frames_folder_path )) ):
            frame_file_name = predicted_frames_folder_path + "frame_" + str(frame_number) + '.jpg'
            current_frame_image = cv2.imread(frame_file_name)
            height, width, layers = current_frame_image.shape
            size = (width, height)
            predicted_images.append(current_frame_image)
            print("\t\treading frame №", frame_number, "|frame_file_name:", frame_file_name, end='\r')
            
        original_video_file_name, original_video_extension = os.path.splitext(self.CURRENT_PROJECT_original_video_file)
        new_video_file_name = self.FILE_PREFIX_ANALYZED + original_video_file_name + '.avi' # с '.mp4' как-то не заладилось, поэтому пришлось использовать .avi
        new_video_file_path = self.CURRENT_PROJECT_path + '/' + new_video_file_name
        
        fourcc = cv2.VideoWriter_fourcc(*'MPEG') # выбираем 4-byte code codec # ранее: 'avc1' 'mp4v', 'MPEG', 'H264', 'X264'
        predicted_video = cv2.VideoWriter(
            # "analyzed_output.mp4",
            new_video_file_path,
            fourcc,
            video_file_fps,
            size
        )
        
        for i in range(len(predicted_images)):
            print("\t\tmerging frame №", frame_number, end='\r')
            predicted_video.write(predicted_images[i])
        
        predicted_video.release()
        cv2.destroyAllWindows()
        print("\n\tVideo has been successfully created:", new_video_file_name)
        return True, new_video_file_path, video_file_fps

    # ========== [Сборка видео из кадров] ==========
    def merge_video_with_audio(self, video_file_path, audio_file_path, fps):
        print("\n\tMerging audio with video:\n\t\tVideo file:\n\t\t", video_file_path, "\n\t\tAudio file:\n\t\t", audio_file_path)
        target_video = moviepy.editor.VideoFileClip(video_file_path)
        target_audio = moviepy.editor.AudioFileClip(audio_file_path)

        merged_video_file_folder_path = os.path.dirname(video_file_path)
        merged_video_file_name, _ = os.path.splitext(os.path.basename(video_file_path))
        new_merged_video_file_path = merged_video_file_folder_path + '/' + self.FILE_PREFIX_ANALYZED + merged_video_file_name + '.mp4'

        merged_video_file = target_video.set_audio(target_audio)
        merged_video_file.write_videofile(new_merged_video_file_path, fps = fps)

        os.remove(video_file_path) # удаляем старый временный .avi файл
        return True, new_merged_video_file_path
    
    # ========== [Распознавание изображения] ==========
    def predict(self, frames_folder_path = None):
        previewDone = False
        frames_folder_path = frames_folder_path or self.CURRENT_PROJECT_path + self.FOLDER_FRAMES
        categories = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        
        print("Predicting result..")
        model = tf.keras.models.load_model(self.FOLDER_MODEL + 'model.h5')

        _,w,h,_ = model.input.shape # here
        def feature_extrator_fun(img):
            resized_image = cv2.resize(img, (w,h))
            resized_image = cv2.cvtColor(resized_image,cv2.COLOR_BGR2RGB)
            x=resized_image.astype(np.float32)
            x[..., 0] -= 103.939
            x[..., 1] -= 116.779
            x[..., 2] -= 123.68
            x = np.expand_dims(x, axis=0)
            features = model.predict(x)
            return features[0]
        
        pred="none"
        pred_stats={"Positive": 0, "Negative": 0, "Neutral": 0}
        json_preds = {
            "angry": [],
            "disgust": [],
            "fear": [],
            "happy": [],
            "neutral": [],
            "sad": [],
            "surprise": []
        } #[]
        
        pos=0
        neg=0
        neu=0
        all_preds = []

        facec = cv2.CascadeClassifier(self.FOLDER_MODEL + 'haarcascade_frontalface_default.xml')
        font = cv2.FONT_HERSHEY_SIMPLEX

        emojisFrameData = {} # emojisFrameData = []

        total_frames_amount = len(os.listdir(frames_folder_path))
        for count in range(total_frames_amount):
            frame_filename = frames_folder_path + '/frame_' + str(count) + '.jpg' # извлекаем изображение кадра

            frame = cv2.imread(frame_filename) # загружаем изображение кадра в библиотеку OpenCV

            feats=feature_extrator_fun(frame)
            pred = categories[np.argmax(feats)]
            all_preds.append(feats)

            gray_fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_locations = facec.detectMultiScale(gray_fr, 1.3, 5, minSize=(w, h)) # найдем все лица на текущем кадре
            face=0

            # emojisFrameData.append( [] ) # добавляем массив, который будет иметь данные для каждого найденного лица на момент текущего рассматриваемого кадра
            for top, right, bottom, left in face_locations:
                # Draw a box around the face
                face=face+1

                frame_second = count#/24%60
                if frame_second in emojisFrameData:
                    emojisFrameData[frame_second].append(pred.lower())
                else:
                    emojisFrameData[frame_second] = [pred.lower()]
                
                # Blue color in BGR 
                rgb_value = (255,255,255)
                if pred=="Negative" or pred=="Angry" or pred=="Disgust" or pred=="Fear" or pred=="Sad":
                    rgb_value=(0,0,255)
                    neg+=1

                elif pred=="Neutral":
                    rgb_value=(255,0,0)
                    neu+=1

                elif pred=="Positive" or pred=="Surprise" or pred=="Happy" :  
                    rgb_value=(0,255,0)
                    pos+=1

                # Превью проекта
                if not previewDone:
                    sub_face = frame[right:(right+left), top:(top+w)] # [right+150:right+left + 150, top+150:top+w + 150]
                    previewFrame = cv2.resize(sub_face, (50, 50))
                    cv2.imwrite(self.CURRENT_PROJECT_path + self.FILE_NAME_IMG_PREVIEW, previewFrame)
                    print("Preview has been updated")
                    previewDone = True

                cv2.putText(frame, pred, (top,right), font, 1, rgb_value, 2)
                cv2.rectangle(frame, (top,right), (top+bottom,right+left), rgb_value, 2)

            if os.path.isfile(frame_filename):
                cv2.imwrite(frame_filename, frame)

            pred_stats["Positive"]=pos
            pred_stats["Negative"]=neg
            pred_stats["Neutral"]=neu

        with open(self.CURRENT_PROJECT_path + '/' + self.FILE_NAME_EMOJIS_DATA, "w", encoding="utf-8") as file:
            json.dump(emojisFrameData, file, indent = 4, ensure_ascii = False)
        
    
        predictionsDataFrame = pd.DataFrame(all_preds, columns=[category.lower() for category in categories])
        predictionsDataFrame.to_csv(self.CURRENT_PROJECT_path +  self.FILE_NAME_DATA, sep=',', index=False, header=True)

        return all_preds, pred_stats, emojisFrameData