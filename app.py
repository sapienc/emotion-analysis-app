# ===== [IMPORT: Importing standard libraries] =====
import os.path
import shutil
import json
import pandas as pd
# from datetime import datetime, date

# ===== [IMPORT: Importing internal libraries] =====
import vea_app

# ===== [IMPORT: Importing external libraries] =====
from flask import Flask, render_template, jsonify, request, send_from_directory, redirect, url_for, Request, Response

import cv2


flask = Flask(__name__)
vea_app = vea_app.VideoEmotionAnalyzer()

flask.config['UPLOAD_EXTENSIONS'] = ['.mp4']
flask.config['UPLOAD_FOLDER'] = vea_app.PATH_PROJECTS
#######################################################################################################
@flask.after_request
def after_request(response):
    response.headers.add('Accept-Ranges', 'bytes')
    return response

#######################################################################################################
# Main page
@flask.route('/')
def main():
    projects, projects_paths = vea_app.browse_all_projects()
    meta_data = []

    for path in projects_paths:
        print("\n\nTrying path:", path)
        meta_data.append(vea_app.get_project_meta_data_info(path))
        print("\nMeta data now:", meta_data)
    
    extra = [
        {
            "descr": "Нет описания",
            "duration": "00:00:30"
        },
        {
            "descr": "Democratic debate: candidates face off during impeachment proceedings.  Here are some of the night's best moments.",
            "duration": "00:03:34"
        }
    ]

    return render_template('main.html', all_projects = projects, extra = extra, meta = meta_data)
# ---------------------------------------- [Загрузка видео по ссылке из YouTube] ----------------------------------------
# Downloading route
@flask.route('/download', methods=['POST'])
def download_file():
    url = request.form['input_video_url'] # url, введенный в соответствующее input поле
    project_name = vea_app.create_project_from_youtube(url)
    return redirect(url_for("project", pjct_name = project_name))
    
# ---------------------------------------- [Выбор проекта] ----------------------------------------
@flask.route('/project/<string:pjct_name>/', methods=["GET"])
def project(pjct_name):
    vea_app.select_project(pjct_name)
    project_path = vea_app.CURRENT_PROJECT_path[1:] # не будем иметь в виду символ точки "."
    print("SELECTED PROJECT")

    pjct_info = {
        "path": project_path,
        "name": vea_app.CURRENT_PROJECT_name,
        "original_video": vea_app.CURRENT_PROJECT_original_video_file,
        "analyzed_video": vea_app.CURRENT_PROJECT_analyzed_video_file,
        "prediction_data": vea_app.CURRENT_PROJECT_data_file,
        "emojis_data": vea_app.CURRENT_PROJECT_emojis_data_file,
        "project_meta": vea_app.CURRENT_PROJECT_meta_data
    }


    print("\t\tpjct_info:", pjct_info)
    
    return render_template('analyzer.html', project = pjct_info)

# ---------------------------------------- [Запуск анализа видео] ----------------------------------------
@flask.route('/analyze', methods=['POST'])
def analyze():
    print("\n\n\tAnalyzing..")

    pjct_name = vea_app.CURRENT_PROJECT_name # request.form['project_name']
    print("\t [Analyzing] pjct_name:", pjct_name)

    if not pjct_name: # если имя проекта в данный момент недоступно, то вероятно history в браузере сбился, а пользователь остался на той же странице
        return redirect(request.referrer) # редиректим на предыдущую страницу, чтобы она обновилась
    
    video_file_name = vea_app.CURRENT_PROJECT_original_video_file # request.form['original_video_file_name']
    print("\t [Analyzing] video_file_name:", video_file_name)

    video_file_path = vea_app.get_path_by_project_name(pjct_name) + video_file_name
    print("\t [Analyzing] video_file_path:", video_file_path)

    frames_folder_exists, frames_folder_path = vea_app.is_project_has_frames_folder(pjct_name)
    predicted_frames_folder_exists, predicted_frames_folder_path = vea_app.is_project_has_predicted_frames_folder(pjct_name)
    print("\t [Analyzing] predicted_frames_folder_exists:", predicted_frames_folder_exists)
    print("\t [Analyzing] predicted_frames_folder_path:", predicted_frames_folder_path)

    _, extracted_audio_file_path = vea_app.extract_audio_of_video(video_file_path)
    vea_app.cut_video_into_frames(video_file_path)
    
    
    if predicted_frames_folder_exists: # если уже существует
        print("removing predicted frames folder...")
        shutil.rmtree(predicted_frames_folder_path) # почистим папку

    shutil.copytree(
        frames_folder_path,
        predicted_frames_folder_path
    )

    prediction_probabilities, prediction_statistics, emojisFrameData = vea_app.predict(predicted_frames_folder_path)

    _, analyzed_video_file_path, video_fps = vea_app.build_video_from_frames(video_file_path)
    _, final_video_path = vea_app.merge_video_with_audio(analyzed_video_file_path, extracted_audio_file_path, video_fps)
    print("\n\nfinal_video_path:", final_video_path, "\n\n")
    print("\n\n\n",emojisFrameData,"\n\n")

    return redirect(url_for("project", pjct_name = pjct_name))

# ---------------------------------------- [Загрузка видео из локального хранилища устройства] ----------------------------------------
# Uploading route
@flask.route('/upload', methods=['POST'])
def upload_file():
    project_name = None
    files = request.files.getlist('upload_video_file')
    for file in files:
        project_name = vea_app.create_project_local_storage(file)
    
    if len(files) > 1:
        return redirect(url_for("main"))
    elif project_name:
        return redirect(url_for("project", pjct_name = project_name))
    else:
        return redirect(url_for("main"))

#######################################################################################################
if __name__ == '__main__':
    flask.run(debug=True)
