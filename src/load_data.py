import pandas as pd
import os


def load_data_TESS(Tess):
    tess_directory_list = os.listdir(Tess)

    file_emotion = []
    file_path = []

    for dir in tess_directory_list:
        directories = os.listdir(Tess + dir)
        for file in directories:
            part = file.split('.')[0]
            part = part.split('_')[2]
            if part=='ps':
                file_emotion.append('surprise')
            else:
                file_emotion.append(part)
            file_path.append(Tess + dir + '/' + file)
            
    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Tess_df = pd.concat([emotion_df, path_df], axis=1)
    # Display the first few rows of Tess_df
    Tess_df.head()
    return Tess_df



def load_data_SAVEE(Savee):
    savee_directory_list = os.listdir(Savee)

    file_emotion = []
    file_path = []

    for file in savee_directory_list:
        file_path.append(Savee + file)
        part = file.split('_')[1]
        ele = part[:-6]
        if ele=='a':
            file_emotion.append('angry')
        elif ele=='d':
            file_emotion.append('disgust')
        elif ele=='f':
            file_emotion.append('fear')
        elif ele=='h':
            file_emotion.append('happy')
        elif ele=='n':
            file_emotion.append('neutral')
        elif ele=='sa':
            file_emotion.append('sad')
        else:
            file_emotion.append('surprise')
            
    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Savee_df = pd.concat([emotion_df, path_df], axis=1)
    Savee_df.head()
    return Savee_df



def load_data_RAVDESS(Ravdess):
    ravdess_directory_list = os.listdir(Ravdess)

    file_emotion = []
    file_path = []

    for dir in ravdess_directory_list:
        # as there are 20 different actors in our previous directory, we need to extract files for each actor
        actor = os.listdir(Ravdess + dir)
        for file in actor:
            part = file.split('.')[0]
            part = part.split('-')
            # the third part in each file represents the emotion associated with that file
            file_emotion.append(int(part[2]))
            file_path.append(Ravdess + dir + '/' + file)
            
    # DataFrame for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # DataFrame for path of files
    path_df = pd.DataFrame(file_path, columns=['Path'])

    # Concatenating the DataFrames
    Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

    # Replacing integers with actual emotions
    Ravdess_df['Emotions'] = Ravdess_df['Emotions'].replace({
        1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'
    })

    Ravdess_df.head()
    return Ravdess_df

def load_data():
    Tess = "data/TESS/tess toronto emotional speech set data/TESS Toronto emotional speech set data/"
    Savee = "data/SAVEE/ALL/"
    Ravdess = "data/RAVDESS/audio_speech_actors_01-24/"
    
    Tess_df = load_data_TESS(Tess)
    Savee_df= load_data_SAVEE(Savee)
    Ravdess_df= load_data_RAVDESS(Ravdess)
    
    data_path = pd.concat([Tess_df, Savee_df, Ravdess_df], axis = 0)
    data_path.to_csv("csvResults/data_path.csv",index=False)
    data_path.head()
    return data_path
    
