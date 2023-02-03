import socket
import json
# import csv
import pandas as pd
from pandas import json_normalize
import pickle
#import seaborn as sns
import os
import numpy as np
import warnings
from datetime import datetime
import time
# import matplotlib.pyplot as plt

import streamlit as st
import plotly.express as px

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
# from sklearn.metrics import classification_report
# from sklearn.metrics import plot_confusion_matrix


from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.common.keys import Keys

# st.set_option('deprecation.showPyplotGlobalUse', False)
#Ignorer les warnings dans le terminal
warnings.filterwarnings("ignore")


#Récupère l'URL / l'enregistre dans le fichier .csv
def get_url():
    options = Options()

    driver = webdriver.Chrome(options=options)
    #driver.maximize_window()

    # ========== OPEN FIRST URL AND GET SECOND URL ==========
    driver.get("https://speedwayr-11-e2-bc/#/configure")

    #driver.implicitly_wait(1) # gives an implicit wait for 20 seconds

    #Passer sur la zone cmeIframe-jtxelq2f
    # driver.switch_to.frame(driver.find_element(By.ID, ""))

    #Trouver l'éléments Form1 -> correspond à la fenêtre QuickStrike intégrée au site
    folder = driver.find_element(By.XPATH, "/html[1]/body[1]/div[1]/nav[1]/div[1]/div[2]/ul[2]/li[1]/a[1]")
    folder.click()
    #On récupère l'URL de cette fenêtre QuickStrike
    URL = folder.get_property('')
    return URL

# get_url()

#Reflist
pathfile_reflist = 'C:\\Users\\adela\\Desktop\\notebook\\dock door RT student project - WIP-20230109T084500Z-001\\dock door RT student project - WIP\\reflist_clo'
Files = os.listdir(pathfile_reflist)
reflist=pd.DataFrame()
for file in Files:
    actual=file.rstrip('.csv')
    filename=os.path.join(pathfile_reflist, file)
    temp=pd.read_csv(filename, sep=';', names=['epc'])
    temp['actual'] = actual
    reflist = reflist.append(temp)
reflist = reflist.reset_index(drop=True)
#
#Window
#
with open('C:\\Users\\adela\\Desktop\\notebook\\dock door RT student project - WIP-20230109T084500Z-001\\dock door RT student project - WIP\\clf_window.pkl', 'rb') as f:
    clf_window = pickle.load(f)

window=3
aggFuncs = ['max', 'min', 'mean', 'sum', 'len']

timeRefs = [str(x) for x in range(window)]
timeRefs.append('window')

antennaRefs = ['ain', 'allants', 'aout']

Xcols_window_forced_multi = [ [aggfunc, timeref, antref] for aggfunc in aggFuncs \
                                     for timeref in timeRefs \
                                     for antref in antennaRefs ]
Xcols_window_forced_multi = pd.MultiIndex.from_tuples(Xcols_window_forced_multi).sort_values()

Xcols_ds_window = ['_'.join([str(y) for y in x]) for x in Xcols_window_forced_multi]
print("TEST WINDOW: " + str(Xcols_ds_window))
taille = len(Xcols_ds_window)
print(taille)
#
#EPC
#
with open('C:\\Users\\adela\\Desktop\\notebook\\dock door RT student project - WIP-20230109T084500Z-001\\dock door RT student project - WIP\\clf_EPC.pkl', 'rb') as f:
    clf_EPC = pickle.load(f)

window_EPC = 7
aggFuncs = ['max', 'min', 'mean', 'sum', 'len']

timeRefs = [str(x) for x in range(window_EPC)]
timeRefs.append('window')
# timeRefs

antennaRefs = ['ain', 'allants', 'aout']  # tags['Antenna_coverage'].unique().tolist()

Xcols_EPC_forced_multi = [[aggfunc, timeref, antref] for aggfunc in aggFuncs \
                          for timeref in timeRefs \
                          for antref in antennaRefs]
Xcols_EPC_forced_multi = pd.MultiIndex.from_tuples(Xcols_EPC_forced_multi).sort_values()

Xcols_ds_EPC = ['_'.join([str(y) for y in x]) for x in Xcols_EPC_forced_multi]
print("TEST EPC: " + str(Xcols_ds_EPC))
taille = len(Xcols_ds_EPC)
print(taille)
#
#On définit la fonction make_ds
#
def make_ds(tags_window, slot_col, Slots_id, index_ds, Xcols_forced):
    Slots_id_order = range(len(Slots_id))
    dict_slot_order = dict(zip(Slots_id, Slots_id_order))
    tags_window.loc[:, 'slot_id_order'] = tags_window.loc[:, slot_col].map(dict_slot_order)

    ds_slot_antcov = pd.pivot_table(data=tags_window, index=index_ds, columns=['slot_id_order', 'Antenna_coverage'],
                                    values='RSSI', fill_value=0,
                                    aggfunc=[max, min, np.mean, sum, len])
    #
    ds_slot_allants = pd.pivot_table(data=tags_window, index=index_ds, columns=['slot_id_order'], values='RSSI',
                                     fill_value=0,
                                     aggfunc=[max, min, np.mean, sum, len])
    Xcols = ds_slot_allants.columns
    # print(Xcols)
    Xcols = pd.MultiIndex.from_tuples([[str(y) for y in x] + ['allants'] for x in Xcols])
    ds_slot_allants.columns = Xcols
    #
    ds_window_antcov = pd.pivot_table(data=tags_window, index=index_ds, columns=['Antenna_coverage'],
                                      values='RSSI', fill_value=0, aggfunc=[max, min, np.mean, sum, len])
    Xcols = ds_window_antcov.columns
    Xcols = pd.MultiIndex.from_tuples([[x[0], 'window', x[1]] for x in Xcols])
    ds_window_antcov.columns = Xcols
    #
    if len(index_ds) != 0:
        ds_window_allants = pd.pivot_table(data=tags_window, index=index_ds, columns=None,
                                           values='RSSI', fill_value=0, aggfunc=[max, min, np.mean, sum, len])
        Xcols = ds_window_allants.columns
        Xcols = pd.MultiIndex.from_tuples([[x, 'window', 'allants'] for x in Xcols])
        ds_window_allants.columns = Xcols
    else:
        Series = tags_window['RSSI'].agg([max, min, np.mean, sum, len])
        ds_window_allants = pd.DataFrame(Series).transpose()
        Xcols = ds_window_allants
        Xcols = pd.MultiIndex.from_tuples([[x] + ['window', 'allants'] for x in Xcols])
        ds_window_allants.columns = Xcols

    ds = pd.concat([ds_slot_antcov, ds_slot_allants, ds_window_antcov, ds_window_allants], axis=1)

    Xcols = ds.columns
    Xcols = ['_'.join([str(y) for y in x]) for x in Xcols]
    ds.columns = Xcols

    ds = ds.reindex(Xcols_forced, fill_value=0, axis=1)

    window_id = '_'.join([str(x) for x in Slots_id])
    slot_center = Slots_id[int(len(Slots_id) / 2)]
    ds.loc[:, 'window_id'] = window_id
    ds.loc[:, 'slot_center'] = slot_center

    ds = ds.reset_index(drop=False)
    ds = ds[index_ds + ['window_id', 'slot_center'] + Xcols_forced]

    if 'crossing_id' in index_ds:
        sort_values = ['crossing_id', 'slot_center']
    else:
        sort_values = ['slot_center']
    ds = ds.sort_values(sort_values, ascending=True).reset_index(drop=True)
    #
    if 'epc' not in index_ds:
        EPCs_window = tags_window['epc'].nunique()
        Xcols_sum_len = [x for x in Xcols_forced if ('len' in x) or ('sum' in x)]
        if EPCs_window != 0:
            ds.loc[:, Xcols_sum_len] = ds.loc[:, Xcols_sum_len] / EPCs_window

    return ds
#
#Exemple d'une trame de donnée sous le format JSON :
#{"antennaPort":7,"epc":"999908290000000000000004","firstSeenTimestamp":"2018-06-14T00:15:54.36879Z","peakRssi":-51,"isHeartBeat":false}
#
#Je définis un dataframe avec le nom df qui possédera la totalité des tags lus par le lecteur :
#
df = pd.DataFrame(columns=['epc', 'firstSeenTimestamp', 'antennaPort', 'peakRssi', 'isHeartBeat'])
#
#Déclaration des différents attributs
timedelta = pd.Timedelta(1, unit='sec')
# dataframes with complete slots with start
tags_rt = pd.DataFrame()
Slots_rt = pd.DataFrame()

# tags with partial slot only
tags_partialSlot = pd.DataFrame()

# first slotStart
slotStart_last = np.nan
Antenna_coverage_dict = {1: 'ain', 2: 'ain', 3: 'aout', 4: 'aout'}
ds_window_display_rt = pd.DataFrame()

# Créez un socket TCP/IP
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#
# Connection au lecteur RFID
#
server_address = ('169.254.1.1', 14150)
sock.connect(server_address)
counter = 0
BUF_SIZE = 4500
# slot_crossing init
slot_crossing = np.nan

# Début du chronomètre
start = time.perf_counter()
# Début2 du chronomètre
start2 = time.perf_counter()
# Récupérer les données du lecteur RFID en boucle

list_slot_id = []
list_RSSI = []
list_EPC = []
list_prediction = []
list_timestamp = []
element1 = st.empty()
element2 = st.empty()
element3 = st.empty()

while True:
    data = sock.recv(BUF_SIZE) #buf size = 6000 octets // data : JSON FORMAT


    # Si la donnée reçue est une chaîne vide, sortez de la boucle while True, cela veut dire qu'on a dépassé la limite des 500 tags détéctés : NO LICENCE SPEEDWAY
    if data == b'':
        break
    #print("data :" + str(data))
    # Split the string into an array of substrings using "\r\n" as the delimiter
    my_array = data.decode().split("\r\n")

    tags_buffer = pd.DataFrame(columns=['epc', 'firstSeenTimestamp', 'antennaPort', 'peakRssi', 'isHeartBeat'])
    # Use a for loop to loop over the array
    for line in my_array:
        if line == '':
            continue
        try:
            # Print each element in the array (i.e. each line of the original string) //
            data_dict = json.loads(line)
        except json.decoder.JSONDecodeError:
            # Handle the error
            # print("Error: Unable to parse JSON document")
            print("")
        counter += 1

        if tags_buffer is None:
            tags_buffer = json_normalize(data_dict)
        else:
            tags_buffer = tags_buffer.append(json_normalize(data_dict), ignore_index=True)

    window = 3
    window_EPC = 7

    df = df.append(tags_buffer)
    # print(df)
    #Formatting
    tags_buffer['firstSeenTimestamp'] = pd.to_datetime(tags_buffer['firstSeenTimestamp'])
    tags_buffer['firstSeenTimestamp'] = tags_buffer['firstSeenTimestamp'].apply(lambda x: x.to_datetime64())
    tags_buffer['antennaPort'] = tags_buffer['antennaPort'].astype(int)
    tags_buffer['Antenna_coverage'] = tags_buffer['antennaPort'].map(Antenna_coverage_dict)
    tags_buffer['peakRssi'] = tags_buffer['peakRssi'].astype(str).apply(lambda x: x.replace(',', '.')).astype(float)
    #tags_buffer = tags_buffer.rename(columns={'peakRssi': 'RSSIdbm'})
    tags_buffer['RSSI'] = 10**6 * 10**(tags_buffer['peakRssi']/10)
    tags_buffer = tags_buffer.sort_values('firstSeenTimestamp', ascending=True)

    # initialization
    if slotStart_last is np.nan:
        Tmin = tags_buffer['firstSeenTimestamp'].min()
        slotStart_last = Tmin
        slot_id_last = 0

    # tags_partialSlot: detections which belongs to the next slot
    tags_partialSlot = tags_partialSlot.append(tags_buffer)

    Tmax = tags_buffer['firstSeenTimestamp'].max()

        # next slot reached
    if Tmax >= (slotStart_last + timedelta):
        Slots_new = pd.DataFrame({'slotStart': pd.date_range(start=slotStart_last, end=Tmax, freq=timedelta)})
        Slots_new['slot_id'] = range(slot_id_last, slot_id_last + len(Slots_new))
        slotStart_last = Slots_new.loc[len(Slots_new) - 1, 'slotStart']
        slot_id_last = Slots_new.loc[len(Slots_new) - 1, 'slot_id']

        tags_partialSlot = tags_partialSlot.sort_values('firstSeenTimestamp', ascending=True)

        Slots_new = Slots_new.sort_values('slotStart', ascending=True)

        tags_slot = pd.merge_asof(tags_partialSlot, Slots_new, left_on='firstSeenTimestamp', right_on='slotStart',
                                      direction='nearest')
        #
        tags_fullSlot = tags_slot[tags_slot['slot_id'] != tags_slot['slot_id'].max()]
        Slots_fullSlot = Slots_new[Slots_new['slot_id'] != Slots_new['slot_id'].max()]

        tags_partialSlot = tags_slot[tags_slot['slot_id'] == tags_slot['slot_id'].max()] \
            .drop(columns=['slotStart', 'slot_id'])

        #
        # append tags_rt and Slots_rt
        #
        tags_rt = tags_rt.append(tags_fullSlot).reset_index(drop=True)
        Slots_rt = Slots_rt.append(Slots_fullSlot).reset_index(drop=True)
        #
        # window prediction
        #
        Slots_id = Slots_rt.nlargest(window, 'slot_id')['slot_id'].tolist()
        Slots_id.sort()
        slot_last = max(Slots_id)
        #print(Slots_id)

        if len(Slots_id) == window:
            tags_rt_window = tags_rt[tags_rt['slot_id'].isin(Slots_id)] \
                    .sort_values('firstSeenTimestamp').reset_index(drop=True)
            #
            # st.dataframe(tags_rt_window)
            index_ds = []
            slot_col = 'slot_id'
            ds_window = make_ds(tags_rt_window, slot_col, Slots_id, index_ds, Xcols_ds_window)
            #
            # st.dataframe(ds_window)
            X = ds_window.loc[:, Xcols_ds_window]

            ypred = clf_window.predict(X)
            #print("ypred = " + str(ypred))

            ds_window.loc[:, 'pred_ml'] = ypred
            #ds_window['pred_ml'] = ypred
            ds_window = ds_window[['window_id', 'slot_center', 'pred_ml'] + Xcols_ds_window]
            # Fin2 du chronomètre
            end2 = time.perf_counter()
            # Temps d'exécution en secondes
            print(f"Le code pour un slot_id a pris {end2 - start2:.4f} secondes pour s'exécuter.")
            start2 = end2

            ##### affichage flask
            list_slot_id.append(Slots_id)
            list_RSSI.append(tags_buffer["peakRssi"][0])
            list_prediction.append(str(ypred[0]))
            list_timestamp.append(tags_buffer['firstSeenTimestamp'][0])

            # print(list_slot_id)
            dict_flask = {"RSSI": list_RSSI, "prediction":  list_prediction, "timestamp" : list_timestamp }
            # dict_flask = {"RSSI": tags_buffer["peakRssi"], "prediction":  ypred[0]}
            df_flask = pd.DataFrame(dict_flask)
            # element1.table(df_flask)
            fig = px.scatter(df_flask, x=df_flask["timestamp"], y=df_flask["RSSI"], color=df_flask["prediction"], color_discrete_map={"crossing" :"red","no_crossing":"blue"})
            # print(df_flask)

            # two_subplot_fig = plt.figure(figsize=(30,30))
            # plt.subplot(422)
            # plt.xlabel('Temps')
            # plt.ylabel('Prediction')
            # plt.title('Signal carré')


            i=0
            etat_pre = df_flask["prediction"][i]
            x_array = []
            taille = len(df_flask)

            while taille >1 :
                x = np.arange(df_flask["timestamp"][i], df_flask["timestamp"][i+1] , 20)

                etat= df_flask["prediction"][i]
                if(etat_pre != etat):
                    print(df_flask["timestamp"][i])
                    x_array.append(df_flask["timestamp"][i])
                etat_pre = etat

                next=int(df_flask["timestamp"][i+1].strftime("%S"))
                if etat == "crossing" :
                    y = np.where(x < df_flask["timestamp"][i+1],"crossing" , "no_crossing" )

                else :
                    y = np.where(x < df_flask["timestamp"][i+1],"no_crossing" , "crossing" )

                # Dessin du signal
                i=i+1
                taille=taille-1


                # plt.plot(x, y, color='black', linestyle='--')
                # add vertical line
                # plt.vlines(x=x_array, ymin="no_crossing", ymax="crossing", colors='black', ls='--')

                #time.sleep(0.1)

            element3.pyplot(two_subplot_fig)

            # st.dataframe(df_flask)
            element2.plotly_chart(fig, theme="streamlit", use_conatiner_width=True)
            print("Slot :" + str(Slots_id) + str(tags_buffer["peakRssi"]), ypred[0])
            # st.metric(Slots_id, ypred[0])

            # st.write(Slots_id)
            dict_line = {"prediction" : list_prediction, "timestamp": list_timestamp}
            df_line = pd.DataFrame(dict_line)
            # element3.line_chart(df_line, y = "timestamp", x = "prediction")

            #####
            ds_window_display_rt = ds_window_display_rt.append(ds_window).reset_index(drop=True)
    
            if 'crossing' in ds_window_display_rt['pred_ml'].tolist():
                idx_max_crossing = ds_window_display_rt [ ds_window_display_rt['pred_ml']=='crossing' ]['slot_center'].max()
                #print(idx_max_crossing)

                ds_window_after_crossing = ds_window_display_rt.iloc [idx_max_crossing:]
                Q_no_crossing = len(ds_window_after_crossing [ ds_window_after_crossing['pred_ml'] == 'no_crossing' ])
                # print(Q_no_crossing)
                if Q_no_crossing == 2: # ancienne valeur = 2
                    slot_crossing = int(ds_window_display_rt \
                                        [ ds_window_display_rt['pred_ml']=='crossing' ]['slot_center'].mean())
                    print(f'slot_crossing: {slot_crossing}')
                    slot_last = max(Slots_id)
            # else :
            #     print("Slot :" + str(Slots_id), ypred[0])
#
# EPC prediction
#
            if slot_crossing is not np.nan:
                if (slot_last-slot_crossing) > int(window_EPC/2):
                    start3 = time.perf_counter()
                    Slots_id = range(slot_crossing-int(window_EPC/2), slot_crossing+int(window_EPC/2)+1)
                    tags_rt_window = tags_rt [ tags_rt ['slot_id'] .isin(Slots_id) ] \
                            .sort_values('firstSeenTimestamp').reset_index(drop=True)
#
                    index_ds=['epc']
                    slot_col='slot_id'
                    ds_EPC_rt = make_ds(tags_rt_window, slot_col, Slots_id, index_ds, Xcols_ds_EPC)
#
                    X = ds_EPC_rt.loc[:, Xcols_ds_EPC]
                    # st.dataframe(X)
                    ypred = clf_EPC.predict(X)
                    ds_EPC_rt.loc[:, 'pred_ml'] = ypred
                    stop3 = time.perf_counter()
                    print(f"Le code pour epc_predict a pris {stop3 - start3:.4f} secondes pour s'exécuter.")
                    ds_EPC_rt = ds_EPC_rt [['epc', 'window_id', 'slot_center', 'pred_ml'] + Xcols_ds_EPC ]
                    print("DS EPC RT :" + str(ds_EPC_rt.shape))
#
# ds_EPC_rt display
#
                    ds_EPC_rt_display = pd.merge(ds_EPC_rt, reflist, on='epc')
                    ds_EPC_rt_display ['predbool_ml'] = (ds_EPC_rt_display ['actual']==ds_EPC_rt_display ['pred_ml'])
                    ds_EPC_rt_display = ds_EPC_rt_display [ ['epc', 'window_id', 'slot_center', 'pred_ml', 'predbool_ml', 'actual'] \
                                        + Xcols_ds_EPC ]
                    print(ds_EPC_rt_display.shape)
#             accuracy
                    accuracy = (ds_EPC_rt_display['actual']==ds_EPC_rt_display['pred_ml']).mean()
                    print(f'Slots_id: {Slots_id}, EPC accuracy: {accuracy}')
#         ConfusionMatrix
                    st.subheader("Confusion Matrix")
                    ConfusionMatrixDisplay.from_predictions(ds_EPC_rt_display['actual'], ds_EPC_rt_display['pred_ml'])
                    # plot_confusion_matrix(clf_EPC,X= X,y_true = ypred)
                    # plot_confusion_matrix(ds_EPC_rt_display['actual'], ds_EPC_rt_display['pred_ml'])
                    st.pyplot() 
                    break
            # print("pas bon")
# Fin du chronomètre
end = time.perf_counter()
# Fermez la connexion
sock.close()

print(df.shape)
taille = len(tags_rt)
print(taille)
tags_rt['firstSeenTimestamp'] = pd.to_datetime(tags_rt['firstSeenTimestamp'])
tags_rt = tags_rt.drop(columns=['isHeartBeat', 'Antenna_coverage', 'RSSI', 'slotStart', 'slot_id'])
# Générez l'horodatage actuel au format "année-mois-jour_heure-minute-seconde"
times2 = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Enregistrez le dataframe dans un fichier CSV avec un nom de fichier qui inclut l'horodatage
print(tags_rt)
tags_rt.to_csv(f"TagsysTest__ko=3m__Tilt_angle=70__run=multi__{times2}.csv", index=False)


# Temps d'exécution en secondes
print(f"Le code a pris {end - start:.4f} secondes pour s'exécuter.")



# Générer une chaîne de caractères contenant le contenu du DataFrame
# output = tags_buffer.to_string()

# Afficher le contenu du DataFrame dans un terminal ou une fenêtre de sortie
# print(output)


