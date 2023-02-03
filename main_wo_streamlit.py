import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import classification_report

import pickle
import time
import json
import socket
from pandas import json_normalize
import streamlit as st
import plotly.express as px


# 
# make_ds
# 
def make_ds(tags_window, slot_col, Slots_id, index_ds, Xcols_forced):
    
    Slots_id_order = range(len(Slots_id))
    dict_slot_order = dict(zip(Slots_id, Slots_id_order))
    tags_window.loc[:, 'slot_id_order'] = tags_window.loc[:, slot_col].map(dict_slot_order)
    
    ds_slot_antcov = pd.pivot_table(data=tags_window, index = index_ds, columns=['slot_id_order', 'Antenna_coverage'], values='RSSI', fill_value=0, \
               aggfunc=[max, min, np.mean, sum, len])
#     
    ds_slot_allants = pd.pivot_table(data=tags_window, index = index_ds, columns=['slot_id_order'], values='RSSI', fill_value=0, \
               aggfunc=[max, min, np.mean, sum, len])
    Xcols = ds_slot_allants.columns
    Xcols = pd.MultiIndex.from_tuples([ [str(y) for y in x]+ ['allants'] for x in Xcols])
    ds_slot_allants.columns = Xcols
#     
    ds_window_antcov = pd.pivot_table(data=tags_window, index=index_ds, columns=['Antenna_coverage'],\
                  values='RSSI', fill_value=0, aggfunc=[max, min, np.mean, sum, len])
    Xcols = ds_window_antcov.columns
    Xcols = pd.MultiIndex.from_tuples([ [x[0], 'window', x[1]] for x in Xcols])
    ds_window_antcov.columns = Xcols
#     
    if len(index_ds) !=0 :
        ds_window_allants = pd.pivot_table(data=tags_window, index=index_ds, columns=None, \
                                       values='RSSI', fill_value=0, aggfunc=[max, min, np.mean, sum, len])
        Xcols = ds_window_allants.columns
        Xcols = pd.MultiIndex.from_tuples([ [x, 'window', 'allants'] for x in Xcols])
        ds_window_allants.columns = Xcols
    else:
        Series = tags_window['RSSI'].agg([max, min, np.mean, sum, len])
        ds_window_allants = pd.DataFrame(Series).transpose()
        Xcols = ds_window_allants
        Xcols = pd.MultiIndex.from_tuples([[x]+['window', 'allants'] for x in Xcols])
        ds_window_allants.columns = Xcols

    ds = pd.concat([ds_slot_antcov, ds_slot_allants, ds_window_antcov, ds_window_allants], axis=1)

    Xcols = ds.columns
    Xcols = ['_'.join([str(y) for y in x]) for x in Xcols]
    ds.columns = Xcols
    
    ds = ds.reindex(Xcols_forced, fill_value=0, axis=1)
    
    window_id = '_'.join([str(x) for x in Slots_id])
    slot_center = Slots_id[int(len(Slots_id)/2)]
    ds.loc[:, 'window_id']=window_id
    ds.loc[:, 'slot_center']=slot_center
    
    ds = ds.reset_index(drop=False)
    ds = ds [index_ds+['window_id', 'slot_center']+Xcols_forced]
    
    if 'crossing_id' in index_ds:
        sort_values=['crossing_id', 'slot_center']
    else:
        sort_values=['slot_center']
    ds = ds.sort_values(sort_values, ascending=True).reset_index(drop=True)
#     
    if 'EPC' not in index_ds:
        EPCs_window = tags_window['EPC'].nunique()
        Xcols_sum_len = [x for x in Xcols_forced if ('len' in x) or ('sum' in x)]
        if EPCs_window != 0:
            ds.loc [:, Xcols_sum_len] = ds.loc [:, Xcols_sum_len] / EPCs_window

    return ds
# 
# def visu
# 
def visu(df, EPC , Slots):

    Tmin=df['Timestamp'].min()
    Tmax=df['Timestamp'].max()
    RSSImin=df['RSSIdbm'].min()
    RSSImax=df['RSSIdbm'].max()
    
    Slots_df = Slots [ (Slots['slotStart']>=Tmin) & (Slots['slotStart']<=Tmax) ]
    
    if EPC is not None:
        df = df [ df['EPC']==EPC ]
    plt.figure(figsize=(14,6))
    
    dict_actual_marker={'moving':'o', 'stationary':'+'}
    dict_Antenna_coverage_color={'ain':'blue', 'aout':'red' }
    
    for key, df_key in df.groupby(['actual', 'Antenna_coverage']):
        actual=key[0]
        Antenna_coverage=key[1]
        marker=dict_actual_marker[actual]
        color=dict_Antenna_coverage_color[Antenna_coverage]
        sns.scatterplot(data=df_key, x='Timestamp', y='RSSIdbm', color=color, marker=marker)

    plt.vlines(Slots_df['slotStart'], ymin=RSSImin, ymax=RSSImax, linestyle='dashed')
    for i, row in Slots_df.iterrows():
        slotStart=row['slotStart']
        slot_id=row['slot_id']
        plt.annotate(slot_id, (slotStart,RSSImax))
    plt.xlim(Tmin, Tmax)
    plt.show()
# 
# reflist
# 
pathfile_reflist = 'C:\\Users\\adela\\Desktop\\notebook\\dock door RT student project - WIP-20230109T084500Z-001\\dock door RT student project - WIP\\reflist_clo'
Files = os.listdir(pathfile_reflist)
reflist=pd.DataFrame()
for file in Files:
    actual=file.rstrip('.csv')
    filename=os.path.join(pathfile_reflist, file)
    temp=pd.read_csv(filename, sep=';', names=['EPC'])
    temp['actual'] = actual
    reflist = reflist.append(temp)
reflist = reflist.reset_index(drop=True)
reflist.head()
# 
# 
# 
EPCs_reflist = reflist.groupby('actual')['EPC'].nunique() 
EPCs_reflist = pd.DataFrame({'EPCs_reflist':EPCs_reflist}).reset_index(drop=False)
EPCs_reflist
# 
# window
# 
with open('C:\\Users\\adela\\Desktop\\notebook\\dock door RT student project - WIP-20230109T084500Z-001\\dock door RT student project - WIP\\clf_window.pkl', 'rb') as f:
    clf_window = pickle.load(f)

window=3
aggFuncs = ['max', 'min', 'mean', 'sum', 'len']

timeRefs = [str(x) for x in range(window)]
timeRefs.append('window')

antennaRefs = ['ain', 'allants', 'aout']
# 
Xcols_window_forced_multi = [ [aggfunc, timeref, antref] for aggfunc in aggFuncs \
                                     for timeref in timeRefs \
                                     for antref in antennaRefs ]
Xcols_window_forced_multi = pd.MultiIndex.from_tuples(Xcols_window_forced_multi).sort_values()

Xcols_ds_window = ['_'.join([str(y) for y in x]) for x in Xcols_window_forced_multi]
print(Xcols_ds_window)
# 
# EPC
# 
with open('C:\\Users\\adela\\Desktop\\notebook\\dock door RT student project - WIP-20230109T084500Z-001\\dock door RT student project - WIP\\clf_EPC.pkl', 'rb') as f:
    clf_EPC = pickle.load(f)
    
window_EPC=7
aggFuncs = ['max', 'min', 'mean', 'sum', 'len']

timeRefs = [str(x) for x in range(window_EPC)]
timeRefs.append('window')
timeRefs

antennaRefs = ['ain', 'allants', 'aout']# tags['Antenna_coverage'].unique().tolist()
# 
Xcols_EPC_forced_multi = [ [aggfunc, timeref, antref] for aggfunc in aggFuncs \
                                     for timeref in timeRefs \
                                     for antref in antennaRefs ]
Xcols_EPC_forced_multi = pd.MultiIndex.from_tuples(Xcols_EPC_forced_multi).sort_values()

Xcols_ds_EPC = ['_'.join([str(y) for y in x]) for x in Xcols_EPC_forced_multi]
print(Xcols_ds_EPC)
# 
# json
# 
# partial frame
data_tailPartialFrame_last=''
# 
errors_json=0
# 
tags_partialSlot=pd.DataFrame()
# 
Cols_json = ['antennaPort', 'epc', 'firstSeenTimestamp', 'peakRssi']
Cols = ['Antenna', 'EPC', 'Timestamp', 'RSSI']
dict_Cols = dict(zip(Cols_json, Cols))
# 
# dataframes with complete slots with start
# 
tags=pd.DataFrame()
Slots=pd.DataFrame()
ds_window_display = pd.DataFrame()
ds_window_crossing = pd.DataFrame()
# 
slot_crossing_list=[]
# 
window=3
window_EPC=7
# 
# init first slotStart, slot_id: first Timestamp
# 
slotStart_last=np.nan

slot_crossing = np.nan
# 
# slot width = 1sec
# 
timedelta=pd.Timedelta(1, unit='s')
#
# Antenna_coverage
# 
Antenna_coverage_dict = {1:'ain', 2:'ain', 3:'aout', 4:'aout'}
# 
# tcp
# 
TCP_IP = '169.254.1.1'    
TCP_PORT = 14150
server_address = (TCP_IP, TCP_PORT)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# BUF_SIZE = 138
BUF_SIZE = 2000
# 
# loop
# 
# data collection starts from connect
sock.connect((TCP_IP, TCP_PORT))


start = time.time()

####################################################################################
###                                                                             ####
###                  PARTIE STREAMLIT                                           ####
###                                                                             ####
####################################################################################

list_slot_id = []
list_RSSI = []
list_EPC = []
list_prediction = []
list_timestamp = []
element1 = st.empty()
element2 = st.empty()
element3 = st.empty()


try:
    while True:
#         buffer retieval and head and tail slot possible repair
        data = sock.recv(BUF_SIZE)
        data = data.decode().split('\r\n')
#     
        data_headPartialFrame = data [0]
        data_tailPartialFrame = data [-1]
        data_fullFrames = data[1:-1]
    
        data_headFullFrame = data_tailPartialFrame_last + data_headPartialFrame
        data_fullFrames.insert(0, data_headFullFrame)
        data_tailPartialFrame_last = data_tailPartialFrame
    
        if len(data_fullFrames)!=0:
            data_fullFrames_json = '\n'.join(data_fullFrames)
            try:
                tags_buffer = pd.read_json(data_fullFrames_json, lines=True)
#        column name change         
                tags_buffer = tags_buffer.drop(columns=['isHeartBeat']).rename(columns=dict_Cols)
# formatting
                tags_buffer ['Timestamp'] = pd.to_datetime(tags_buffer ['Timestamp'])
                tags_buffer['Timestamp'] = tags_buffer['Timestamp'].apply(lambda x:x.to_datetime64())
                tags_buffer['Antenna'] = tags_buffer['Antenna'].astype(int)
                tags_buffer['Antenna_coverage'] = tags_buffer['Antenna'].map(Antenna_coverage_dict)
                tags_buffer['RSSI'] = tags_buffer['RSSI'].apply(lambda x:str(x).replace(',','.')).astype(float)
                tags_buffer = tags_buffer.rename(columns={'RSSI':'RSSIdbm'})
                tags_buffer['RSSI'] = 10**6 * 10**(tags_buffer['RSSIdbm']/10) # microW
                tags_buffer = tags_buffer.sort_values('Timestamp', ascending=True)    
# first loop: slotStart and slot_id init
                if slotStart_last is np.nan:
                    Tmin=tags_buffer['Timestamp'].min()
                    slotStart_last=Tmin
                    slot_id_last=0       
# tags_partialSlot: detections which belongs to the next slot
                tags_partialSlot = tags_partialSlot.append(tags_buffer)
#     
                Tmax = tags_buffer['Timestamp'].max()
# until slot n+1 is reached, append tags_partialSlot
                if Tmax >= (slotStart_last+timedelta):
# slot >= n+1 reached
                    Slots_new = pd.DataFrame({'slotStart':pd.date_range(start=slotStart_last, end=Tmax, freq=timedelta)})
                    Slots_new['slot_id']=range(slot_id_last, slot_id_last+len(Slots_new))
                    slotStart_last=Slots_new.loc[len(Slots_new)-1, 'slotStart']
                    slot_id_last=Slots_new.loc[len(Slots_new)-1, 'slot_id']
#         
                    tags_partialSlot = tags_partialSlot.sort_values('Timestamp', ascending=True)
                    Slots_new = Slots_new.sort_values('slotStart', ascending=True)
#         
                    tags_slot = pd.merge_asof(tags_partialSlot, Slots_new, left_on='Timestamp', right_on='slotStart', direction='nearest')
#         
                    tags_fullSlot = tags_slot [ tags_slot['slot_id']!=tags_slot['slot_id'].max() ] 
                    Slots_fullSlot = Slots_new [ Slots_new['slot_id']!=Slots_new['slot_id'].max() ] 
#         
# last slot is uncomplete ==> tags_partialSlot
# 
                    tags_partialSlot = tags_slot [ tags_slot['slot_id']==tags_slot['slot_id'].max() ]\
                                .drop(columns=['slotStart', 'slot_id'])
#         
# append tags and Slots
# 
                    tags = tags.append(tags_fullSlot).reset_index(drop=True)
                    Slots = Slots.append(Slots_fullSlot).reset_index(drop=True)            
#     
# window prediction
# 
# largest windows
                    Slots_id = Slots.nlargest(window, 'slot_id') ['slot_id'].tolist() 
                    Slots_id.sort()
                    slot_last = max(Slots_id)
                    if len(Slots_id)==window:
                        tags_window = tags [ tags ['slot_id'] .isin(Slots_id) ] \
                            .sort_values('Timestamp').reset_index(drop=True)
#       ds_window
                        index_ds=[]
                        slot_col='slot_id'
                        ds_window = make_ds(tags_window, slot_col, Slots_id, index_ds, Xcols_ds_window)
#       window prediction
                        X = ds_window.loc[:, Xcols_ds_window]
                        ypred=clf_window.predict(X)
                        ds_window['pred_ml']=ypred
                        ds_window = ds_window [['window_id', 'slot_center', 'pred_ml']+Xcols_ds_window]
                        print(Slots_id, ypred[0])



                        #STREAMLIT


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

                        two_subplot_fig = plt.figure(figsize=(30,30))
                        plt.subplot(422)
                        plt.xlabel('Temps')
                        plt.ylabel('Prediction')
                        plt.title('Signal carré')


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


                            plt.plot(x, y, color='black', linestyle='--')
                            # add vertical line
                            plt.vlines(x=x_array, ymin="no_crossing", ymax="crossing", colors='black', ls='--')

                            #time.sleep(0.1)

                        element3.pyplot(two_subplot_fig)

                        # st.dataframe(df_flask)
                        element2.plotly_chart(fig, theme="streamlit", use_conatiner_width=True)
                        print("Slot :" + str(Slots_id) + str(tags_buffer["peakRssi"]), ypred[0])
                        # st.metric(Slots_id, ypred[0])

                        # st.write(Slots_id)
                        dict_line = {"prediction" : list_prediction, "timestamp": list_timestamp}
                        df_line = pd.DataFrame(dict_line)



                        ds_window_crossing = ds_window_crossing.append(ds_window).reset_index(drop=True)
                        ds_window_display = ds_window_display.append(ds_window)           
#         slot_crossing
                        if 'crossing' in ds_window_crossing['pred_ml'].tolist():
                            idx_max_crossing = ds_window_crossing [ ds_window_crossing['pred_ml']=='crossing' ]['slot_center'].idxmax()
                            ds_window_after_crossing = ds_window_crossing.iloc [idx_max_crossing+1:] 
                            Q_no_crossing = len(ds_window_after_crossing [ ds_window_after_crossing['pred_ml'] == 'no_crossing' ])
#                             needs 2 consecutive no_crossing to calculate slot_crossing
                            if Q_no_crossing == 2:
                                slot_crossing = int(ds_window_crossing \
                                        [ ds_window_crossing['pred_ml']=='crossing' ]['slot_center'].mean())
                                slot_crossing_list.append(slot_crossing)
                                print(f'slot_crossing: {slot_crossing}')
                                ds_window_crossing = pd.DataFrame()
#                     
# EPC prediction
# 
#  needs a crossing window
                        if slot_crossing is not np.nan:
#   needs enough slots
                            if (slot_last-slot_crossing) >= int(window_EPC/2):
                                Slots_id = range(slot_crossing-int(window_EPC/2), slot_crossing+int(window_EPC/2)+1)
                                tags_window = tags [ tags ['slot_id'] .isin(Slots_id) ] \
                                                        .sort_values('Timestamp').reset_index(drop=True)
#       ds_EPC            
                                index_ds=['EPC']
                                slot_col='slot_id'
                                ds_EPC = make_ds(tags_window, slot_col, Slots_id, index_ds, Xcols_ds_EPC)
#             
                                X = ds_EPC.loc[:, Xcols_ds_EPC]
                                ypred = clf_EPC.predict(X)
                                ds_EPC.loc[:, 'pred_ml'] = ypred
                                ds_EPC = ds_EPC [['EPC', 'window_id', 'slot_center', 'pred_ml'] + Xcols_ds_EPC ]
#                 
#     ds_EPC_display
# 
                                ds_EPC_display = pd.merge(ds_EPC, reflist, on='EPC')
                                ds_EPC_display ['predbool_ml'] = (ds_EPC_display ['actual']==ds_EPC_display ['pred_ml'])
                                ds_EPC_display = ds_EPC_display [ ['EPC', 'window_id', 'slot_center', 'pred_ml', 'predbool_ml', 'actual'] \
                                        + Xcols_ds_EPC ]
#       detection
                                EPCs_detection = ds_EPC_display.groupby('actual')['EPC'].nunique()
                                EPCs_detection = pd.DataFrame({'EPCs_detected':EPCs_detection}).reset_index(drop=False)

                                EPCs_detection = pd.merge(EPCs_detection, EPCs_reflist, on='actual')
                                EPCs_detection['%'] = EPCs_detection.apply(lambda x:100*x['EPCs_detected']/x['EPCs_reflist'], axis=1)
                                (print(EPCs_detection))
#       clf accuracy
                                accuracy = (ds_EPC_display['actual']==ds_EPC_display['pred_ml']).mean()
                                print(f'Slots_id: {Slots_id}, EPC accuracy: {accuracy}')
#         ConfusionMatrix
                                ConfusionMatrixDisplay.from_predictions(ds_EPC_display['actual'], ds_EPC_display['pred_ml'])
                                plt.show()
#     
                                slot_crossing=np.nan
                
                
            except:
                print(f'json decoding issue, frames lost: {len(data_fullFrames)}')
#                 frame by frame

except KeyboardInterrupt:
    print('interrupted!')
    errors_json += 1

end = time.time()
print(end-start)
    
sock.close()

Tmax - tags['Timestamp'].min()