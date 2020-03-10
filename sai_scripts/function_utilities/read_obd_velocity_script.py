# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 13:13:42 2018

@author: lenovo
"""
import numpy as np
from scipy import interpolate
import time

def convert24(hrs_mins_secs,am_pm): 
      
    # Checking if last two elements of time 
    # is AM and first two elements are 12 
    if am_pm == "AM" and hrs_mins_secs[:2] == "12": 
        return "00" + hrs_mins_secs[2::] 
          
    # remove the AM     
    elif am_pm == "AM": 
        return hrs_mins_secs
      
    # Checking if last two elements of time 
    # is PM and first two elements are 12    
    elif am_pm == "PM" and hrs_mins_secs[:2] == "12": 
        return hrs_mins_secs 
          
    else: 
          
        # add 12 to hours and remove PM 
        return str(int(hrs_mins_secs[:2]) + 12) + hrs_mins_secs[2::]


fpath = 'D:\\Steradian\\Radar_Data\\recorded_data\\driveData_withinArcade_withoutLid_07.12.2018\\'
flag_obd_data_available = 1
if flag_obd_data_available:
    ptrfilename = 'Obd_Data'
    obd_pntr = np.load(fpath+'obd_data\\'+ptrfilename+'.npy')
    frame_num = obd_pntr[:,0].astype('int')
    line_num = obd_pntr[:,1].astype('int')
    frame_abs_time = obd_pntr[:,2]
    frame_start_time_24_hr = time.ctime(frame_abs_time[0]).split(' ')[-2]
    hrs_mins_sec_frame_start = frame_start_time_24_hr.split(':')
    hrs_frame_start = np.float64(hrs_mins_sec_frame_start[0])
    mins_frame_start = np.float64(hrs_mins_sec_frame_start[1])
    secs_frame_start = np.float64(hrs_mins_sec_frame_start[2]) + (frame_abs_time[0] - int(frame_abs_time[0])) # To get milliseconds precision
    frame_relative_time = frame_abs_time-frame_abs_time[0]
    csvfilename =  'CSVLog_20181207_182021'   
    obd_data_list = []
    cnt = 0    
    with open(fpath+'obd_data\\'+csvfilename+'.csv') as file:
        for line in file:
            tmp = line.strip('\n')
            tmp = tmp.split(',',2)
            if cnt < 1:
                dateTime = tmp[0].split(' ')[3::]
                date = dateTime[0]
                time_12_hr = dateTime[1]
                am_pm = dateTime[2]
                time_24_hr = convert24(time_12_hr,am_pm)
                hrs_mins_secs = time_24_hr.split(':')
                hrs = np.float64(hrs_mins_secs[0])
                mins = np.float64(hrs_mins_secs[1])
                secs = np.float64(hrs_mins_secs[2])
            if cnt<2 : # 2 because it has two lines of strings and we dont want the string lines
                obd_data_list.append(['0','0'])
            else:
                obd_data_list.append(tmp)
            cnt +=1
    
    if (len(obd_data_list) > 0):      
        raw_obd_data = np.array(obd_data_list).astype('float64')
        Qmin,Rsec = np.divmod(secs+raw_obd_data[:,0],60)
        secs_new = Rsec
        Qhrs,Rmin = np.divmod(mins+Qmin,60)
        mins_new = Rmin
        Qdays,Rhrs = np.divmod(hrs+Qhrs,24)
        hrs_new = Rhrs
        abstime_raw_obd_data = np.concatenate((hrs_new[:,None],mins_new[:,None],secs_new[:,None],raw_obd_data[:,1][:,None]),axis=1)
        match_hrs_ind = np.where(abstime_raw_obd_data[:,0]==hrs_frame_start)
        match_min_ind = np.where(abstime_raw_obd_data[:,1]==mins_frame_start) 
        match_hrs_min_ind = np.intersect1d(match_hrs_ind[0],match_min_ind[0])
        temp = (np.abs(abstime_raw_obd_data[match_hrs_min_ind,2]-secs_frame_start)).argmin()
        trueLineNum = match_hrs_min_ind[temp]
        obd_data_frame_sync = raw_obd_data[trueLineNum::,:]
        obd_data_frame_sync[:,0] = obd_data_frame_sync[:,0] - obd_data_frame_sync[0,0]
        vel_time_interp = interpolate.interp1d(obd_data_frame_sync[:,0],obd_data_frame_sync[:,1])
        
    else:
        vel_time_interp = None 
else:
    obd_vel_kmph = 0






### Apu's approach
#if flag_obd_data_available:
#    ptrfilename = 'Obd_Data'
#    obd_pntr = np.load(fpath+'obd_data\\'+ptrfilename+'.npy')
#    frame_num = obd_pntr[:,0].astype('int')
#    line_num = obd_pntr[:,1].astype('int')
#    frame_abs_time = obd_pntr[:,2]
#    frame_relative_time = frame_abs_time-frame_abs_time[0]
#    csvfilename =  'CSVLog_20181205_161950'   
#    obd_data_list = []
#    cnt = 0    
#    with open(fpath+'obd_data\\'+csvfilename+'.csv') as file:
#        for line in file:
#            tmp = line.strip('\n')
#            tmp = tmp.split(',',2)
#            if cnt<2 : # 2 because it has two lines of strings and we dont want the string lines
#                obd_data_list.append(['0','0'])
#            else:
#                obd_data_list.append(tmp)
#            cnt +=1
#    
#    if (len(obd_data_list) > 0):      
#        raw_obd_data = np.array(obd_data_list).astype('float')
#        obd_data_frame_sync = raw_obd_data[line_num[0]-1::,:]
#        obd_data_frame_sync[:,0] = obd_data_frame_sync[:,0] - obd_data_frame_sync[0,0]
#        vel_time_interp = interpolate.interp1d(obd_data_frame_sync[:,0],obd_data_frame_sync[:,1])
#        
#    else:
#        vel_time_interp = None 
#else:
#    obd_vel_kmph = 0
    
    
    
    
######## OBD data access ##########################(Sachin's approach)

#if flag_obd_data_available:
#    ptrfilename = 'Obd_Data'
#    obd_pntr = np.load(fpath+'obd_data\\'+ptrfilename+'.npy').astype(int)
#    line_num = obd_pntr[:,1]
#    line_num_change = np.diff(line_num) # to identify where new OBD data has come
#    ind_obdchange = np.where(line_num_change > 0)[0]    
#    obd_line_num = np.zeros((ind_obdchange[-1]+1))
#    cnt = 0
#    for framenum in range(ind_obdchange[-1]+1):
#        obd_line_num[framenum] = line_num[ind_obdchange[cnt]]
#        if framenum > ind_obdchange[cnt]:
#            cnt+=1
#    # account for extra frames in end
#    delta_frames =  obd_pntr.shape[0] - obd_line_num.shape[0]
#    obd_line_num = np.hstack((obd_line_num, obd_line_num[-1]*np.ones(delta_frames))).astype(np.int)
#    csvfilename =  'CSVLog_20181204_122649'   
#    obd_data_list = []
#    cnt = 0    
#    with open(fpath+'obd_data\\'+csvfilename+'.csv') as file:
#        for line in file:
#            tmp = line.strip('\n')
#            tmp = tmp.split(',',2)
#            if cnt<2 : # 2 because it has two lines of strings and we dont want the string lines
#                obd_data_list.append(['0','0'])
#            else:
#                obd_data_list.append(tmp)
#            cnt +=1
#    
#    if (len(obd_data_list) > 0):      
#        obd_data = np.array(obd_data_list).astype('float')
#    else:
#        obd_data = None 
#else:
#    obd_vel_kmph = 0    
    