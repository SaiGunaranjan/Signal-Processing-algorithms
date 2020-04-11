# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 15:30:40 2019

@author: Sai Gunaranjan Pelluri
"""
####################################CFAR-OS###################################################
# FFT_SignalVector:Give Range FFT or Doppler FFT as your signal vector
# GuardBandLength : To discard some of the adjacent samples surrounding the Cell under test
# CFAR_Window_Length: Sliding window length 
# Threshold_Beta : Linear multiplicative factor
# OrderedStatisticIndex:Pick the Kth index peak out of the window values

import numpy as np



 
def CFAR_OS(signal, num_gaurd, num_train, rate_fa, ord_stat_ind):
    '''
    Inputs [1] signal : input signal vector
           [2] num_train: number of cells from where noise power is estimated (include only one side around cell under test)
           [3] num_gaurd: number of gaurd bands around cell under test (include only one side around cell under test)
           [4] rate_fa: Target false alarm probability
           [5] ord_stat_ind=k : Kth largest noise sample is chosen to calculate noise power
           
    Outputs [1] Target_BoolVector (is a bool vector with same shape as input signal vector and with 1s at the locations of peaks)
    '''
    
    signal_shape = len(signal)
    signal_ext = np.hstack((np.flipud(signal[1::]), signal, np.flipud(signal[0:-1])))  # Typically done to handle boundary condition for 1st sample of the Range FFT sample
    #num_gaurd = 5;
    GuardBandVector = np.zeros(num_gaurd)
    CFAR_Half_Window_Length = num_train + num_gaurd
    Vector_Ones = np.ones((num_train))
    CFAR_Window = np.hstack((Vector_Ones,GuardBandVector,np.array([1]),GuardBandVector,Vector_Ones)) 
    Threshold_Beta = 2*num_train*(rate_fa**(-1/(2*num_train)) -1) # multiplication by a factor of 2  to include valid samples both sides of the CUT
    #ord_stat_ind = 3;
    count = 0
    Target_BoolVector = np.zeros((signal_shape)).astype('int')
    for SignalIndex in np.arange(signal_shape-1,2*signal_shape-1):
      CUT = signal_ext[SignalIndex]
      if CUT >= np.amax(np.array([signal_ext[SignalIndex+1], signal_ext[SignalIndex-1]])):
          Temp = CFAR_Window*signal_ext[SignalIndex-CFAR_Half_Window_Length:SignalIndex+CFAR_Half_Window_Length+1]
    #      CUT = Temp[CFAR_Half_Window_Length]
          Temp_CUT_removed = np.hstack((Temp[0:CFAR_Half_Window_Length],Temp[CFAR_Half_Window_Length+1::]))
          Sorted_Temp_CUT_removed = -1*np.sort(-1*Temp_CUT_removed)
          if (CUT > Threshold_Beta*Sorted_Temp_CUT_removed[ord_stat_ind-1]):
            Target_BoolVector[count] = 1
      count = count + 1


    return Target_BoolVector


def CFAR_CA(signal, num_gaurd, num_train, rate_fa):
    '''
    Inputs [1] signal : input signal vector
           [2] num_train: number of cells from where noise power is estimated (include only one side around cell under test)
           [3] num_gaurd: number of gaurd bands around cell under test (include only one side around cell under test)
           [4] rate_fa: Target false alarm probability
           
           
    Outputs [1] Target_BoolVector (is a bool vector with same shape as input signal vector and with 1s at the locations of peaks)
    '''
    
    signal_shape = len(signal)
    signal_ext = np.hstack((np.flipud(signal[1::]), signal, np.flipud(signal[0:-1])))  # Typically done to handle boundary condition for 1st sample of the Range FFT sample
    #num_gaurd = 5;
    GuardBandVector = np.zeros(num_gaurd)
    CFAR_Half_Window_Length = num_train + num_gaurd
    Vector_Ones = np.ones((num_train))
    CFAR_Window = np.hstack((Vector_Ones,GuardBandVector,np.array([1]),GuardBandVector,Vector_Ones)) 
    Threshold_Beta = 2*num_train*(rate_fa**(-1/(2*num_train)) -1) # multiplication by a factor of 2  to include valid samples both sides of the CUT
    count = 0
    Target_BoolVector = np.zeros((signal_shape)).astype('int')
    for SignalIndex in np.arange(signal_shape-1,2*signal_shape-1):
      CUT = signal_ext[SignalIndex]
      if CUT >= np.amax(np.array([signal_ext[SignalIndex+1], signal_ext[SignalIndex-1]])):
          Temp = CFAR_Window*signal_ext[SignalIndex-CFAR_Half_Window_Length:SignalIndex+CFAR_Half_Window_Length+1]
          Temp_CUT_removed = np.hstack((Temp[0:CFAR_Half_Window_Length],Temp[CFAR_Half_Window_Length+1::]))
          noise_power = np.sum(Temp_CUT_removed)/(2*num_train)
          if (CUT > Threshold_Beta*noise_power):
            Target_BoolVector[count] = 1
      count = count + 1


    return Target_BoolVector






def CFAR_OS_2D(signal, guardband_len_x, guardband_len_y, valid_samp_len_x, valid_samp_len_y, false_alarm_rate, ord_stat_ind):
  
  signal_shape = signal.shape
  signal_shape_rows = signal_shape[0]
  signal_shape_cols = signal_shape[1]
  extend_signal_x = np.hstack((np.fliplr(signal[:,1::]), signal, np.fliplr(signal[:,0:-1])))  # Typically done to habdle boundary condition for 1st sample of the Range FFT sample
  extend_signal = np.vstack((np.flipud(extend_signal_x[1::,:]),extend_signal_x,np.flipud(extend_signal_x[0:-1,:])))
  #GuardBandLength = 5;
  #CFAR_Window_Length = 51; % Should be an odd number
  cfar_window_2_D = np.ones((2*(valid_samp_len_y+guardband_len_y)+1,2*(valid_samp_len_x+guardband_len_x)+1))
  window_patch_zeros = np.zeros((2*guardband_len_y+1,2*guardband_len_x+1))
  window_patch_zeros[guardband_len_y-1,guardband_len_x-1] = 1
  cfar_window_2_D[valid_samp_len_y-1:valid_samp_len_y+2*guardband_len_y,valid_samp_len_x-1:valid_samp_len_x+2*guardband_len_x] = window_patch_zeros
  
  
  cfar_half_window_len_x =  valid_samp_len_x + guardband_len_x
  cfar_half_window_len_y = valid_samp_len_y + guardband_len_y
  
  valid_samp_num = np.count_nonzero(cfar_window_2_D) -1 # -1 to exclude the CUT 
  threshold_alpha = valid_samp_num*(false_alarm_rate**(-1/valid_samp_num) -1) # multiplication by a factor of 2  to include valid samples both sides of the CUT
  #ord_stat_ind = 3;
  Target_BoolVector = np.zeros(signal_shape).astype('int')
  count_y = 0
  for signal_index_y in np.arange(signal_shape_rows-1,2*signal_shape_rows-1):
    count_x = 0
    for signal_index_x in np.arange(signal_shape_cols-1,2*signal_shape_cols-1):
      cut = extend_signal[signal_index_y,signal_index_x]
      # if 1:
      if cut >= np.amax(np.array([extend_signal[signal_index_y+1,signal_index_x],   \
                                  extend_signal[signal_index_y-1,signal_index_x],   \
                                  extend_signal[signal_index_y,signal_index_x+1],   \
                                  extend_signal[signal_index_y,signal_index_x-1],   \
                                  extend_signal[signal_index_y+1,signal_index_x+1], \
                                  extend_signal[signal_index_y+1,signal_index_x-1], \
                                  extend_signal[signal_index_y-1,signal_index_x+1], \
                                  extend_signal[signal_index_y-1,signal_index_x-1]])):
          temp = cfar_window_2_D*extend_signal[signal_index_y-cfar_half_window_len_y:signal_index_y+cfar_half_window_len_y+1, \
                                               signal_index_x-cfar_half_window_len_x:signal_index_x+cfar_half_window_len_x+1]
          sorted_Temp = -1*np.sort(-1*temp.flatten())
          if (cut > threshold_alpha*sorted_Temp[ord_stat_ind-1]):
            Target_BoolVector[count_y,count_x] = 1
      count_x = count_x + 1
    count_y = count_y + 1

  
  return Target_BoolVector



def CFAR_OS_2D_cross(signal, guardband_len_x, guardband_len_y, valid_samp_len_x, valid_samp_len_y, false_alarm_rate, ord_stat_ind):
  
  signal_shape = signal.shape
  signal_shape_rows = signal_shape[0]
  signal_shape_cols = signal_shape[1]
  extend_signal_x = np.hstack((np.fliplr(signal[:,1::]), signal, np.fliplr(signal[:,0:-1])))  # Typically done to habdle boundary condition for 1st sample of the Range FFT sample
  extend_signal = np.vstack((np.flipud(extend_signal_x[1::,:]),extend_signal_x,np.flipud(extend_signal_x[0:-1,:])))

  cfar_window_2_D = np.zeros((2*(valid_samp_len_y+guardband_len_y)+1,2*(valid_samp_len_x+guardband_len_x)+1))
  GuardBandVector_x = np.zeros(guardband_len_x)
  Vector_Ones_x = np.ones((valid_samp_len_x))
  CFAR_Window_x = np.hstack((Vector_Ones_x,GuardBandVector_x,np.array([1]),GuardBandVector_x,Vector_Ones_x))
  GuardBandVector_y = np.zeros(guardband_len_y)
  Vector_Ones_y = np.ones((valid_samp_len_y))
  CFAR_Window_y = np.hstack((Vector_Ones_y,GuardBandVector_y,np.array([1]),GuardBandVector_y,Vector_Ones_y))
  cfar_window_2_D[valid_samp_len_y+guardband_len_y,:] = CFAR_Window_x
  cfar_window_2_D[:,valid_samp_len_x+guardband_len_x] = CFAR_Window_y
  
  cfar_half_window_len_x =  valid_samp_len_x + guardband_len_x
  cfar_half_window_len_y = valid_samp_len_y + guardband_len_y
  
  valid_samp_num = np.count_nonzero(cfar_window_2_D) -1 # -1 to exclude the CUT 
  threshold_alpha = valid_samp_num*(false_alarm_rate**(-1/valid_samp_num) -1) # multiplication by a factor of 2  to include valid samples both sides of the CUT
  #ord_stat_ind = 3;
  Target_BoolVector = np.zeros(signal_shape).astype('int')
  count_y = 0
  for signal_index_y in np.arange(signal_shape_rows-1,2*signal_shape_rows-1):
    count_x = 0
    for signal_index_x in np.arange(signal_shape_cols-1,2*signal_shape_cols-1):
      cut = extend_signal[signal_index_y,signal_index_x]
      # if 1:
      if cut >= np.amax(np.array([extend_signal[signal_index_y+1,signal_index_x],   \
                                  extend_signal[signal_index_y-1,signal_index_x],   \
                                  extend_signal[signal_index_y,signal_index_x+1],   \
                                  extend_signal[signal_index_y,signal_index_x-1],   \
                                  extend_signal[signal_index_y+1,signal_index_x+1], \
                                  extend_signal[signal_index_y+1,signal_index_x-1], \
                                  extend_signal[signal_index_y-1,signal_index_x+1], \
                                  extend_signal[signal_index_y-1,signal_index_x-1]])):
          temp = cfar_window_2_D*extend_signal[signal_index_y-cfar_half_window_len_y:signal_index_y+cfar_half_window_len_y+1, \
                                               signal_index_x-cfar_half_window_len_x:signal_index_x+cfar_half_window_len_x+1]
          sorted_Temp = -1*np.sort(-1*temp.flatten())
          if (cut > threshold_alpha*sorted_Temp[ord_stat_ind-1]):
            Target_BoolVector[count_y,count_x] = 1
      count_x = count_x + 1
    count_y = count_y + 1

  
  return Target_BoolVector

  
  return Target_BoolVector



def CFAR_CA_2D(signal, guardband_len_x, guardband_len_y, valid_samp_len_x, valid_samp_len_y, false_alarm_rate):
  
  signal_shape = signal.shape
  signal_shape_rows = signal_shape[0]
  signal_shape_cols = signal_shape[1]
  extend_signal_x = np.hstack((np.fliplr(signal[:,1::]), signal, np.fliplr(signal[:,0:-1])))  # Typically done to habdle boundary condition for 1st sample of the Range FFT sample
  extend_signal = np.vstack((np.flipud(extend_signal_x[1::,:]),extend_signal_x,np.flipud(extend_signal_x[0:-1,:])))
  #GuardBandLength = 5;
  #CFAR_Window_Length = 51; % Should be an odd number
  cfar_window_2_D = np.ones((2*(valid_samp_len_y+guardband_len_y)+1,2*(valid_samp_len_x+guardband_len_x)+1))
  window_patch_zeros = np.zeros((2*guardband_len_y+1,2*guardband_len_x+1))
  window_patch_zeros[guardband_len_y-1,guardband_len_x-1] = 1
  cfar_window_2_D[valid_samp_len_y-1:valid_samp_len_y+2*guardband_len_y,valid_samp_len_x-1:valid_samp_len_x+2*guardband_len_x] = window_patch_zeros
  
  cfar_half_window_len_x =  valid_samp_len_x + guardband_len_x
  cfar_half_window_len_y = valid_samp_len_y + guardband_len_y
  
  valid_samp_num = np.count_nonzero(cfar_window_2_D) -1 # -1 to exclude the CUT 
  threshold_alpha = valid_samp_num*(false_alarm_rate**(-1/valid_samp_num) -1) 
  #ord_stat_ind = 3;
  Target_BoolVector = np.zeros(signal_shape).astype('int')
  count_y = 0
  for signal_index_y in np.arange(signal_shape_rows-1,2*signal_shape_rows-1):
    count_x = 0
    for signal_index_x in np.arange(signal_shape_cols-1,2*signal_shape_cols-1):
      cut = extend_signal[signal_index_y,signal_index_x]
      # if 1:
      if cut >= np.amax(np.array([extend_signal[signal_index_y+1,signal_index_x], \
                                  extend_signal[signal_index_y-1,signal_index_x], \
                                      extend_signal[signal_index_y,signal_index_x+1], \
                                          extend_signal[signal_index_y,signal_index_x-1], \
                                              extend_signal[signal_index_y+1,signal_index_x+1], \
                                                  extend_signal[signal_index_y+1,signal_index_x-1], \
                                                      extend_signal[signal_index_y-1,signal_index_x+1], \
                                                          extend_signal[signal_index_y-1,signal_index_x-1]])):
          temp = cfar_window_2_D*extend_signal[signal_index_y-cfar_half_window_len_y:signal_index_y+cfar_half_window_len_y+1, \
                                               signal_index_x-cfar_half_window_len_x:signal_index_x+cfar_half_window_len_x+1]
          noise_power = np.sum(temp)/(valid_samp_len_x+valid_samp_len_y)
          if (cut > threshold_alpha*noise_power):
            Target_BoolVector[count_y,count_x] = 1
      count_x = count_x + 1
    count_y = count_y + 1

  
  return Target_BoolVector



def CFAR_CA_2D_cross(signal, guardband_len_x, guardband_len_y, valid_samp_len_x, valid_samp_len_y, false_alarm_rate):
  
  signal_shape = signal.shape
  signal_shape_rows = signal_shape[0]
  signal_shape_cols = signal_shape[1]
  extend_signal_x = np.hstack((np.fliplr(signal[:,1::]), signal, np.fliplr(signal[:,0:-1])))  # Typically done to habdle boundary condition for 1st sample of the Range FFT sample
  extend_signal = np.vstack((np.flipud(extend_signal_x[1::,:]),extend_signal_x,np.flipud(extend_signal_x[0:-1,:])))

  cfar_window_2_D = np.zeros((2*(valid_samp_len_y+guardband_len_y)+1,2*(valid_samp_len_x+guardband_len_x)+1))
  GuardBandVector_x = np.zeros(guardband_len_x)
  Vector_Ones_x = np.ones((valid_samp_len_x))
  CFAR_Window_x = np.hstack((Vector_Ones_x,GuardBandVector_x,np.array([1]),GuardBandVector_x,Vector_Ones_x))
  GuardBandVector_y = np.zeros(guardband_len_y)
  Vector_Ones_y = np.ones((valid_samp_len_y))
  CFAR_Window_y = np.hstack((Vector_Ones_y,GuardBandVector_y,np.array([1]),GuardBandVector_y,Vector_Ones_y))
  cfar_window_2_D[valid_samp_len_y+guardband_len_y,:] = CFAR_Window_x
  cfar_window_2_D[:,valid_samp_len_x+guardband_len_x] = CFAR_Window_y
  
  cfar_half_window_len_x =  valid_samp_len_x + guardband_len_x
  cfar_half_window_len_y = valid_samp_len_y + guardband_len_y
  
  valid_samp_num = np.count_nonzero(cfar_window_2_D) -1 # -1 to exclude the CUT 
  threshold_alpha = valid_samp_num*(false_alarm_rate**(-1/valid_samp_num) -1) 
  #ord_stat_ind = 3;
  Target_BoolVector = np.zeros(signal_shape).astype('int')
  count_y = 0
  for signal_index_y in np.arange(signal_shape_rows-1,2*signal_shape_rows-1):
    count_x = 0
    for signal_index_x in np.arange(signal_shape_cols-1,2*signal_shape_cols-1):
      cut = extend_signal[signal_index_y,signal_index_x]
      # if 1:
      if cut >= np.amax(np.array([extend_signal[signal_index_y+1,signal_index_x], \
                                  extend_signal[signal_index_y-1,signal_index_x], \
                                      extend_signal[signal_index_y,signal_index_x+1], \
                                          extend_signal[signal_index_y,signal_index_x-1], \
                                              extend_signal[signal_index_y+1,signal_index_x+1], \
                                                  extend_signal[signal_index_y+1,signal_index_x-1], \
                                                      extend_signal[signal_index_y-1,signal_index_x+1], \
                                                          extend_signal[signal_index_y-1,signal_index_x-1]])):
          temp = cfar_window_2_D*extend_signal[signal_index_y-cfar_half_window_len_y:signal_index_y+cfar_half_window_len_y+1, \
                                               signal_index_x-cfar_half_window_len_x:signal_index_x+cfar_half_window_len_x+1]
          noise_power = np.sum(temp)/(2*(valid_samp_len_x+valid_samp_len_y))
          if (cut > threshold_alpha*noise_power):
            Target_BoolVector[count_y,count_x] = 1
      count_x = count_x + 1
    count_y = count_y + 1

  
  return Target_BoolVector

  
  return Target_BoolVector



def CFAR_CA_2D_cross_algo_stack(signal, guardband_len_x, guardband_len_y, valid_samp_len_x, valid_samp_len_y, false_alarm_rate, local_max_along_x, local_max_along_y):
  
  signal_shape = signal.shape
  signal_shape_rows = signal_shape[0]
  signal_shape_cols = signal_shape[1]
  extend_signal_x = np.hstack((np.fliplr(signal[:,1::]), signal, np.fliplr(signal[:,0:-1])))  # Typically done to habdle boundary condition for 1st sample of the Range FFT sample
  extend_signal = np.vstack((np.flipud(extend_signal_x[1::,:]),extend_signal_x,np.flipud(extend_signal_x[0:-1,:])))

  cfar_window_2_D = np.zeros((2*(valid_samp_len_y+guardband_len_y)+1,2*(valid_samp_len_x+guardband_len_x)+1))
  GuardBandVector_x = np.zeros(guardband_len_x)
  Vector_Ones_x = np.ones((valid_samp_len_x))
  CFAR_Window_x = np.hstack((Vector_Ones_x,GuardBandVector_x,np.array([1]),GuardBandVector_x,Vector_Ones_x))
  GuardBandVector_y = np.zeros(guardband_len_y)
  Vector_Ones_y = np.ones((valid_samp_len_y))
  CFAR_Window_y = np.hstack((Vector_Ones_y,GuardBandVector_y,np.array([1]),GuardBandVector_y,Vector_Ones_y))
  cfar_window_2_D[valid_samp_len_y+guardband_len_y,:] = CFAR_Window_x
  cfar_window_2_D[:,valid_samp_len_x+guardband_len_x] = CFAR_Window_y
  

  
  
  cfar_half_window_len_x =  valid_samp_len_x + guardband_len_x
  cfar_half_window_len_y = valid_samp_len_y + guardband_len_y
  
  valid_samp_num = np.count_nonzero(cfar_window_2_D) -1 # -1 to exclude the CUT 
  threshold_alpha = valid_samp_num*(false_alarm_rate**(-1/valid_samp_num) -1) 
  #ord_stat_ind = 3;
  Target_BoolVector = np.zeros(signal_shape).astype('int')
  count_y = 0
  for signal_index_y in np.arange(signal_shape_rows-1,2*signal_shape_rows-1):
    count_x = 0
    for signal_index_x in np.arange(signal_shape_cols-1,2*signal_shape_cols-1):
      cut = extend_signal[signal_index_y,signal_index_x]
      if local_max_along_x == True and local_max_along_y == True:
          max_val =  np.amax(np.array([extend_signal[signal_index_y+1,signal_index_x], \
          extend_signal[signal_index_y-1,signal_index_x], \
          extend_signal[signal_index_y,signal_index_x+1], \
          extend_signal[signal_index_y,signal_index_x-1], \
          extend_signal[signal_index_y+1,signal_index_x+1], \
          extend_signal[signal_index_y+1,signal_index_x-1], \
          extend_signal[signal_index_y-1,signal_index_x+1], \
          extend_signal[signal_index_y-1,signal_index_x-1]]))
          if cut >= max_val:
              temp = cfar_window_2_D*extend_signal[signal_index_y-cfar_half_window_len_y:signal_index_y+cfar_half_window_len_y+1,signal_index_x-cfar_half_window_len_x:signal_index_x+cfar_half_window_len_x+1]
              noise_power = np.sum(temp)/(2*(valid_samp_len_x+valid_samp_len_y))
              if (cut > threshold_alpha*noise_power):
                Target_BoolVector[count_y,count_x] = 1
              
      elif local_max_along_x == False and local_max_along_y == True:
          max_val =  np.amax(np.array([extend_signal[signal_index_y+1,signal_index_x], extend_signal[signal_index_y-1,signal_index_x]]))
          if cut >= max_val:
              temp = cfar_window_2_D*extend_signal[signal_index_y-cfar_half_window_len_y:signal_index_y+cfar_half_window_len_y+1,signal_index_x-cfar_half_window_len_x:signal_index_x+cfar_half_window_len_x+1]
              noise_power = np.sum(temp)/(2*(valid_samp_len_x+valid_samp_len_y))
              if (cut > threshold_alpha*noise_power):
                Target_BoolVector[count_y,count_x] = 1
          
      elif local_max_along_x == True and local_max_along_y == False:
          max_val = np.amax(np.array([extend_signal[signal_index_y,signal_index_x+1], extend_signal[signal_index_y,signal_index_x-1]]))
          if cut >= max_val:
              temp = cfar_window_2_D*extend_signal[signal_index_y-cfar_half_window_len_y:signal_index_y+cfar_half_window_len_y+1,signal_index_x-cfar_half_window_len_x:signal_index_x+cfar_half_window_len_x+1]
              noise_power = np.sum(temp)/(2*(valid_samp_len_x+valid_samp_len_y))
              if (cut > threshold_alpha*noise_power):
                Target_BoolVector[count_y,count_x] = 1          
      else:
          temp = cfar_window_2_D*extend_signal[signal_index_y-cfar_half_window_len_y:signal_index_y+cfar_half_window_len_y+1,signal_index_x-cfar_half_window_len_x:signal_index_x+cfar_half_window_len_x+1]
          noise_power = np.sum(temp)/(2*(valid_samp_len_x+valid_samp_len_y))
          if (cut > threshold_alpha*noise_power):
            Target_BoolVector[count_y,count_x] = 1          

      count_x = count_x + 1
    count_y = count_y + 1

  
  return Target_BoolVector


def CFAR_CA_2D_cross_map(signal, guardband_len_x, guardband_len_y, valid_samp_len_x, valid_samp_len_y):
  
  signal_shape = signal.shape
  signal_shape_rows = signal_shape[0]
  signal_shape_cols = signal_shape[1]
  extend_signal_x = np.hstack((np.fliplr(signal[:,1::]), signal, np.fliplr(signal[:,0:-1])))  # Typically done to habdle boundary condition for 1st sample of the Range FFT sample
  extend_signal = np.vstack((np.flipud(extend_signal_x[1::,:]),extend_signal_x,np.flipud(extend_signal_x[0:-1,:])))

  cfar_window_2_D = np.zeros((2*(valid_samp_len_y+guardband_len_y)+1,2*(valid_samp_len_x+guardband_len_x)+1))
  GuardBandVector_x = np.zeros(guardband_len_x)
  Vector_Ones_x = np.ones((valid_samp_len_x))
  CFAR_Window_x = np.hstack((Vector_Ones_x,GuardBandVector_x,np.array([1]),GuardBandVector_x,Vector_Ones_x))
  GuardBandVector_y = np.zeros(guardband_len_y)
  Vector_Ones_y = np.ones((valid_samp_len_y))
  CFAR_Window_y = np.hstack((Vector_Ones_y,GuardBandVector_y,np.array([1]),GuardBandVector_y,Vector_Ones_y))
  cfar_window_2_D[valid_samp_len_y+guardband_len_y,:] = CFAR_Window_x
  cfar_window_2_D[:,valid_samp_len_x+guardband_len_x] = CFAR_Window_y
  
  cfar_half_window_len_x =  valid_samp_len_x + guardband_len_x
  cfar_half_window_len_y = valid_samp_len_y + guardband_len_y
  
#  valid_samp_num = np.count_nonzero(cfar_window_2_D) -1 # -1 to exclude the CUT 
  threshold_map = np.zeros(signal_shape).astype('float32')
  count_y = 0
  for signal_index_y in np.arange(signal_shape_rows-1,2*signal_shape_rows-1):
    count_x = 0
    for signal_index_x in np.arange(signal_shape_cols-1,2*signal_shape_cols-1):
      temp = cfar_window_2_D*extend_signal[signal_index_y-cfar_half_window_len_y:signal_index_y+cfar_half_window_len_y+1, \
                                           signal_index_x-cfar_half_window_len_x:signal_index_x+cfar_half_window_len_x+1]
      avg_noise_power = np.sum(temp)/(2*(valid_samp_len_x+valid_samp_len_y))
      threshold_map[count_y,count_x] = avg_noise_power
      count_x = count_x + 1
    count_y = count_y + 1

  
  return threshold_map





    

