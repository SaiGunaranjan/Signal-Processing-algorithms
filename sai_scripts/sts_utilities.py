# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 11:18:32 2018

@author: Sai Gunaranjan P
"""

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