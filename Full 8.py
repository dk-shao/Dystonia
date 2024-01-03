import mne
import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ---------- Time Date Information ---------- #
start = time.time()
last_updated = "02/01/2023"

# ---------- Selecting the data location ---------- #
root = 'C:/Users/dksha/Documents/BME SPM/Hi5 Study Preprocessed EEG Data/'
meta = 'C:/Users/dksha/Documents/BME SPM/Hi5 Study Preprocessed EEG Data/hi5 clean.xlsx'

# ---------- Defining the analysis parameters ---------- #
ch_left = "C4" # setting the channel for the left-hand task (right hemisphere electrodes)
ch_right = "C3" # setting the channel for the right-hand task (left hemisphere electrodes)

n = 24 # selecting the number of epoch to extract per patient

wavelet = 'cmor2.0-12.0' # selecting the wavelet type
scales = np.arange(1,3) # selecting the wavelet scale

kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": None}

# ---------- Meta data ---------- #
df = pd.DataFrame()
df=pd.read_excel(meta)

pre_default_time_start = -4.5
pre_default_time_end = -3.5

post_default_time_start = -3.5
post_default_time_end = 0

data_left_hi = []
data_left_lo = []

data_right_hi = []
data_right_lo = []

# ---------- Importing Group A Dystonia LHP ---------- #
A11_left = mne.io.read_raw_brainvision(root+'Patient_Group_A_LHP/SSACD11A_02_Artifact Rejection_PreProc.vhdr', preload=False)
A30_left = mne.io.read_raw_brainvision(root+'Patient_Group_A_LHP/SSACD30A_02_Artifact Rejection_PreProc.vhdr', preload=False)
A39_left = mne.io.read_raw_brainvision(root+'Patient_Group_A_LHP/SSACD39A_01_Artifact Rejection_PreProc.vhdr', preload=False)
A50_left = mne.io.read_raw_brainvision(root+'Patient_Group_A_LHP/SSACD50A_3_Artifact Rejection_PreProc.vhdr', preload=False)
A65_left = mne.io.read_raw_brainvision(root+'Patient_Group_A_LHP/SSACD65A_03_Artifact Rejection 2_PreProc_2.vhdr', preload=False)
A70_left = mne.io.read_raw_brainvision(root+'Patient_Group_A_LHP/SSACD70A_01_Artifact Rejection_PreProc.vhdr', preload=False)
A77_left = mne.io.read_raw_brainvision(root+'Patient_Group_A_LHP/SSACD77A_06_Artifact Rejection_PreProc.vhdr', preload=False)
A80_left = mne.io.read_raw_brainvision(root+'Patient_Group_A_LHP/SSACD80A_01_Artifact Rejection_PreProc.vhdr', preload=False)
A92_left = mne.io.read_raw_brainvision(root+'Patient_Group_A_LHP/SSACD92A_5_Artifact Rejection_PreProc.vhdr', preload=False)
A101_left = mne.io.read_raw_brainvision(root+'Patient_Group_A_LHP/SSACD101A_03_Artifact Rejection_Comb_PreProc.vhdr', preload=False)
A104_left = mne.io.read_raw_brainvision(root+'Patient_Group_A_LHP/SSACD104A_03_Artifact Rejection_PreProc.vhdr', preload=False)

# ---------- Importing Group A Dystonia RHP ---------- #
A11_right = mne.io.read_raw_brainvision(root+'Patient_Group_A_RHP/SSACD11A_01_Artifact Rejection_PreProc.vhdr', preload=False)
A30_right = mne.io.read_raw_brainvision(root+'Patient_Group_A_RHP/SSACD30A_01_Artifact Rejection_PreProc.vhdr', preload=False)
A39_right = mne.io.read_raw_brainvision(root+'Patient_Group_A_RHP/SSACD39A_02_Artifact Rejection_PreProc.vhdr', preload=False)
A50_right = mne.io.read_raw_brainvision(root+'Patient_Group_A_RHP/SSACD50A_2_Artifact Rejection_PreProc.vhdr', preload=False)
A65_right = mne.io.read_raw_brainvision(root+'Patient_Group_A_RHP/SSACD65A_01_Artifact Rejection_PreProc.vhdr', preload=False)
A70_right = mne.io.read_raw_brainvision(root+'Patient_Group_A_RHP/SSACD70A_03_Artifact Rejection_PreProc.vhdr', preload=False)
A77_right = mne.io.read_raw_brainvision(root+'Patient_Group_A_RHP/SSACD77A_02_Artifact Rejection_PreProc.vhdr', preload=False)
A80_right = mne.io.read_raw_brainvision(root+'Patient_Group_A_RHP/SSACD80A_02_Artifact Rejection_PreProc.vhdr', preload=False)
A92_right = mne.io.read_raw_brainvision(root+'Patient_Group_A_RHP/SSACD92A_2_Artifact Rejection_PreProc.vhdr', preload=False)
A101_right = mne.io.read_raw_brainvision(root+'Patient_Group_A_RHP/SSACD101A_01_Artifact Rejection_Comb_PreProc.vhdr', preload=False)
A104_right = mne.io.read_raw_brainvision(root+'Patient_Group_A_RHP/SSACD104A_01_Artifact Rejection_PreProc.vhdr', preload=False)

# ---------- Importing Group P Dystonia LHP ---------- #
P05_left = mne.io.read_raw_brainvision(root+'Patient_Group_P_LHP/SSACD05P_02_Artifact Rejection_PreProc.vhdr', preload=False)
P16_left = mne.io.read_raw_brainvision(root+'Patient_Group_P_LHP/SSACD16P_3_Artifact Rejection_PreProc.vhdr', preload=False)
P21_left = mne.io.read_raw_brainvision(root+'Patient_Group_P_LHP/SSACD21P_2_Artifact Rejection_PreProc.vhdr', preload=False)
P23_left = mne.io.read_raw_brainvision(root+'Patient_Group_P_LHP/SSACD23P_01_Artifact Rejection_PreProc.vhdr', preload=False)
P28_left = mne.io.read_raw_brainvision(root+'Patient_Group_P_LHP/SSACD28P_03_Artifact Rejection_PreProc.vhdr', preload=False)
P43_left = mne.io.read_raw_brainvision(root+'Patient_Group_P_LHP/SSACD43P_10_Artifact Rejection_PreProc.vhdr', preload=False)
P45_left = mne.io.read_raw_brainvision(root+'Patient_Group_P_LHP/SSACD45P_01_Artifact Rejection_PreProc.vhdr', preload=False)
P53_left = mne.io.read_raw_brainvision(root+'Patient_Group_P_LHP/SSACD53P_02_Artifact Rejection_PreProc.vhdr', preload=False)
P81_left = mne.io.read_raw_brainvision(root+'Patient_Group_P_LHP/SSACD81_02_Artifact Rejection_PreProc.vhdr', preload=False)
P83_left = mne.io.read_raw_brainvision(root+'Patient_Group_P_LHP/SSACD83P_3_Artifact Rejection_Baseline Correction_Comb_PreProc.vhdr', preload=False)
P93_left = mne.io.read_raw_brainvision(root+'Patient_Group_P_LHP/SSACD93P_1_Artifact Rejection_PreProc.vhdr', preload=False)
P97_left = mne.io.read_raw_brainvision(root+'Patient_Group_P_LHP/SSACD97P_02_Artifact Rejection_PreProc.vhdr', preload=False)
P109_left = mne.io.read_raw_brainvision(root+'Patient_Group_P_LHP/SSACD109P_02_Artifact Rejection_PreProc.vhdr', preload=False)
P131_left = mne.io.read_raw_brainvision(root+'Patient_Group_P_LHP/SSACD131P_01_Artifact Rejection_Comb_PreProc.vhdr', preload=False)
P162_left = mne.io.read_raw_brainvision(root+'Patient_Group_P_LHP/SSACD162P_04_Artifact Rejection_PreProc.vhdr', preload=False)

# ---------- Importing Group P Dystonia RHP ---------- #
P05_right = mne.io.read_raw_brainvision(root+'Patient_Group_P_RHP/SSACD05P_01_Artifact Rejection_PreProc.vhdr', preload=False)
P16_right = mne.io.read_raw_brainvision(root+'Patient_Group_P_RHP/SSACD16P_1_Artifact Rejection_PreProc.vhdr', preload=False)
P21_right = mne.io.read_raw_brainvision(root+'Patient_Group_P_RHP/SSACD21P_4_Artifact Rejection_PreProc.vhdr', preload=False)
P23_right = mne.io.read_raw_brainvision(root+'Patient_Group_P_RHP/SSACD23P_02_Artifact Rejection_PreProc.vhdr', preload=False)
P28_right = mne.io.read_raw_brainvision(root+'Patient_Group_P_RHP/SSACD28P_01_Artifact Rejection_PreProc.vhdr', preload=False)
P43_right = mne.io.read_raw_brainvision(root+'Patient_Group_P_RHP/SSACD43P_01_Artifact Rejection_PreProc.vhdr', preload=False)
P45_right = mne.io.read_raw_brainvision(root+'Patient_Group_P_RHP/SSACD45P_04_Artifact Rejection_PreProc.vhdr', preload=False)
P53_right = mne.io.read_raw_brainvision(root+'Patient_Group_P_RHP/SSACD53P_03_Artifact Rejection_PreProc.vhdr', preload=False)
P81_right = mne.io.read_raw_brainvision(root+'Patient_Group_P_RHP/SSACD81P_05_Artifact Rejection_PreProc.vhdr', preload=False)
P83_right = mne.io.read_raw_brainvision(root+'Patient_Group_P_RHP/SSACD83P_6_Artifact Rejection_Baseline Correction_Comb_PreProc.vhdr', preload=False)
P93_right = mne.io.read_raw_brainvision(root+'Patient_Group_P_RHP/SSACD93P_3_Artifact Rejection_PreProc.vhdr', preload=False)
P97_right = mne.io.read_raw_brainvision(root+'Patient_Group_P_RHP/SSACD97P_01_Artifact Rejection_PreProc.vhdr', preload=False)
P109_right = mne.io.read_raw_brainvision(root+'Patient_Group_P_RHP/SSACD109P_01_Artifact Rejection_PreProc.vhdr', preload=False)
P131_right = mne.io.read_raw_brainvision(root+'Patient_Group_P_RHP/SSACD97P_01_Artifact Rejection_PreProc.vhdr', preload=False)
P162_right = mne.io.read_raw_brainvision(root+'Patient_Group_P_RHP/SSACD162P_01_Artifact Rejection_PreProc.vhdr', preload=False)

# ---------- Selecting the channels --------- #
A11_left = A11_left.pick([ch_left])
A30_left = A30_left.pick([ch_left])
A39_left = A39_left.pick([ch_left])
A50_left = A50_left.pick([ch_left])
A65_left = A65_left.pick([ch_left])
A70_left = A70_left.pick([ch_left])
A77_left = A77_left.pick([ch_left])
A80_left = A80_left.pick([ch_left])
A92_left = A92_left.pick([ch_left])
A101_left= A101_left.pick([ch_left])
A104_left = A104_left.pick([ch_left])

A11_right = A11_right.pick([ch_right])
A30_right = A30_right.pick([ch_right])
A39_right = A39_right.pick([ch_right])
A50_right = A50_right.pick([ch_right])
A65_right = A65_right.pick([ch_right])
A70_right = A70_right.pick([ch_right])
A77_right = A77_right.pick([ch_right])
A80_right = A80_right.pick([ch_right])
A92_right = A92_right.pick([ch_right])
A101_right= A101_right.pick([ch_right])
A104_right = A104_right.pick([ch_right])

P05_left = P05_left.pick([ch_left])
P16_left = P16_left.pick([ch_left])
P21_left = P21_left.pick([ch_left])
P23_left = P23_left.pick([ch_left])
P28_left = P28_left.pick([ch_left])
P43_left = P43_left.pick([ch_left])
P45_left = P45_left.pick([ch_left])
P53_left = P53_left.pick([ch_left])
P81_left = P81_left.pick([ch_left])
P83_left= P83_left.pick([ch_left])
P93_left = P93_left.pick([ch_left])
P97_left = P97_left.pick([ch_left])
P109_left = P109_left.pick([ch_left])
P131_left= P131_left.pick([ch_left])
P162_left = P162_left.pick([ch_left])

P05_right = P05_right.pick([ch_right])
P16_right = P16_right.pick([ch_right])
P21_right = P21_right.pick([ch_right])
P23_right = P23_right.pick([ch_right])
P28_right = P28_right.pick([ch_right])
P43_right = P43_right.pick([ch_right])
P45_right = P45_right.pick([ch_right])
P53_right = P53_right.pick([ch_right])
P81_right = P81_right.pick([ch_right])
P83_right= P83_right.pick([ch_right])
P93_right = P93_right.pick([ch_right])
P97_right = P97_right.pick([ch_right])
P109_right = P109_right.pick([ch_right])
P131_right= P131_right.pick([ch_right])
P162_right = P162_right.pick([ch_right])

# ---------- Analysis for A39_left ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["A39_left_epoch_{0}".format(1+x)] = A39_left.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["A39_left_epoch_{0}".format(1+x)] = A39_left.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["A39_left_epoch_{0}".format(1+x)] = d1_pre.get("A39_left_epoch_{0}".format(1+x))[0,:]
    d1_post["A39_left_epoch_{0}".format(1+x)] = d1_post.get("A39_left_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["A39_left_data_{0}".format(1+y)] = d1_pre.get("A39_left_epoch_{0}".format(1+y))[0]
    d2_post["A39_left_data_{0}".format(1+y)] = d1_post.get("A39_left_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_left_hi.append(np.mean(delta_hi))
data_left_lo.append(np.mean(delta_lo))

del A39_left, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for A92_left ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["A92_left_epoch_{0}".format(1+x)] = A92_left.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["A92_left_epoch_{0}".format(1+x)] = A92_left.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["A92_left_epoch_{0}".format(1+x)] = d1_pre.get("A92_left_epoch_{0}".format(1+x))[0,:]
    d1_post["A92_left_epoch_{0}".format(1+x)] = d1_post.get("A92_left_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["A92_left_data_{0}".format(1+y)] = d1_pre.get("A92_left_epoch_{0}".format(1+y))[0]
    d2_post["A92_left_data_{0}".format(1+y)] = d1_post.get("A92_left_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_left_hi.append(np.mean(delta_hi))
data_left_lo.append(np.mean(delta_lo))

del A92_left, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo
 
# ---------- Analysis for A30_left ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["A30_left_epoch_{0}".format(1+x)] = A30_left.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["A30_left_epoch_{0}".format(1+x)] = A30_left.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["A30_left_epoch_{0}".format(1+x)] = d1_pre.get("A30_left_epoch_{0}".format(1+x))[0,:]
    d1_post["A30_left_epoch_{0}".format(1+x)] = d1_post.get("A30_left_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["A30_left_data_{0}".format(1+y)] = d1_pre.get("A30_left_epoch_{0}".format(1+y))[0]
    d2_post["A30_left_data_{0}".format(1+y)] = d1_post.get("A30_left_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_left_hi.append(np.mean(delta_hi))
data_left_lo.append(np.mean(delta_lo))

del A30_left, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo
 
# ---------- Analysis for A101_left ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["A101_left_epoch_{0}".format(1+x)] = A101_left.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["A101_left_epoch_{0}".format(1+x)] = A101_left.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["A101_left_epoch_{0}".format(1+x)] = d1_pre.get("A101_left_epoch_{0}".format(1+x))[0,:]
    d1_post["A101_left_epoch_{0}".format(1+x)] = d1_post.get("A101_left_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["A101_left_data_{0}".format(1+y)] = d1_pre.get("A101_left_epoch_{0}".format(1+y))[0]
    d2_post["A101_left_data_{0}".format(1+y)] = d1_post.get("A101_left_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_left_hi.append(np.mean(delta_hi))
data_left_lo.append(np.mean(delta_lo))

del A101_left, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P16_left ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P16_left_epoch_{0}".format(1+x)] = P16_left.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P16_left_epoch_{0}".format(1+x)] = P16_left.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P16_left_epoch_{0}".format(1+x)] = d1_pre.get("P16_left_epoch_{0}".format(1+x))[0,:]
    d1_post["P16_left_epoch_{0}".format(1+x)] = d1_post.get("P16_left_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P16_left_data_{0}".format(1+y)] = d1_pre.get("P16_left_epoch_{0}".format(1+y))[0]
    d2_post["P16_left_data_{0}".format(1+y)] = d1_post.get("P16_left_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_left_hi.append(np.mean(delta_hi))
data_left_lo.append(np.mean(delta_lo))

del P16_left, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P53_left ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P53_left_epoch_{0}".format(1+x)] = P53_left.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P53_left_epoch_{0}".format(1+x)] = P53_left.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P53_left_epoch_{0}".format(1+x)] = d1_pre.get("P53_left_epoch_{0}".format(1+x))[0,:]
    d1_post["P53_left_epoch_{0}".format(1+x)] = d1_post.get("P53_left_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P53_left_data_{0}".format(1+y)] = d1_pre.get("P53_left_epoch_{0}".format(1+y))[0]
    d2_post["P53_left_data_{0}".format(1+y)] = d1_post.get("P53_left_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_left_hi.append(np.mean(delta_hi))
data_left_lo.append(np.mean(delta_lo))

del P53_left, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for A50_left ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["A50_left_epoch_{0}".format(1+x)] = A50_left.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["A50_left_epoch_{0}".format(1+x)] = A50_left.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["A50_left_epoch_{0}".format(1+x)] = d1_pre.get("A50_left_epoch_{0}".format(1+x))[0,:]
    d1_post["A50_left_epoch_{0}".format(1+x)] = d1_post.get("A50_left_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["A50_left_data_{0}".format(1+y)] = d1_pre.get("A50_left_epoch_{0}".format(1+y))[0]
    d2_post["A50_left_data_{0}".format(1+y)] = d1_post.get("A50_left_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_left_hi.append(np.mean(delta_hi))
data_left_lo.append(np.mean(delta_lo))

del A50_left, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P21_left ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P21_left_epoch_{0}".format(1+x)] = P21_left.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P21_left_epoch_{0}".format(1+x)] = P21_left.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P21_left_epoch_{0}".format(1+x)] = d1_pre.get("P21_left_epoch_{0}".format(1+x))[0,:]
    d1_post["P21_left_epoch_{0}".format(1+x)] = d1_post.get("P21_left_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P21_left_data_{0}".format(1+y)] = d1_pre.get("P21_left_epoch_{0}".format(1+y))[0]
    d2_post["P21_left_data_{0}".format(1+y)] = d1_post.get("P21_left_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_left_hi.append(np.mean(delta_hi))
data_left_lo.append(np.mean(delta_lo))

del P21_left, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P45_left ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P45_left_epoch_{0}".format(1+x)] = P45_left.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P45_left_epoch_{0}".format(1+x)] = P45_left.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P45_left_epoch_{0}".format(1+x)] = d1_pre.get("P45_left_epoch_{0}".format(1+x))[0,:]
    d1_post["P45_left_epoch_{0}".format(1+x)] = d1_post.get("P45_left_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P45_left_data_{0}".format(1+y)] = d1_pre.get("P45_left_epoch_{0}".format(1+y))[0]
    d2_post["P45_left_data_{0}".format(1+y)] = d1_post.get("P45_left_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_left_hi.append(np.mean(delta_hi))
data_left_lo.append(np.mean(delta_lo))

del P45_left, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P162_left ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P162_left_epoch_{0}".format(1+x)] = P162_left.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P162_left_epoch_{0}".format(1+x)] = P162_left.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P162_left_epoch_{0}".format(1+x)] = d1_pre.get("P162_left_epoch_{0}".format(1+x))[0,:]
    d1_post["P162_left_epoch_{0}".format(1+x)] = d1_post.get("P162_left_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P162_left_data_{0}".format(1+y)] = d1_pre.get("P162_left_epoch_{0}".format(1+y))[0]
    d2_post["P162_left_data_{0}".format(1+y)] = d1_post.get("P162_left_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_left_hi.append(np.mean(delta_hi))
data_left_lo.append(np.mean(delta_lo))

del P162_left, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P97_left ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P97_left_epoch_{0}".format(1+x)] = P97_left.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P97_left_epoch_{0}".format(1+x)] = P97_left.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P97_left_epoch_{0}".format(1+x)] = d1_pre.get("P97_left_epoch_{0}".format(1+x))[0,:]
    d1_post["P97_left_epoch_{0}".format(1+x)] = d1_post.get("P97_left_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P97_left_data_{0}".format(1+y)] = d1_pre.get("P97_left_epoch_{0}".format(1+y))[0]
    d2_post["P97_left_data_{0}".format(1+y)] = d1_post.get("P97_left_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_left_hi.append(np.mean(delta_hi))
data_left_lo.append(np.mean(delta_lo))

del P97_left, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P05_left ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P05_left_epoch_{0}".format(1+x)] = P05_left.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P05_left_epoch_{0}".format(1+x)] = P05_left.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P05_left_epoch_{0}".format(1+x)] = d1_pre.get("P05_left_epoch_{0}".format(1+x))[0,:]
    d1_post["P05_left_epoch_{0}".format(1+x)] = d1_post.get("P05_left_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P05_left_data_{0}".format(1+y)] = d1_pre.get("P05_left_epoch_{0}".format(1+y))[0]
    d2_post["P05_left_data_{0}".format(1+y)] = d1_post.get("P05_left_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_left_hi.append(np.mean(delta_hi))
data_left_lo.append(np.mean(delta_lo))

del P05_left, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for A104_left ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["A104_left_epoch_{0}".format(1+x)] = A104_left.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["A104_left_epoch_{0}".format(1+x)] = A104_left.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["A104_left_epoch_{0}".format(1+x)] = d1_pre.get("A104_left_epoch_{0}".format(1+x))[0,:]
    d1_post["A104_left_epoch_{0}".format(1+x)] = d1_post.get("A104_left_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["A104_left_data_{0}".format(1+y)] = d1_pre.get("A104_left_epoch_{0}".format(1+y))[0]
    d2_post["A104_left_data_{0}".format(1+y)] = d1_post.get("A104_left_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_left_hi.append(np.mean(delta_hi))
data_left_lo.append(np.mean(delta_lo))

del A104_left, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for A70_left ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["A70_left_epoch_{0}".format(1+x)] = A70_left.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["A70_left_epoch_{0}".format(1+x)] = A70_left.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["A70_left_epoch_{0}".format(1+x)] = d1_pre.get("A70_left_epoch_{0}".format(1+x))[0,:]
    d1_post["A70_left_epoch_{0}".format(1+x)] = d1_post.get("A70_left_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["A70_left_data_{0}".format(1+y)] = d1_pre.get("A70_left_epoch_{0}".format(1+y))[0]
    d2_post["A70_left_data_{0}".format(1+y)] = d1_post.get("A70_left_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_left_hi.append(np.mean(delta_hi))
data_left_lo.append(np.mean(delta_lo))

del A70_left, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P83_left ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P83_left_epoch_{0}".format(1+x)] = P83_left.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P83_left_epoch_{0}".format(1+x)] = P83_left.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P83_left_epoch_{0}".format(1+x)] = d1_pre.get("P83_left_epoch_{0}".format(1+x))[0,:]
    d1_post["P83_left_epoch_{0}".format(1+x)] = d1_post.get("P83_left_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P83_left_data_{0}".format(1+y)] = d1_pre.get("P83_left_epoch_{0}".format(1+y))[0]
    d2_post["P83_left_data_{0}".format(1+y)] = d1_post.get("P83_left_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_left_hi.append(np.mean(delta_hi))
data_left_lo.append(np.mean(delta_lo))

del P83_left, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P28_left ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P28_left_epoch_{0}".format(1+x)] = P28_left.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P28_left_epoch_{0}".format(1+x)] = P28_left.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P28_left_epoch_{0}".format(1+x)] = d1_pre.get("P28_left_epoch_{0}".format(1+x))[0,:]
    d1_post["P28_left_epoch_{0}".format(1+x)] = d1_post.get("P28_left_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P28_left_data_{0}".format(1+y)] = d1_pre.get("P28_left_epoch_{0}".format(1+y))[0]
    d2_post["P28_left_data_{0}".format(1+y)] = d1_post.get("P28_left_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_left_hi.append(np.mean(delta_hi))
data_left_lo.append(np.mean(delta_lo))

del P28_left, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for A11_left ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["A11_left_epoch_{0}".format(1+x)] = A11_left.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["A11_left_epoch_{0}".format(1+x)] = A11_left.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["A11_left_epoch_{0}".format(1+x)] = d1_pre.get("A11_left_epoch_{0}".format(1+x))[0,:]
    d1_post["A11_left_epoch_{0}".format(1+x)] = d1_post.get("A11_left_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["A11_left_data_{0}".format(1+y)] = d1_pre.get("A11_left_epoch_{0}".format(1+y))[0]
    d2_post["A11_left_data_{0}".format(1+y)] = d1_post.get("A11_left_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_left_hi.append(np.mean(delta_hi))
data_left_lo.append(np.mean(delta_lo))

del A11_left, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P109_left ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P109_left_epoch_{0}".format(1+x)] = P109_left.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P109_left_epoch_{0}".format(1+x)] = P109_left.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P109_left_epoch_{0}".format(1+x)] = d1_pre.get("P109_left_epoch_{0}".format(1+x))[0,:]
    d1_post["P109_left_epoch_{0}".format(1+x)] = d1_post.get("P109_left_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P109_left_data_{0}".format(1+y)] = d1_pre.get("P109_left_epoch_{0}".format(1+y))[0]
    d2_post["P109_left_data_{0}".format(1+y)] = d1_post.get("P109_left_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_left_hi.append(np.mean(delta_hi))
data_left_lo.append(np.mean(delta_lo))

del P109_left, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P23_left ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P23_left_epoch_{0}".format(1+x)] = P23_left.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P23_left_epoch_{0}".format(1+x)] = P23_left.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P23_left_epoch_{0}".format(1+x)] = d1_pre.get("P23_left_epoch_{0}".format(1+x))[0,:]
    d1_post["P23_left_epoch_{0}".format(1+x)] = d1_post.get("P23_left_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P23_left_data_{0}".format(1+y)] = d1_pre.get("P23_left_epoch_{0}".format(1+y))[0]
    d2_post["P23_left_data_{0}".format(1+y)] = d1_post.get("P23_left_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_left_hi.append(np.mean(delta_hi))
data_left_lo.append(np.mean(delta_lo))

del P23_left, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P131_left ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P131_left_epoch_{0}".format(1+x)] = P131_left.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P131_left_epoch_{0}".format(1+x)] = P131_left.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P131_left_epoch_{0}".format(1+x)] = d1_pre.get("P131_left_epoch_{0}".format(1+x))[0,:]
    d1_post["P131_left_epoch_{0}".format(1+x)] = d1_post.get("P131_left_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P131_left_data_{0}".format(1+y)] = d1_pre.get("P131_left_epoch_{0}".format(1+y))[0]
    d2_post["P131_left_data_{0}".format(1+y)] = d1_post.get("P131_left_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_left_hi.append(np.mean(delta_hi))
data_left_lo.append(np.mean(delta_lo))

del P131_left, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for A65_left ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["A65_left_epoch_{0}".format(1+x)] = A65_left.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["A65_left_epoch_{0}".format(1+x)] = A65_left.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["A65_left_epoch_{0}".format(1+x)] = d1_pre.get("A65_left_epoch_{0}".format(1+x))[0,:]
    d1_post["A65_left_epoch_{0}".format(1+x)] = d1_post.get("A65_left_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["A65_left_data_{0}".format(1+y)] = d1_pre.get("A65_left_epoch_{0}".format(1+y))[0]
    d2_post["A65_left_data_{0}".format(1+y)] = d1_post.get("A65_left_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_left_hi.append(np.mean(delta_hi))
data_left_lo.append(np.mean(delta_lo))

del A65_left, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P81_left ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P81_left_epoch_{0}".format(1+x)] = P81_left.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P81_left_epoch_{0}".format(1+x)] = P81_left.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P81_left_epoch_{0}".format(1+x)] = d1_pre.get("P81_left_epoch_{0}".format(1+x))[0,:]
    d1_post["P81_left_epoch_{0}".format(1+x)] = d1_post.get("P81_left_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P81_left_data_{0}".format(1+y)] = d1_pre.get("P81_left_epoch_{0}".format(1+y))[0]
    d2_post["P81_left_data_{0}".format(1+y)] = d1_post.get("P81_left_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_left_hi.append(np.mean(delta_hi))
data_left_lo.append(np.mean(delta_lo))

del P81_left, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P93_left ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P93_left_epoch_{0}".format(1+x)] = P93_left.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P93_left_epoch_{0}".format(1+x)] = P93_left.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P93_left_epoch_{0}".format(1+x)] = d1_pre.get("P93_left_epoch_{0}".format(1+x))[0,:]
    d1_post["P93_left_epoch_{0}".format(1+x)] = d1_post.get("P93_left_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P93_left_data_{0}".format(1+y)] = d1_pre.get("P93_left_epoch_{0}".format(1+y))[0]
    d2_post["P93_left_data_{0}".format(1+y)] = d1_post.get("P93_left_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_left_hi.append(np.mean(delta_hi))
data_left_lo.append(np.mean(delta_lo))

del P93_left, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for A77_left ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["A77_left_epoch_{0}".format(1+x)] = A77_left.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["A77_left_epoch_{0}".format(1+x)] = A77_left.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["A77_left_epoch_{0}".format(1+x)] = d1_pre.get("A77_left_epoch_{0}".format(1+x))[0,:]
    d1_post["A77_left_epoch_{0}".format(1+x)] = d1_post.get("A77_left_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["A77_left_data_{0}".format(1+y)] = d1_pre.get("A77_left_epoch_{0}".format(1+y))[0]
    d2_post["A77_left_data_{0}".format(1+y)] = d1_post.get("A77_left_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_left_hi.append(np.mean(delta_hi))
data_left_lo.append(np.mean(delta_lo))

del A77_left, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for A80_left ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["A80_left_epoch_{0}".format(1+x)] = A80_left.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["A80_left_epoch_{0}".format(1+x)] = A80_left.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["A80_left_epoch_{0}".format(1+x)] = d1_pre.get("A80_left_epoch_{0}".format(1+x))[0,:]
    d1_post["A80_left_epoch_{0}".format(1+x)] = d1_post.get("A80_left_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["A80_left_data_{0}".format(1+y)] = d1_pre.get("A80_left_epoch_{0}".format(1+y))[0]
    d2_post["A80_left_data_{0}".format(1+y)] = d1_post.get("A80_left_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_left_hi.append(np.mean(delta_hi))
data_left_lo.append(np.mean(delta_lo))

del A80_left, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P43_left ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P43_left_epoch_{0}".format(1+x)] = P43_left.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P43_left_epoch_{0}".format(1+x)] = P43_left.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P43_left_epoch_{0}".format(1+x)] = d1_pre.get("P43_left_epoch_{0}".format(1+x))[0,:]
    d1_post["P43_left_epoch_{0}".format(1+x)] = d1_post.get("P43_left_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P43_left_data_{0}".format(1+y)] = d1_pre.get("P43_left_epoch_{0}".format(1+y))[0]
    d2_post["P43_left_data_{0}".format(1+y)] = d1_post.get("P43_left_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_left_hi.append(np.mean(delta_hi))
data_left_lo.append(np.mean(delta_lo))

del P43_left, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for A39_right ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["A39_right_epoch_{0}".format(1+x)] = A39_right.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["A39_right_epoch_{0}".format(1+x)] = A39_right.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["A39_right_epoch_{0}".format(1+x)] = d1_pre.get("A39_right_epoch_{0}".format(1+x))[0,:]
    d1_post["A39_right_epoch_{0}".format(1+x)] = d1_post.get("A39_right_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["A39_right_data_{0}".format(1+y)] = d1_pre.get("A39_right_epoch_{0}".format(1+y))[0]
    d2_post["A39_right_data_{0}".format(1+y)] = d1_post.get("A39_right_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_right_hi.append(np.mean(delta_hi))
data_right_lo.append(np.mean(delta_lo))

del A39_right, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for A92_right ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["A92_right_epoch_{0}".format(1+x)] = A92_right.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["A92_right_epoch_{0}".format(1+x)] = A92_right.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["A92_right_epoch_{0}".format(1+x)] = d1_pre.get("A92_right_epoch_{0}".format(1+x))[0,:]
    d1_post["A92_right_epoch_{0}".format(1+x)] = d1_post.get("A92_right_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["A92_right_data_{0}".format(1+y)] = d1_pre.get("A92_right_epoch_{0}".format(1+y))[0]
    d2_post["A92_right_data_{0}".format(1+y)] = d1_post.get("A92_right_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_right_hi.append(np.mean(delta_hi))
data_right_lo.append(np.mean(delta_lo))

del A92_right, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo
 
# ---------- Analysis for A30_right ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["A30_right_epoch_{0}".format(1+x)] = A30_right.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["A30_right_epoch_{0}".format(1+x)] = A30_right.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["A30_right_epoch_{0}".format(1+x)] = d1_pre.get("A30_right_epoch_{0}".format(1+x))[0,:]
    d1_post["A30_right_epoch_{0}".format(1+x)] = d1_post.get("A30_right_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["A30_right_data_{0}".format(1+y)] = d1_pre.get("A30_right_epoch_{0}".format(1+y))[0]
    d2_post["A30_right_data_{0}".format(1+y)] = d1_post.get("A30_right_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_right_hi.append(np.mean(delta_hi))
data_right_lo.append(np.mean(delta_lo))

del A30_right, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo
 
# ---------- Analysis for A101_right ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["A101_right_epoch_{0}".format(1+x)] = A101_right.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["A101_right_epoch_{0}".format(1+x)] = A101_right.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["A101_right_epoch_{0}".format(1+x)] = d1_pre.get("A101_right_epoch_{0}".format(1+x))[0,:]
    d1_post["A101_right_epoch_{0}".format(1+x)] = d1_post.get("A101_right_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["A101_right_data_{0}".format(1+y)] = d1_pre.get("A101_right_epoch_{0}".format(1+y))[0]
    d2_post["A101_right_data_{0}".format(1+y)] = d1_post.get("A101_right_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_right_hi.append(np.mean(delta_hi))
data_right_lo.append(np.mean(delta_lo))

del A101_right, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P16_right ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P16_right_epoch_{0}".format(1+x)] = P16_right.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P16_right_epoch_{0}".format(1+x)] = P16_right.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P16_right_epoch_{0}".format(1+x)] = d1_pre.get("P16_right_epoch_{0}".format(1+x))[0,:]
    d1_post["P16_right_epoch_{0}".format(1+x)] = d1_post.get("P16_right_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P16_right_data_{0}".format(1+y)] = d1_pre.get("P16_right_epoch_{0}".format(1+y))[0]
    d2_post["P16_right_data_{0}".format(1+y)] = d1_post.get("P16_right_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_right_hi.append(np.mean(delta_hi))
data_right_lo.append(np.mean(delta_lo))

del P16_right, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P53_right ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P53_right_epoch_{0}".format(1+x)] = P53_right.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P53_right_epoch_{0}".format(1+x)] = P53_right.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P53_right_epoch_{0}".format(1+x)] = d1_pre.get("P53_right_epoch_{0}".format(1+x))[0,:]
    d1_post["P53_right_epoch_{0}".format(1+x)] = d1_post.get("P53_right_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P53_right_data_{0}".format(1+y)] = d1_pre.get("P53_right_epoch_{0}".format(1+y))[0]
    d2_post["P53_right_data_{0}".format(1+y)] = d1_post.get("P53_right_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_right_hi.append(np.mean(delta_hi))
data_right_lo.append(np.mean(delta_lo))

del P53_right, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for A50_right ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["A50_right_epoch_{0}".format(1+x)] = A50_right.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["A50_right_epoch_{0}".format(1+x)] = A50_right.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["A50_right_epoch_{0}".format(1+x)] = d1_pre.get("A50_right_epoch_{0}".format(1+x))[0,:]
    d1_post["A50_right_epoch_{0}".format(1+x)] = d1_post.get("A50_right_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["A50_right_data_{0}".format(1+y)] = d1_pre.get("A50_right_epoch_{0}".format(1+y))[0]
    d2_post["A50_right_data_{0}".format(1+y)] = d1_post.get("A50_right_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_right_hi.append(np.mean(delta_hi))
data_right_lo.append(np.mean(delta_lo))

del A50_right, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P21_right ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P21_right_epoch_{0}".format(1+x)] = P21_right.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P21_right_epoch_{0}".format(1+x)] = P21_right.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P21_right_epoch_{0}".format(1+x)] = d1_pre.get("P21_right_epoch_{0}".format(1+x))[0,:]
    d1_post["P21_right_epoch_{0}".format(1+x)] = d1_post.get("P21_right_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P21_right_data_{0}".format(1+y)] = d1_pre.get("P21_right_epoch_{0}".format(1+y))[0]
    d2_post["P21_right_data_{0}".format(1+y)] = d1_post.get("P21_right_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_right_hi.append(np.mean(delta_hi))
data_right_lo.append(np.mean(delta_lo))

del P21_right, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P45_right ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P45_right_epoch_{0}".format(1+x)] = P45_right.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P45_right_epoch_{0}".format(1+x)] = P45_right.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P45_right_epoch_{0}".format(1+x)] = d1_pre.get("P45_right_epoch_{0}".format(1+x))[0,:]
    d1_post["P45_right_epoch_{0}".format(1+x)] = d1_post.get("P45_right_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P45_right_data_{0}".format(1+y)] = d1_pre.get("P45_right_epoch_{0}".format(1+y))[0]
    d2_post["P45_right_data_{0}".format(1+y)] = d1_post.get("P45_right_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_right_hi.append(np.mean(delta_hi))
data_right_lo.append(np.mean(delta_lo))

del P45_right, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P162_right ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P162_right_epoch_{0}".format(1+x)] = P162_right.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P162_right_epoch_{0}".format(1+x)] = P162_right.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P162_right_epoch_{0}".format(1+x)] = d1_pre.get("P162_right_epoch_{0}".format(1+x))[0,:]
    d1_post["P162_right_epoch_{0}".format(1+x)] = d1_post.get("P162_right_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P162_right_data_{0}".format(1+y)] = d1_pre.get("P162_right_epoch_{0}".format(1+y))[0]
    d2_post["P162_right_data_{0}".format(1+y)] = d1_post.get("P162_right_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_right_hi.append(np.mean(delta_hi))
data_right_lo.append(np.mean(delta_lo))

del P162_right, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P97_right ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P97_right_epoch_{0}".format(1+x)] = P97_right.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P97_right_epoch_{0}".format(1+x)] = P97_right.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P97_right_epoch_{0}".format(1+x)] = d1_pre.get("P97_right_epoch_{0}".format(1+x))[0,:]
    d1_post["P97_right_epoch_{0}".format(1+x)] = d1_post.get("P97_right_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P97_right_data_{0}".format(1+y)] = d1_pre.get("P97_right_epoch_{0}".format(1+y))[0]
    d2_post["P97_right_data_{0}".format(1+y)] = d1_post.get("P97_right_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_right_hi.append(np.mean(delta_hi))
data_right_lo.append(np.mean(delta_lo))

del P97_right, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P05_right ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P05_right_epoch_{0}".format(1+x)] = P05_right.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P05_right_epoch_{0}".format(1+x)] = P05_right.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P05_right_epoch_{0}".format(1+x)] = d1_pre.get("P05_right_epoch_{0}".format(1+x))[0,:]
    d1_post["P05_right_epoch_{0}".format(1+x)] = d1_post.get("P05_right_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P05_right_data_{0}".format(1+y)] = d1_pre.get("P05_right_epoch_{0}".format(1+y))[0]
    d2_post["P05_right_data_{0}".format(1+y)] = d1_post.get("P05_right_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_right_hi.append(np.mean(delta_hi))
data_right_lo.append(np.mean(delta_lo))

del P05_right, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for A104_right ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["A104_right_epoch_{0}".format(1+x)] = A104_right.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["A104_right_epoch_{0}".format(1+x)] = A104_right.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["A104_right_epoch_{0}".format(1+x)] = d1_pre.get("A104_right_epoch_{0}".format(1+x))[0,:]
    d1_post["A104_right_epoch_{0}".format(1+x)] = d1_post.get("A104_right_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["A104_right_data_{0}".format(1+y)] = d1_pre.get("A104_right_epoch_{0}".format(1+y))[0]
    d2_post["A104_right_data_{0}".format(1+y)] = d1_post.get("A104_right_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_right_hi.append(np.mean(delta_hi))
data_right_lo.append(np.mean(delta_lo))

del A104_right, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for A70_right ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["A70_right_epoch_{0}".format(1+x)] = A70_right.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["A70_right_epoch_{0}".format(1+x)] = A70_right.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["A70_right_epoch_{0}".format(1+x)] = d1_pre.get("A70_right_epoch_{0}".format(1+x))[0,:]
    d1_post["A70_right_epoch_{0}".format(1+x)] = d1_post.get("A70_right_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["A70_right_data_{0}".format(1+y)] = d1_pre.get("A70_right_epoch_{0}".format(1+y))[0]
    d2_post["A70_right_data_{0}".format(1+y)] = d1_post.get("A70_right_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_right_hi.append(np.mean(delta_hi))
data_right_lo.append(np.mean(delta_lo))

del A70_right, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P83_right ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P83_right_epoch_{0}".format(1+x)] = P83_right.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P83_right_epoch_{0}".format(1+x)] = P83_right.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P83_right_epoch_{0}".format(1+x)] = d1_pre.get("P83_right_epoch_{0}".format(1+x))[0,:]
    d1_post["P83_right_epoch_{0}".format(1+x)] = d1_post.get("P83_right_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P83_right_data_{0}".format(1+y)] = d1_pre.get("P83_right_epoch_{0}".format(1+y))[0]
    d2_post["P83_right_data_{0}".format(1+y)] = d1_post.get("P83_right_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_right_hi.append(np.mean(delta_hi))
data_right_lo.append(np.mean(delta_lo))

del P83_right, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P28_right ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P28_right_epoch_{0}".format(1+x)] = P28_right.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P28_right_epoch_{0}".format(1+x)] = P28_right.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P28_right_epoch_{0}".format(1+x)] = d1_pre.get("P28_right_epoch_{0}".format(1+x))[0,:]
    d1_post["P28_right_epoch_{0}".format(1+x)] = d1_post.get("P28_right_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P28_right_data_{0}".format(1+y)] = d1_pre.get("P28_right_epoch_{0}".format(1+y))[0]
    d2_post["P28_right_data_{0}".format(1+y)] = d1_post.get("P28_right_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_right_hi.append(np.mean(delta_hi))
data_right_lo.append(np.mean(delta_lo))

del P28_right, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for A11_right ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["A11_right_epoch_{0}".format(1+x)] = A11_right.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["A11_right_epoch_{0}".format(1+x)] = A11_right.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["A11_right_epoch_{0}".format(1+x)] = d1_pre.get("A11_right_epoch_{0}".format(1+x))[0,:]
    d1_post["A11_right_epoch_{0}".format(1+x)] = d1_post.get("A11_right_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["A11_right_data_{0}".format(1+y)] = d1_pre.get("A11_right_epoch_{0}".format(1+y))[0]
    d2_post["A11_right_data_{0}".format(1+y)] = d1_post.get("A11_right_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_right_hi.append(np.mean(delta_hi))
data_right_lo.append(np.mean(delta_lo))

del A11_right, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P109_right ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P109_right_epoch_{0}".format(1+x)] = P109_right.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P109_right_epoch_{0}".format(1+x)] = P109_right.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P109_right_epoch_{0}".format(1+x)] = d1_pre.get("P109_right_epoch_{0}".format(1+x))[0,:]
    d1_post["P109_right_epoch_{0}".format(1+x)] = d1_post.get("P109_right_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P109_right_data_{0}".format(1+y)] = d1_pre.get("P109_right_epoch_{0}".format(1+y))[0]
    d2_post["P109_right_data_{0}".format(1+y)] = d1_post.get("P109_right_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_right_hi.append(np.mean(delta_hi))
data_right_lo.append(np.mean(delta_lo))

del P109_right, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P23_right ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P23_right_epoch_{0}".format(1+x)] = P23_right.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P23_right_epoch_{0}".format(1+x)] = P23_right.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P23_right_epoch_{0}".format(1+x)] = d1_pre.get("P23_right_epoch_{0}".format(1+x))[0,:]
    d1_post["P23_right_epoch_{0}".format(1+x)] = d1_post.get("P23_right_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P23_right_data_{0}".format(1+y)] = d1_pre.get("P23_right_epoch_{0}".format(1+y))[0]
    d2_post["P23_right_data_{0}".format(1+y)] = d1_post.get("P23_right_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_right_hi.append(np.mean(delta_hi))
data_right_lo.append(np.mean(delta_lo))

del P23_right, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P131_right ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P131_right_epoch_{0}".format(1+x)] = P131_right.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P131_right_epoch_{0}".format(1+x)] = P131_right.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P131_right_epoch_{0}".format(1+x)] = d1_pre.get("P131_right_epoch_{0}".format(1+x))[0,:]
    d1_post["P131_right_epoch_{0}".format(1+x)] = d1_post.get("P131_right_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P131_right_data_{0}".format(1+y)] = d1_pre.get("P131_right_epoch_{0}".format(1+y))[0]
    d2_post["P131_right_data_{0}".format(1+y)] = d1_post.get("P131_right_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_right_hi.append(np.mean(delta_hi))
data_right_lo.append(np.mean(delta_lo))

del P131_right, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for A65_right ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["A65_right_epoch_{0}".format(1+x)] = A65_right.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["A65_right_epoch_{0}".format(1+x)] = A65_right.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["A65_right_epoch_{0}".format(1+x)] = d1_pre.get("A65_right_epoch_{0}".format(1+x))[0,:]
    d1_post["A65_right_epoch_{0}".format(1+x)] = d1_post.get("A65_right_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["A65_right_data_{0}".format(1+y)] = d1_pre.get("A65_right_epoch_{0}".format(1+y))[0]
    d2_post["A65_right_data_{0}".format(1+y)] = d1_post.get("A65_right_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_right_hi.append(np.mean(delta_hi))
data_right_lo.append(np.mean(delta_lo))

del A65_right, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P81_right ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P81_right_epoch_{0}".format(1+x)] = P81_right.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P81_right_epoch_{0}".format(1+x)] = P81_right.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P81_right_epoch_{0}".format(1+x)] = d1_pre.get("P81_right_epoch_{0}".format(1+x))[0,:]
    d1_post["P81_right_epoch_{0}".format(1+x)] = d1_post.get("P81_right_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P81_right_data_{0}".format(1+y)] = d1_pre.get("P81_right_epoch_{0}".format(1+y))[0]
    d2_post["P81_right_data_{0}".format(1+y)] = d1_post.get("P81_right_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_right_hi.append(np.mean(delta_hi))
data_right_lo.append(np.mean(delta_lo))

del P81_right, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P93_right ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P93_right_epoch_{0}".format(1+x)] = P93_right.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P93_right_epoch_{0}".format(1+x)] = P93_right.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P93_right_epoch_{0}".format(1+x)] = d1_pre.get("P93_right_epoch_{0}".format(1+x))[0,:]
    d1_post["P93_right_epoch_{0}".format(1+x)] = d1_post.get("P93_right_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P93_right_data_{0}".format(1+y)] = d1_pre.get("P93_right_epoch_{0}".format(1+y))[0]
    d2_post["P93_right_data_{0}".format(1+y)] = d1_post.get("P93_right_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_right_hi.append(np.mean(delta_hi))
data_right_lo.append(np.mean(delta_lo))

del P93_right, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for A77_right ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["A77_right_epoch_{0}".format(1+x)] = A77_right.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["A77_right_epoch_{0}".format(1+x)] = A77_right.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["A77_right_epoch_{0}".format(1+x)] = d1_pre.get("A77_right_epoch_{0}".format(1+x))[0,:]
    d1_post["A77_right_epoch_{0}".format(1+x)] = d1_post.get("A77_right_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["A77_right_data_{0}".format(1+y)] = d1_pre.get("A77_right_epoch_{0}".format(1+y))[0]
    d2_post["A77_right_data_{0}".format(1+y)] = d1_post.get("A77_right_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_right_hi.append(np.mean(delta_hi))
data_right_lo.append(np.mean(delta_lo))

del A77_right, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for A80_right ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["A80_right_epoch_{0}".format(1+x)] = A80_right.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["A80_right_epoch_{0}".format(1+x)] = A80_right.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["A80_right_epoch_{0}".format(1+x)] = d1_pre.get("A80_right_epoch_{0}".format(1+x))[0,:]
    d1_post["A80_right_epoch_{0}".format(1+x)] = d1_post.get("A80_right_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["A80_right_data_{0}".format(1+y)] = d1_pre.get("A80_right_epoch_{0}".format(1+y))[0]
    d2_post["A80_right_data_{0}".format(1+y)] = d1_post.get("A80_right_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_right_hi.append(np.mean(delta_hi))
data_right_lo.append(np.mean(delta_lo))

del A80_right, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Analysis for P43_right ---------- #
pre_time_start = pre_default_time_start
pre_time_end = pre_default_time_end

post_time_start = post_default_time_start
post_time_end = post_default_time_end

d1_pre = {}
d1_post = {}
for x in range(n):

    pre_time_start = pre_time_start + 4.5
    pre_time_end = pre_time_end + 4.5

    post_time_start = post_time_start + 4.5
    post_time_end = post_time_end + 4.5

    d1_pre["P43_right_epoch_{0}".format(1+x)] = P43_right.copy().crop(tmin=pre_time_start, tmax=pre_time_end)
    d1_post["P43_right_epoch_{0}".format(1+x)] = P43_right.copy().crop(tmin=post_time_start, tmax=post_time_end)
    
    d1_pre["P43_right_epoch_{0}".format(1+x)] = d1_pre.get("P43_right_epoch_{0}".format(1+x))[0,:]
    d1_post["P43_right_epoch_{0}".format(1+x)] = d1_post.get("P43_right_epoch_{0}".format(1+x))[0,:]
    
d2_pre = {} #dictioanry for extracted epoch data
d2_post = {}
for y in range (n):
    d2_pre["P43_right_data_{0}".format(1+y)] = d1_pre.get("P43_right_epoch_{0}".format(1+y))[0]
    d2_post["P43_right_data_{0}".format(1+y)] = d1_post.get("P43_right_epoch_{0}".format(1+y))[0]

d3_pre = {} #dictionary for CWT epoch data
d3_post = {}

pre_hi_temp = []
pre_lo_temp = []

post_hi_temp = []
post_lo_temp = []

for key, value in d2_pre.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    pre_hi_temp.append(power[0,0])
    pre_lo_temp.append(power[1,0])
    d3_pre[key] = {'coefficients': power} 
    
for key, value in d2_post.items():
    cwt, freq = pywt.cwt(value, scales, wavelet)
    squared = np.square(cwt)
    power = np.mean(squared, axis =2)
    post_hi_temp.append(power[0,0])
    post_lo_temp.append(power[1,0])
    d3_post[key] = {'coefficients': power}
    
for z in range (n):
    delta_hi = [a - b for a, b in zip(post_hi_temp, pre_hi_temp)]
    delta_lo = [a - b for a, b in zip(post_lo_temp, pre_lo_temp)]

data_right_hi.append(np.mean(delta_hi))
data_right_lo.append(np.mean(delta_lo))

del P43_right, d1_pre, d1_post, d2_pre, d2_post, d3_pre, d3_post, x, y, z, pre_hi_temp, pre_lo_temp, post_hi_temp, post_lo_temp, cwt, freq, squared, power, delta_hi, delta_lo

# ---------- Creating separate dataframes based on dominant and non-dominant hands ---------- #
df['Left 12Hz'] = data_left_hi
df['Left 6Hz'] = data_left_lo

df['Right 12Hz'] = data_right_hi
df['Right 6Hz'] = data_right_lo

df_right_dom = df.loc[df['Hand'] == 1, ['Patient ID', 'Right 12Hz', 'Right 6Hz']]
df_left_dom = df.loc[df['Hand'] == 2, ['Patient ID', 'Left 12Hz', 'Left 6Hz']]

df_right_dom = df_right_dom.rename(columns={'Right 12Hz': 'Dominant 12Hz', 'Right 6Hz': 'Dominant 6Hz'})
df_left_dom = df_left_dom.rename(columns={'Left 12Hz': 'Dominant 12Hz', 'Left 6Hz':'Dominant 6Hz'})

data_dominant = pd.concat([df_right_dom, df_left_dom])

df_right_nondom = df.loc[df['Hand'] == 1, ['Patient ID', 'Left 12Hz', 'Left 6Hz']]
df_left_nondom = df.loc[df['Hand'] == 2, ['Patient ID', 'Right 12Hz', 'Right 6Hz']]

df_right_nondom = df_right_nondom.rename(columns={'Left 12Hz': 'Nondominant 12Hz', 'Left 6Hz': 'Nondominant 6Hz'})
df_left_nondom = df_left_nondom.rename(columns={'Right 12Hz': 'Nondominant 12Hz', 'Right 6Hz':'Nondominant 6Hz'})

data_nondominant = pd.concat([df_right_nondom, df_left_nondom])

data_dominant['Dominant 12Hz'] = np.real(data_dominant['Dominant 12Hz'])
data_dominant['Dominant 6Hz'] = np.real(data_dominant['Dominant 6Hz'])

data_nondominant['Nondominant 12Hz'] = np.real(data_nondominant['Nondominant 12Hz'])
data_nondominant['Nondominant 6Hz'] = np.real(data_nondominant['Nondominant 6Hz'])

df = pd.merge(df, data_dominant, on='Patient ID')
df = pd.merge(df, data_nondominant, on='Patient ID')

del data_left_hi, data_left_lo, data_right_hi, data_right_lo, df_right_dom, df_left_dom, df_right_nondom, df_left_nondom

# ---------- Selecting Subgroups for analysis ---------- #
hand = 'Nondominant' #pick either "dominant" or "Non-dominant"
#Type = 'Genetic' # pick either "Genetic", "Acquired", or "Idiopathic"
#DBS = 2 #pick either "1" for no DBS, or "2" for treated with DBS


#df = df[(df['Type'] == Type) & (df['Treatment'] == DBS)]
df = df[(df['Age Group'] == 3)]

hand_hi= hand + ' 12Hz'
hand_lo = hand + ' 6Hz'

# ---------- Standardising the data ---------- #
scaler = StandardScaler()

df[['Dominant 12Hz', 'Dominant 6Hz']] = scaler.fit_transform(df[['Dominant 12Hz', 'Dominant 6Hz']])
df[['Nondominant 12Hz', 'Nondominant 6Hz']] = scaler.fit_transform(df[['Nondominant 12Hz', 'Nondominant 6Hz']])
# ---------- Determining optimal K number ---------- # 
sse = []
size = len (df)
size=size+1
for k in range(1, size):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(df[[hand_hi, hand_lo]])
    sse.append(kmeans.inertia_)

kl= KneeLocator(range(1, size), sse, curve="convex", direction="decreasing")

c=kl.elbow

sns.set_style("dark")
plt.plot(range(1, size), sse)
plt.xticks(range(1, size))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
#plt.title("Optimal K Number for "+ hand + " Hand Analysis of " + Type + ' Dystonia' )
plt.title("Optimal K Number for "+ hand + " Hand Analysis of  Dystonia" )
plt.show()

# ---------- Applying the K means clustering ---------- #
kmeans = KMeans(n_clusters = c, **kmeans_kwargs)

kmeans.fit(df[[hand_hi, hand_lo]])

labels = kmeans.labels_

df["Predicted Group"]= kmeans.labels_

# ---------- Plotting ---------- #
sns.set_theme(rc={"figure.dpi": 300})
#t = hand + ' Hand Analysis of ' + Type +' Dystonia' + 'with DBS ='     
t = hand + ' Hand Analysis of Dystonia' 

sns.scatterplot(df, x = hand_hi, y = hand_lo, hue = "Predicted Group", style = "Etiology", size="Treatment", sizes=(50, 200)).set(title=t)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

# ---------- Printing Statistics and Credits---------- #
print("==========================================================================================")
#print(f"Subgroup analysis for {hand} hands in Dystonia with DBS={DBS}")
print("The total number of data points is: {}".format(size-1))
print("The optimal K number is: {}".format(c))
print("The total time elapsed was: {0:0.1f} seconds".format(time.time() - start))
print("------------------------------------------------------------------------------------------")
print("Script as part of submission for 5MBBS303 Scholarly Project")
print("Supervised by Dr Verity McClelland & Dr Crina-Daniela Grosan")
print("Last updated: {}".format(last_updated))
print("==========================================================================================")


