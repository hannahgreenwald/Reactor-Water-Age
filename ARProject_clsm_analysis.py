#!/usr/bin/env python
# coding: utf-8

# # CLSM Biofilm Z-stacks Analysis for the AR Water Age Project

# This notebook was developed by Hannah Greenwald with assistance from Rose Kantor 2020-2023 in order to quantitatively analyze drinking water biofilm z stack images taken on a Zeiss 710 LSM confocal microscope. The samples were stained with 3 fluorescent dyes and imaged across 3 channels. 

# Necessary installs to run this code on a PC:
# 
# 1. need to install a C compiler and Java developer kit and Windows 10 SDK to use bioformats
# 2. Go to anaconda prompt, change directory to C://, then pip install numpy, scikitimage, javabridge, and python-bioformats
# 
# To run on a MAC:
# 1. Install updated java. Install java JDK. Install java JRE v8 (from java websites). MUST MAKE SURE JAVA_HOME IS SET TO THE JAVA 8 VERSION
# 2. Install C compiler by typing "xcode-select --install" into terminal
# 3. Install numpy and make sure it's up to date
# 2. Pip install python-javabridge and python-bioformats, then scikitimage
# 
# NOTE: installing javabridge with the correct dependencies, versions, and paths involved a lot of troubleshooting
# 
# Helpful links for troubleshooting:
# * https://github.com/LeeKamentsky/python-javabridge/issues/168
# * https://github.com/LeeKamentsky/python-javabridge/issues/168#issuecomment-706401008
# * https://github.com/LeeKamentsky/python-javabridge/issues/152
# * https://stackoverflow.com/questions/21964709/how-to-set-or-change-the-default-java-jdk-version-on-macos
# * https://pythonhosted.org/python-bioformats/
# * https://forum.image.sc/t/problems-installing-python-bioformats-java-not-found/47977
# * https://stackoverflow.com/questions/73237204/changing-jupyter-notebook-java-version
# * https://stackoverflow.com/questions/5178292/pip-install-mysql-python-fails-with-environmenterror-mysql-config-not-found
# * https://github.com/LeeKamentsky/python-javabridge/issues/167
# 

# Explanation of dye names:
# 
# * In the original file acquisition, the name of Channel 3 was Ch1-T3 which corresponded to orange. Emission wavelength 590, excitation wavelength 488, pinhole 22.5
# * The name of channel 2 was Ch2 M-T2, emission wavelength 515.5, excitation 488, pinhole 22.5
# * The name of channel 1 was Ch3-T1, emission 707.5, excitation 633, pinhole 21.4
# 
# Therefore, 
# * o.image().Pixels.Channel(0).Name is Ch3-T1 which corresponds to Alexa Fluor 647 (polysaccharides)
# * o.image().Pixels.Channel(1).Name, name of channel 1 is Ch2 M-T2, this is for sure syto9 (nucleic acids, cells)
# * o.image().Pixels.Channel(2).Name, name of channel 2 is Ch1-T3, which is Sypro Orange, fluor 488
# 
# In final table:
# * Ch1 corresponds to Alexa Fluor 647 (polysaccharides)
# * Ch2 corresponds to syto9 (nucleic acids, cells)
# * Ch3 corresponds to Sypro Orange, fluor 488

# In[3]:


# pip install --upgrade numpy
# pip install scikit-image
# pip install matplotlib
# pip install opencv-python
# pip install session_info


# In[130]:


# os.environ['JAVA_HOME'] = '/Library/Java/JavaVirtualMachines/jdk1.8.0_361.jdk/Contents/Home'
# # os.environ['PATH'] = '/usr/lib/jvm/jre/bin:' + os.environ['PATH']


# ## Import Packages and Start Java VM

# In[1]:


import os
import csv
import numpy as np
import pandas as pd
import matplotlib
import cv2
import glob

os.environ['JAVA_HOME'] = '/Library/Java/JavaVirtualMachines/jdk1.8.0_361.jdk/Contents/Home'
    #this line sets which java version to use. need to use v8 even though it's old!

import bioformats #lets us read in .czi files, https://pythonhosted.org/python-bioformats/ 
import javabridge
from skimage import data #https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_otsu 
from skimage.filters import threshold_otsu
from skimage.filters import try_all_threshold
from skimage.filters import median as image_median

#You can only start the virtual machine once per session. If you exit and need to reenter, must restart the kernal.
javabridge.start_vm(class_path=bioformats.JARS)


# In[5]:


import session_info
session_info.show()


# ## Load Images and Access Metadata

# In[49]:


#import metadata table
metadata_full = pd.read_csv('/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/CLSM_biofilm_python/AR_MasterDataSheet-clsm.tsv', sep='\t')

metadata_round2 = metadata_full[metadata_full["round2"] == "Y"]
metadata_full = metadata_full[(metadata_full["control_or_sample"] != "discard") & (metadata_full["round2"] != "Y")]


# In[4]:


#maybe filter out these samples so only considering first 6 controls of each type (similar to sample numbers)
# metadata[(metadata["control_or_sample"] == "control ")& (metadata["date"] == 101920) & (metadata["micr_replicate"] <= 6) ]


# In[5]:


#practice using bioformats to extract information from a .czi file

# control3 = '/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/CLSM_biofilm_python/101920_nodye_blank_6.czi'
# control3 = '/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/CLSM_biofilm_python/20_0104_AR3_5.czi'


# m = bioformats.get_omexml_metadata(control3)
# o = bioformats.OMEXML(m)
# stack_count = o.image().Pixels.SizeZ

# # # access metadata
# # image_name = o.image().Pixels.SizeT
# # image_name
# # o.instrument().Detector.ID
# o.image().AcquisitionDate
# # stack_count = o.image().Pixels.SizeZ
# # channel_count = o.image().Pixels.SizeC
# # timepoint_count = o.image().Pixels.SizeT # there should only be one timepoint

# ## can get names of channels 
# ch1 = o.image().Pixels.Channel(0).Name
# ch2 = o.image().Pixels.Channel(1).Name
# o.image().Pixels.Channel(2).Name


# ## Define Functions

# In[7]:


def thresholding_function(sample_image):
    '''using prewritten functions for median filtering and Otsu thresholding from scikit-image'''
    image_filter = image_median(sample_image) #more about median filtering: https://medium.com/@florestony5454/median-filtering-with-python-and-opencv-2bce390be0d1
    thresh = threshold_otsu(image_filter)
    image_binary =  image_filter > thresh #binary T/F arrays
    return(image_binary)
# # Thresholding
# # https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_thresholding.html#sphx-glr-auto-examples-segmentation-plot-thresholding-py


# In[8]:


def area_function(image_binary):
    '''using prewritten functions for median filtering and Otsu thresholding from scikit-image'''
    image_size = 1048576 #number of pixels in each z slice bc 1024x1024
    area_fraction = (np.sum(image_binary))/image_size
    area = area_fraction* (212.5*212.5) #convert area of each slice to microns squared
    return(area)


# In[9]:


def calculate_biovolume(sample_image):
    '''function for calculating biovolume of a thresholded image'''
    
    
def calculate_thickness(sample_image):
    '''function for calculating thickness of a thresholded image'''
    
# ... you get the idea


# In[10]:


# image_raw= bioformats.load_image(control1, rescale = False, z = z, c = 2)
# image_raw.size # (images are 1024x1024 pixels = 1048576)
# image_raw.sum()


# ## Decide Thresholding Method

# In[12]:


#practice paths

control1 = '/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/CLSM_biofilm_python/20_1215_controls/121520_slidecontrol_syto_sypro_conA_2.czi'
control2 = '/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/CLSM_biofilm_python/20_1019-controls/101920_syto_sypro_conA_blank_1.czi'
control3 = '/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/CLSM_biofilm_BiofilmQ/20_1019-controls/101920_nodye_blank_6.czi'
control4 = '/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/CLSM_biofilm_BiofilmQ/20_1019-controls/101920_syto_blank_4.czi'

image1=  '/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/biofilm_images/biofilm_images/19_1121_AR1_conA_Syto_sypro_3.czi'
image2 = '/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/CLSM_biofilm_BiofilmQ/20_1020/102020_syto_sypro_conA_AR1_3_C.czi'


# In[13]:


#figure out which thresholding method is best using the scikitimage function 
#skimage.filters.try_all_threshold(MxNimagename, figsize= (inches,inches), verbose=TRUE)
#compares threshold by isodata, li, mean, minimum, otsu, triangle, and yen
trial_image_slice = bioformats.load_image(control1, rescale = False, z = 1, c = 2)
fig, ax = try_all_threshold(trial_image_slice, verbose=False)
ax


# In[14]:


# cycle through all the controls, figure out how many z stacks to delete from the samples

#to get all images in file folder
dir = '/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/CLSM_biofilm_python/'

#maybe filter out these samples so only considering first 6 controls of each type (similar to sample numbers)
# metadata2 = metadata[(metadata["control_or_sample"] == "control ")& (metadata["date"] == 101920) & (metadata["micr_replicate"] <= 6) ]

stacks_over1= []
stack_count_total = []
for row in metadata.iterrows(): 
        if (row[1]["control_or_sample"] == "control"):
            file_name = (row[1]["file_name"])
            path= dir+ '/' + file_name
            m = bioformats.get_omexml_metadata(path)
            o = bioformats.OMEXML(m)
            stack_count_o = o.image().Pixels.SizeZ
            stack_count_total.append(stack_count_o)

            image_list = []
            z_list = np.arange(stack_count_o)
            image_size = 1048576 #number of pixels in each z slice bc 1024x1024
            areas = []
            area_fractions = []

            for z in z_list: 
                image_raw= bioformats.load_image(path, rescale = False, z = z, c = 2)
                image_filter = image_median(image_raw)
                thresh = threshold_otsu(image_filter)
                image_binary =  image_filter > thresh #binary T/F arrays
                image_list.append(image_binary)
                area_fraction = np.sum(image_binary)/image_size
                area = area_fraction* (212.5*212.5) #convert to area in microns squared
                new_row = [z, area]
                areas.append(area)
                area_fractions.append(area_fraction)

            stacks_over1_i = sum(np.array(area_fractions) > 0.01)  #count the frames in the z stack where the area coverage is >1%
            stacks_over1.append(stacks_over1_i)

np.median(np.array(stacks_over1))
# (np.array(stacks_over1)).mean() #the mean number of stacks in the controls with 
# np.percentile(stacks_over1, 75)
# (np.array(stacks_over1)).mean() #the mean number of stacks in the controls with 

#this set of controls had max 9 stacks with >1% coverage. 8.5 stacks in the 95th percentile, 6 in 75th, and median of 4
    


# ## Process Images (after substracting for controls)

# In[15]:


# create practice df for faster practice processing
metadata1 = metadata_full.iloc[0:21]
metadata2 = metadata_full.iloc[21:52]
metadata3 = metadata_full.iloc[52:73]
metadata4 = metadata_full.iloc[73:94]
metadata5 = metadata_full.iloc[94:115]
metadata6 = metadata_full.iloc[115:136]
metadata7 = metadata_full.iloc[136:157]
metadata8a = metadata_full.iloc[157:168] #168 is the messed up one
metadata8b = metadata_full.iloc[169:178] 
metadata9 = metadata_full.iloc[178:199] 
metadata10 = metadata_full.iloc[199:219] #end at 218
metadata11 = metadata_full.iloc[219:239] 
metadata12 = metadata_full.iloc[239:255] #end at 254
metadata13= metadata_round2

#PROCESS 169 ON ITS OWN, 20_0104_AR3_3.czi, idk why it's not working
#actually PROCESS 168 ON ITS OWN, 20_0104_AR3_2.czi, idk why it's not working


# ### Batch 13

# In[67]:


# start building the sample processing loops

metadata= metadata13 #need to parse out because computer crashes if try to run all files at once

#to get all images in file folder
dir = '/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/CLSM_biofilm_python/'

metadata["stack_count_total"] = ""
metadata["ch1_area_max"] = ""
metadata["ch2_area_max"] = ""
metadata["ch3_area_max"] = ""
metadata["total_area_max"] = ""

metadata["ch1_vol"] = ""
metadata["ch2_vol"] = ""
metadata["ch3_vol"] = ""
metadata["EPS_vol"] = ""
metadata["total_vol"] = ""
metadata["ch2v1_vol"] = ""
metadata["ch2v3_vol"] = ""


metadata['ch1_spread']= ""
metadata['ch2_spread']= ""
metadata['ch3_spread']= ""

metadata['ch1_height']= ""
metadata['ch2_height']= ""
metadata['ch3_height']= ""

for row in metadata.itertuples(): #loop through each file
        r= row.Index
        file_name = (row.file_name)
        path= dir+ '/' + file_name
        m = bioformats.get_omexml_metadata(path)
        o = bioformats.OMEXML(m)
        stack_count_o = o.image().Pixels.SizeZ
        # stack_count_total.append(stack_count_o)
        metadata.loc[r,'stack_count_total'] = int(stack_count_o)
        
        if stack_count_o > 4 : #if file is tall enough, loop through each z stack in the file
            z_list = np.arange(4, stack_count_o) #cut off first 4 slices bc controls tended to fluoresce in first 4 slices
            
            image_list_ch1 = []
            image_list_ch2 = []
            image_list_ch3 = []
            image_list_EPS = []
            image_list_total = []
            image_list_2v1 = []
            image_list_2v3 = []
            
            ch1_areas = []
            ch2_areas = []
            ch3_areas = []
            areas_EPS = []
            areas_total = []
            areas_2v1 = []
            areas_2v3 = []

            for z in z_list: 
                #load image, filter, and threshold for each channel
                image_raw_ch1= bioformats.load_image(path, rescale = False, z = z, c = 0)
                image_binary_ch1= thresholding_function(image_raw_ch1)
                
                image_raw_ch2= bioformats.load_image(path, rescale = False, z = z, c = 1)
                image_binary_ch2= thresholding_function(image_raw_ch2)
                
                image_raw_ch3= bioformats.load_image(path, rescale = False, z = z, c = 2)
                image_binary_ch3= thresholding_function(image_raw_ch3)
                
                #combine channels that we want to combine
                image_binary_EPS = np.logical_or(image_binary_ch1, image_binary_ch3) #combine ch1 and ch3 to see EPS
                image_binary_total = np.logical_or(image_binary_EPS, image_binary_ch2) #combine ch1, ch2, and ch3 to capture total biovolume
                image_binary_2v1 = np.logical_and(image_binary_ch1, image_binary_ch2) #look at percent overlap of ch1 with ch2 (just relative biovolume)
                image_binary_2v3 = np.logical_and(image_binary_ch3, image_binary_ch2) #look at percent overlap of ch3 with ch2 (just relative biovolume)
                
                #calculate parameters 
                area_ch1 = area_function(image_binary_ch1)
                area_ch2 = area_function(image_binary_ch2)
                area_ch3 = area_function(image_binary_ch3)
                area_EPS = area_function(image_binary_EPS)
                area_total = area_function(image_binary_total)
                area_2v1 = area_function(image_binary_2v1)
                area_2v3 = area_function(image_binary_2v3)
                
                #append parameters to lists if needed for whole image
                # image_list_ch1.append(image_binary_ch1)
                # image_list_ch2.append(image_binary_ch2)
                # image_list_ch3.append(image_binary_ch3)
                # image_list_EPS.append(image_binary_EPS)
                # image_list_total.append(image_binary_total)
                # image_list_2v1.append(image_binary_2v1)
                # image_list_2v3.append(image_binary_2v3)
                
                ch1_areas.append(area_ch1)
                ch2_areas.append(area_ch2)
                ch3_areas.append(area_ch3)
                areas_EPS.append(area_EPS)
                areas_total.append(area_total)
                areas_2v1.append(area_2v1)
                areas_2v3.append(area_2v3)
            
        else:
            ch1_areas = [0]
            ch2_areas = [0]
            ch3_areas = [0]
            areas_EPS = [0]
            areas_total = [0]
            areas_2v1 = [0]
            areas_2v3 = [0]
        
        metadata.loc[r,'ch1_area_max'] = max(ch1_areas) #max area of biofilm in ch1
        metadata.loc[r,'ch2_area_max'] = max(ch2_areas) #max area of biofilm in ch2
        metadata.loc[r,'ch3_area_max'] = max(ch3_areas) #max area of biofilm in ch3
        metadata.loc[r,'total_area_max'] = max(areas_total) #max area of biofilm in combined channels
       
        metadata.loc[r,'ch1_vol'] = sum(ch1_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2_vol'] = sum(ch2_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch3_vol'] = sum(ch3_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'EPS_vol'] = sum(areas_EPS) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'total_vol'] = sum(areas_total) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2v1_vol'] = sum(areas_2v1) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2v3_vol'] = sum(areas_2v3) *1 #volume calc, each stack is 1 um thick
        
        if metadata.loc[r,'ch1_area_max'] != 0:
            metadata.loc[r,'ch1_spread'] = metadata.loc[r,'ch1_vol']/metadata.loc[r,'ch1_area_max'] #spread is a proxy for height in Fish et al
        if metadata.loc[r,'ch2_area_max'] != 0:
            metadata.loc[r,'ch2_spread'] = metadata.loc[r,'ch2_vol']/metadata.loc[r,'ch2_area_max'] #spread is a proxy for height in Fish et al
        if metadata.loc[r,'ch3_area_max'] != 0:
            metadata.loc[r,'ch3_spread'] = metadata.loc[r,'ch3_vol']/metadata.loc[r,'ch3_area_max'] #spread is a proxy for height in Fish et al
        
        metadata.loc[r,'ch1_height'] = sum(np.array(ch1_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        metadata.loc[r,'ch2_height'] = sum(np.array(ch2_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        metadata.loc[r,'ch3_height'] = sum(np.array(ch3_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        
        metadata.loc[r,'ch1_height_0'] = sum(np.array(ch1_areas) > (0)) #max of stack count when stack count has at least 1 pixel
        metadata.loc[r,'ch2_height_0'] = sum(np.array(ch2_areas) > (0)) #max of stack count when stack count has at least 1 pixel
        metadata.loc[r,'ch3_height_0'] = sum(np.array(ch3_areas) > (0)) #max of stack count when stack count has at least 1 pixel

metadata13= metadata
    


# ### Batch 11 

# In[69]:


# start building the sample processing loops

metadata= metadata11 #need to parse out because computer crashes if try to run all files at once

#to get all images in file folder
dir = '/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/CLSM_biofilm_python/Hannah042220/'

metadata["stack_count_total"] = ""
metadata["ch1_area_max"] = ""
metadata["ch2_area_max"] = ""
metadata["ch3_area_max"] = ""
metadata["total_area_max"] = ""

metadata["ch1_vol"] = ""
metadata["ch2_vol"] = ""
metadata["ch3_vol"] = ""
metadata["EPS_vol"] = ""
metadata["total_vol"] = ""
metadata["ch2v1_vol"] = ""
metadata["ch2v3_vol"] = ""


metadata['ch1_spread']= ""
metadata['ch2_spread']= ""
metadata['ch3_spread']= ""

metadata['ch1_height']= ""
metadata['ch2_height']= ""
metadata['ch3_height']= ""

for row in metadata.itertuples(): #loop through each file
        r= row.Index
        file_name = (row.file_name)
        path= dir+ '/' + file_name
        m = bioformats.get_omexml_metadata(path)
        o = bioformats.OMEXML(m)
        stack_count_o = o.image().Pixels.SizeZ
        # stack_count_total.append(stack_count_o)
        metadata.loc[r,'stack_count_total'] = int(stack_count_o)
        
        if stack_count_o > 4 : #if file is tall enough, loop through each z stack in the file
            z_list = np.arange(4, stack_count_o) #cut off first 4 slices bc controls tended to fluoresce in first 4 slices
            
            image_list_ch1 = []
            image_list_ch2 = []
            image_list_ch3 = []
            image_list_EPS = []
            image_list_total = []
            image_list_2v1 = []
            image_list_2v3 = []
            
            ch1_areas = []
            ch2_areas = []
            ch3_areas = []
            areas_EPS = []
            areas_total = []
            areas_2v1 = []
            areas_2v3 = []

            for z in z_list: 
                #load image, filter, and threshold for each channel
                image_raw_ch1= bioformats.load_image(path, rescale = False, z = z, c = 0)
                image_binary_ch1= thresholding_function(image_raw_ch1)
                
                image_raw_ch2= bioformats.load_image(path, rescale = False, z = z, c = 1)
                image_binary_ch2= thresholding_function(image_raw_ch2)
                
                image_raw_ch3= bioformats.load_image(path, rescale = False, z = z, c = 2)
                image_binary_ch3= thresholding_function(image_raw_ch3)
                
                #combine channels that we want to combine
                image_binary_EPS = np.logical_or(image_binary_ch1, image_binary_ch3) #combine ch1 and ch3 to see EPS
                image_binary_total = np.logical_or(image_binary_EPS, image_binary_ch2) #combine ch1, ch2, and ch3 to capture total biovolume
                image_binary_2v1 = np.logical_and(image_binary_ch1, image_binary_ch2) #look at percent overlap of ch1 with ch2 (just relative biovolume)
                image_binary_2v3 = np.logical_and(image_binary_ch3, image_binary_ch2) #look at percent overlap of ch3 with ch2 (just relative biovolume)
                
                #calculate parameters 
                area_ch1 = area_function(image_binary_ch1)
                area_ch2 = area_function(image_binary_ch2)
                area_ch3 = area_function(image_binary_ch3)
                area_EPS = area_function(image_binary_EPS)
                area_total = area_function(image_binary_total)
                area_2v1 = area_function(image_binary_2v1)
                area_2v3 = area_function(image_binary_2v3)
                
                #append parameters to lists if needed for whole image
                # image_list_ch1.append(image_binary_ch1)
                # image_list_ch2.append(image_binary_ch2)
                # image_list_ch3.append(image_binary_ch3)
                # image_list_EPS.append(image_binary_EPS)
                # image_list_total.append(image_binary_total)
                # image_list_2v1.append(image_binary_2v1)
                # image_list_2v3.append(image_binary_2v3)
                
                ch1_areas.append(area_ch1)
                ch2_areas.append(area_ch2)
                ch3_areas.append(area_ch3)
                areas_EPS.append(area_EPS)
                areas_total.append(area_total)
                areas_2v1.append(area_2v1)
                areas_2v3.append(area_2v3)
            
        else:
            ch1_areas = [0]
            ch2_areas = [0]
            ch3_areas = [0]
            areas_EPS = [0]
            areas_total = [0]
            areas_2v1 = [0]
            areas_2v3 = [0]
        
        metadata.loc[r,'ch1_area_max'] = max(ch1_areas) #max area of biofilm in ch1
        metadata.loc[r,'ch2_area_max'] = max(ch2_areas) #max area of biofilm in ch2
        metadata.loc[r,'ch3_area_max'] = max(ch3_areas) #max area of biofilm in ch3
        metadata.loc[r,'total_area_max'] = max(areas_total) #max area of biofilm in combined channels
       
        metadata.loc[r,'ch1_vol'] = sum(ch1_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2_vol'] = sum(ch2_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch3_vol'] = sum(ch3_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'EPS_vol'] = sum(areas_EPS) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'total_vol'] = sum(areas_total) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2v1_vol'] = sum(areas_2v1) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2v3_vol'] = sum(areas_2v3) *1 #volume calc, each stack is 1 um thick
        
        if metadata.loc[r,'ch1_area_max'] != 0:
            metadata.loc[r,'ch1_spread'] = metadata.loc[r,'ch1_vol']/metadata.loc[r,'ch1_area_max'] #spread is a proxy for height in Fish et al
        if metadata.loc[r,'ch2_area_max'] != 0:
            metadata.loc[r,'ch2_spread'] = metadata.loc[r,'ch2_vol']/metadata.loc[r,'ch2_area_max'] #spread is a proxy for height in Fish et al
        if metadata.loc[r,'ch3_area_max'] != 0:
            metadata.loc[r,'ch3_spread'] = metadata.loc[r,'ch3_vol']/metadata.loc[r,'ch3_area_max'] #spread is a proxy for height in Fish et al
        
        metadata.loc[r,'ch1_height'] = sum(np.array(ch1_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        metadata.loc[r,'ch2_height'] = sum(np.array(ch2_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        metadata.loc[r,'ch3_height'] = sum(np.array(ch3_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        
        metadata.loc[r,'ch1_height_0'] = sum(np.array(ch1_areas) > (0)) #max of stack count when stack count has at least 1 pixel
        metadata.loc[r,'ch2_height_0'] = sum(np.array(ch2_areas) > (0)) #max of stack count when stack count has at least 1 pixel
        metadata.loc[r,'ch3_height_0'] = sum(np.array(ch3_areas) > (0)) #max of stack count when stack count has at least 1 pixel

metadata11= metadata
    


# ### Batch 12

# In[71]:


# start building the sample processing loops

metadata= metadata12 #need to parse out because computer crashes if try to run all files at once

#to get all images in file folder
dir = '/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/CLSM_biofilm_python/Hannah042220/'

metadata["stack_count_total"] = ""
metadata["ch1_area_max"] = ""
metadata["ch2_area_max"] = ""
metadata["ch3_area_max"] = ""
metadata["total_area_max"] = ""

metadata["ch1_vol"] = ""
metadata["ch2_vol"] = ""
metadata["ch3_vol"] = ""
metadata["EPS_vol"] = ""
metadata["total_vol"] = ""
metadata["ch2v1_vol"] = ""
metadata["ch2v3_vol"] = ""


metadata['ch1_spread']= ""
metadata['ch2_spread']= ""
metadata['ch3_spread']= ""

metadata['ch1_height']= ""
metadata['ch2_height']= ""
metadata['ch3_height']= ""

for row in metadata.itertuples(): #loop through each file
        r= row.Index
        file_name = (row.file_name)
        path= dir+ '/' + file_name
        m = bioformats.get_omexml_metadata(path)
        o = bioformats.OMEXML(m)
        stack_count_o = o.image().Pixels.SizeZ
        # stack_count_total.append(stack_count_o)
        metadata.loc[r,'stack_count_total'] = int(stack_count_o)
        
        if stack_count_o > 4 : #if file is tall enough, loop through each z stack in the file
            z_list = np.arange(4, stack_count_o) #cut off first 4 slices bc controls tended to fluoresce in first 4 slices
            
            image_list_ch1 = []
            image_list_ch2 = []
            image_list_ch3 = []
            image_list_EPS = []
            image_list_total = []
            image_list_2v1 = []
            image_list_2v3 = []
            
            ch1_areas = []
            ch2_areas = []
            ch3_areas = []
            areas_EPS = []
            areas_total = []
            areas_2v1 = []
            areas_2v3 = []

            for z in z_list: 
                #load image, filter, and threshold for each channel
                image_raw_ch1= bioformats.load_image(path, rescale = False, z = z, c = 0)
                image_binary_ch1= thresholding_function(image_raw_ch1)
                
                image_raw_ch2= bioformats.load_image(path, rescale = False, z = z, c = 1)
                image_binary_ch2= thresholding_function(image_raw_ch2)
                
                image_raw_ch3= bioformats.load_image(path, rescale = False, z = z, c = 2)
                image_binary_ch3= thresholding_function(image_raw_ch3)
                
                #combine channels that we want to combine
                image_binary_EPS = np.logical_or(image_binary_ch1, image_binary_ch3) #combine ch1 and ch3 to see EPS
                image_binary_total = np.logical_or(image_binary_EPS, image_binary_ch2) #combine ch1, ch2, and ch3 to capture total biovolume
                image_binary_2v1 = np.logical_and(image_binary_ch1, image_binary_ch2) #look at percent overlap of ch1 with ch2 (just relative biovolume)
                image_binary_2v3 = np.logical_and(image_binary_ch3, image_binary_ch2) #look at percent overlap of ch3 with ch2 (just relative biovolume)
                
                #calculate parameters 
                area_ch1 = area_function(image_binary_ch1)
                area_ch2 = area_function(image_binary_ch2)
                area_ch3 = area_function(image_binary_ch3)
                area_EPS = area_function(image_binary_EPS)
                area_total = area_function(image_binary_total)
                area_2v1 = area_function(image_binary_2v1)
                area_2v3 = area_function(image_binary_2v3)
                
                #append parameters to lists if needed for whole image
                # image_list_ch1.append(image_binary_ch1)
                # image_list_ch2.append(image_binary_ch2)
                # image_list_ch3.append(image_binary_ch3)
                # image_list_EPS.append(image_binary_EPS)
                # image_list_total.append(image_binary_total)
                # image_list_2v1.append(image_binary_2v1)
                # image_list_2v3.append(image_binary_2v3)
                
                ch1_areas.append(area_ch1)
                ch2_areas.append(area_ch2)
                ch3_areas.append(area_ch3)
                areas_EPS.append(area_EPS)
                areas_total.append(area_total)
                areas_2v1.append(area_2v1)
                areas_2v3.append(area_2v3)
            
        else:
            ch1_areas = [0]
            ch2_areas = [0]
            ch3_areas = [0]
            areas_EPS = [0]
            areas_total = [0]
            areas_2v1 = [0]
            areas_2v3 = [0]
        
        metadata.loc[r,'ch1_area_max'] = max(ch1_areas) #max area of biofilm in ch1
        metadata.loc[r,'ch2_area_max'] = max(ch2_areas) #max area of biofilm in ch2
        metadata.loc[r,'ch3_area_max'] = max(ch3_areas) #max area of biofilm in ch3
        metadata.loc[r,'total_area_max'] = max(areas_total) #max area of biofilm in combined channels
       
        
        metadata.loc[r,'ch1_vol'] = sum(ch1_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2_vol'] = sum(ch2_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch3_vol'] = sum(ch3_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'EPS_vol'] = sum(areas_EPS) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'total_vol'] = sum(areas_total) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2v1_vol'] = sum(areas_2v1) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2v3_vol'] = sum(areas_2v3) *1 #volume calc, each stack is 1 um thick
        
        if metadata.loc[r,'ch1_area_max'] != 0:
            metadata.loc[r,'ch1_spread'] = metadata.loc[r,'ch1_vol']/metadata.loc[r,'ch1_area_max'] #spread is a proxy for height in Fish et al
        if metadata.loc[r,'ch2_area_max'] != 0:
            metadata.loc[r,'ch2_spread'] = metadata.loc[r,'ch2_vol']/metadata.loc[r,'ch2_area_max'] #spread is a proxy for height in Fish et al
        if metadata.loc[r,'ch3_area_max'] != 0:
            metadata.loc[r,'ch3_spread'] = metadata.loc[r,'ch3_vol']/metadata.loc[r,'ch3_area_max'] #spread is a proxy for height in Fish et al
        
        metadata.loc[r,'ch1_height'] = sum(np.array(ch1_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        metadata.loc[r,'ch2_height'] = sum(np.array(ch2_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        metadata.loc[r,'ch3_height'] = sum(np.array(ch3_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        
        metadata.loc[r,'ch1_height_0'] = sum(np.array(ch1_areas) > (0)) #max of stack count when stack count has at least 1 pixel
        metadata.loc[r,'ch2_height_0'] = sum(np.array(ch2_areas) > (0)) #max of stack count when stack count has at least 1 pixel
        metadata.loc[r,'ch3_height_0'] = sum(np.array(ch3_areas) > (0)) #max of stack count when stack count has at least 1 pixel


metadata12= metadata
    


# ### Batch 1

# In[72]:


# start building the sample processing loops

metadata= metadata1 #need to parse out because computer crashes if try to run all files at once

#to get all images in file folder
dir = '/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/CLSM_biofilm_python/'

metadata["stack_count_total"] = ""
metadata["ch1_area_max"] = ""
metadata["ch2_area_max"] = ""
metadata["ch3_area_max"] = ""
metadata["total_area_max"] = ""

metadata["ch1_vol"] = ""
metadata["ch2_vol"] = ""
metadata["ch3_vol"] = ""
metadata["EPS_vol"] = ""
metadata["total_vol"] = ""
metadata["ch2v1_vol"] = ""
metadata["ch2v3_vol"] = ""


metadata['ch1_spread']= ""
metadata['ch2_spread']= ""
metadata['ch3_spread']= ""

metadata['ch1_height']= ""
metadata['ch2_height']= ""
metadata['ch3_height']= ""

for row in metadata.itertuples(): #loop through each file
        r= row.Index
        file_name = (row.file_name)
        path= dir+ '/' + file_name
        m = bioformats.get_omexml_metadata(path)
        o = bioformats.OMEXML(m)
        stack_count_o = o.image().Pixels.SizeZ
        # stack_count_total.append(stack_count_o)
        metadata.loc[r,'stack_count_total'] = int(stack_count_o)
        
        if stack_count_o > 4 : #if file is tall enough, loop through each z stack in the file
            z_list = np.arange(4, stack_count_o) #cut off first 4 slices bc controls tended to fluoresce in first 4 slices
            
            image_list_ch1 = []
            image_list_ch2 = []
            image_list_ch3 = []
            image_list_EPS = []
            image_list_total = []
            image_list_2v1 = []
            image_list_2v3 = []
            
            ch1_areas = []
            ch2_areas = []
            ch3_areas = []
            areas_EPS = []
            areas_total = []
            areas_2v1 = []
            areas_2v3 = []

            for z in z_list: 
                #load image, filter, and threshold for each channel
                image_raw_ch1= bioformats.load_image(path, rescale = False, z = z, c = 0)
                image_binary_ch1= thresholding_function(image_raw_ch1)
                
                image_raw_ch2= bioformats.load_image(path, rescale = False, z = z, c = 1)
                image_binary_ch2= thresholding_function(image_raw_ch2)
                
                image_raw_ch3= bioformats.load_image(path, rescale = False, z = z, c = 2)
                image_binary_ch3= thresholding_function(image_raw_ch3)
                
                #combine channels that we want to combine
                image_binary_EPS = np.logical_or(image_binary_ch1, image_binary_ch3) #combine ch1 and ch3 to see EPS
                image_binary_total = np.logical_or(image_binary_EPS, image_binary_ch2) #combine ch1, ch2, and ch3 to capture total biovolume
                image_binary_2v1 = np.logical_and(image_binary_ch1, image_binary_ch2) #look at percent overlap of ch1 with ch2 (just relative biovolume)
                image_binary_2v3 = np.logical_and(image_binary_ch3, image_binary_ch2) #look at percent overlap of ch3 with ch2 (just relative biovolume)
                
                #calculate parameters 
                area_ch1 = area_function(image_binary_ch1)
                area_ch2 = area_function(image_binary_ch2)
                area_ch3 = area_function(image_binary_ch3)
                area_EPS = area_function(image_binary_EPS)
                area_total = area_function(image_binary_total)
                area_2v1 = area_function(image_binary_2v1)
                area_2v3 = area_function(image_binary_2v3)
                
                #append parameters to lists if needed for whole image
                # image_list_ch1.append(image_binary_ch1)
                # image_list_ch2.append(image_binary_ch2)
                # image_list_ch3.append(image_binary_ch3)
                # image_list_EPS.append(image_binary_EPS)
                # image_list_total.append(image_binary_total)
                # image_list_2v1.append(image_binary_2v1)
                # image_list_2v3.append(image_binary_2v3)
                
                ch1_areas.append(area_ch1)
                ch2_areas.append(area_ch2)
                ch3_areas.append(area_ch3)
                areas_EPS.append(area_EPS)
                areas_total.append(area_total)
                areas_2v1.append(area_2v1)
                areas_2v3.append(area_2v3)
            
        else:
            ch1_areas = [0]
            ch2_areas = [0]
            ch3_areas = [0]
            areas_EPS = [0]
            areas_total = [0]
            areas_2v1 = [0]
            areas_2v3 = [0]
        
        metadata.loc[r,'ch1_area_max'] = max(ch1_areas) #max area of biofilm in ch1
        metadata.loc[r,'ch2_area_max'] = max(ch2_areas) #max area of biofilm in ch2
        metadata.loc[r,'ch3_area_max'] = max(ch3_areas) #max area of biofilm in ch3
        metadata.loc[r,'total_area_max'] = max(areas_total) #max area of biofilm in combined channels
       
        
        metadata.loc[r,'ch1_vol'] = sum(ch1_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2_vol'] = sum(ch2_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch3_vol'] = sum(ch3_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'EPS_vol'] = sum(areas_EPS) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'total_vol'] = sum(areas_total) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2v1_vol'] = sum(areas_2v1) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2v3_vol'] = sum(areas_2v3) *1 #volume calc, each stack is 1 um thick
        
        if metadata.loc[r,'ch1_area_max'] != 0:
            metadata.loc[r,'ch1_spread'] = metadata.loc[r,'ch1_vol']/metadata.loc[r,'ch1_area_max'] #spread is a proxy for height in Fish et al
        if metadata.loc[r,'ch2_area_max'] != 0:
            metadata.loc[r,'ch2_spread'] = metadata.loc[r,'ch2_vol']/metadata.loc[r,'ch2_area_max'] #spread is a proxy for height in Fish et al
        if metadata.loc[r,'ch3_area_max'] != 0:
            metadata.loc[r,'ch3_spread'] = metadata.loc[r,'ch3_vol']/metadata.loc[r,'ch3_area_max'] #spread is a proxy for height in Fish et al
        
        metadata.loc[r,'ch1_height'] = sum(np.array(ch1_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        metadata.loc[r,'ch2_height'] = sum(np.array(ch2_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        metadata.loc[r,'ch3_height'] = sum(np.array(ch3_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        
        metadata.loc[r,'ch1_height_0'] = sum(np.array(ch1_areas) > (0)) #max of stack count when stack count has at least 1 pixel
        metadata.loc[r,'ch2_height_0'] = sum(np.array(ch2_areas) > (0)) #max of stack count when stack count has at least 1 pixel
        metadata.loc[r,'ch3_height_0'] = sum(np.array(ch3_areas) > (0)) #max of stack count when stack count has at least 1 pixel


metadata1= metadata
    


# ### Batch 2

# In[74]:


# start building the sample processing loops

metadata= metadata2 #need to parse out because computer crashes if try to run all files at once

#to get all images in file folder
dir = '/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/CLSM_biofilm_python/'

metadata["stack_count_total"] = ""
metadata["ch1_area_max"] = ""
metadata["ch2_area_max"] = ""
metadata["ch3_area_max"] = ""
metadata["total_area_max"] = ""

metadata["ch1_vol"] = ""
metadata["ch2_vol"] = ""
metadata["ch3_vol"] = ""
metadata["EPS_vol"] = ""
metadata["total_vol"] = ""
metadata["ch2v1_vol"] = ""
metadata["ch2v3_vol"] = ""


metadata['ch1_spread']= ""
metadata['ch2_spread']= ""
metadata['ch3_spread']= ""

metadata['ch1_height']= ""
metadata['ch2_height']= ""
metadata['ch3_height']= ""

for row in metadata.itertuples(): #loop through each file
        r= row.Index
        file_name = (row.file_name)
        path= dir+ '/' + file_name
        m = bioformats.get_omexml_metadata(path)
        o = bioformats.OMEXML(m)
        stack_count_o = o.image().Pixels.SizeZ
        # stack_count_total.append(stack_count_o)
        metadata.loc[r,'stack_count_total'] = int(stack_count_o)
        
        if stack_count_o > 4 : #if file is tall enough, loop through each z stack in the file
            z_list = np.arange(4, stack_count_o) #cut off first 4 slices bc controls tended to fluoresce in first 4 slices
            
            image_list_ch1 = []
            image_list_ch2 = []
            image_list_ch3 = []
            image_list_EPS = []
            image_list_total = []
            image_list_2v1 = []
            image_list_2v3 = []
            
            ch1_areas = []
            ch2_areas = []
            ch3_areas = []
            areas_EPS = []
            areas_total = []
            areas_2v1 = []
            areas_2v3 = []

            for z in z_list: 
                #load image, filter, and threshold for each channel
                image_raw_ch1= bioformats.load_image(path, rescale = False, z = z, c = 0)
                image_binary_ch1= thresholding_function(image_raw_ch1)
                
                image_raw_ch2= bioformats.load_image(path, rescale = False, z = z, c = 1)
                image_binary_ch2= thresholding_function(image_raw_ch2)
                
                image_raw_ch3= bioformats.load_image(path, rescale = False, z = z, c = 2)
                image_binary_ch3= thresholding_function(image_raw_ch3)
                
                #combine channels that we want to combine
                image_binary_EPS = np.logical_or(image_binary_ch1, image_binary_ch3) #combine ch1 and ch3 to see EPS
                image_binary_total = np.logical_or(image_binary_EPS, image_binary_ch2) #combine ch1, ch2, and ch3 to capture total biovolume
                image_binary_2v1 = np.logical_and(image_binary_ch1, image_binary_ch2) #look at percent overlap of ch1 with ch2 (just relative biovolume)
                image_binary_2v3 = np.logical_and(image_binary_ch3, image_binary_ch2) #look at percent overlap of ch3 with ch2 (just relative biovolume)
                
                #calculate parameters 
                area_ch1 = area_function(image_binary_ch1)
                area_ch2 = area_function(image_binary_ch2)
                area_ch3 = area_function(image_binary_ch3)
                area_EPS = area_function(image_binary_EPS)
                area_total = area_function(image_binary_total)
                area_2v1 = area_function(image_binary_2v1)
                area_2v3 = area_function(image_binary_2v3)
                
                #append parameters to lists if needed for whole image
                # image_list_ch1.append(image_binary_ch1)
                # image_list_ch2.append(image_binary_ch2)
                # image_list_ch3.append(image_binary_ch3)
                # image_list_EPS.append(image_binary_EPS)
                # image_list_total.append(image_binary_total)
                # image_list_2v1.append(image_binary_2v1)
                # image_list_2v3.append(image_binary_2v3)
                
                ch1_areas.append(area_ch1)
                ch2_areas.append(area_ch2)
                ch3_areas.append(area_ch3)
                areas_EPS.append(area_EPS)
                areas_total.append(area_total)
                areas_2v1.append(area_2v1)
                areas_2v3.append(area_2v3)
            
        else:
            ch1_areas = [0]
            ch2_areas = [0]
            ch3_areas = [0]
            areas_EPS = [0]
            areas_total = [0]
            areas_2v1 = [0]
            areas_2v3 = [0]
        
        metadata.loc[r,'ch1_area_max'] = max(ch1_areas) #max area of biofilm in ch1
        metadata.loc[r,'ch2_area_max'] = max(ch2_areas) #max area of biofilm in ch2
        metadata.loc[r,'ch3_area_max'] = max(ch3_areas) #max area of biofilm in ch3
        metadata.loc[r,'total_area_max'] = max(areas_total) #max area of biofilm in combined channels
       
        
        metadata.loc[r,'ch1_vol'] = sum(ch1_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2_vol'] = sum(ch2_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch3_vol'] = sum(ch3_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'EPS_vol'] = sum(areas_EPS) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'total_vol'] = sum(areas_total) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2v1_vol'] = sum(areas_2v1) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2v3_vol'] = sum(areas_2v3) *1 #volume calc, each stack is 1 um thick
        
        if metadata.loc[r,'ch1_area_max'] != 0:
            metadata.loc[r,'ch1_spread'] = metadata.loc[r,'ch1_vol']/metadata.loc[r,'ch1_area_max'] #spread is a proxy for height in Fish et al
        if metadata.loc[r,'ch2_area_max'] != 0:
            metadata.loc[r,'ch2_spread'] = metadata.loc[r,'ch2_vol']/metadata.loc[r,'ch2_area_max'] #spread is a proxy for height in Fish et al
        if metadata.loc[r,'ch3_area_max'] != 0:
            metadata.loc[r,'ch3_spread'] = metadata.loc[r,'ch3_vol']/metadata.loc[r,'ch3_area_max'] #spread is a proxy for height in Fish et al
        
        metadata.loc[r,'ch1_height'] = sum(np.array(ch1_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        metadata.loc[r,'ch2_height'] = sum(np.array(ch2_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        metadata.loc[r,'ch3_height'] = sum(np.array(ch3_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        
        metadata.loc[r,'ch1_height_0'] = sum(np.array(ch1_areas) > (0)) #max of stack count when stack count has at least 1 pixel
        metadata.loc[r,'ch2_height_0'] = sum(np.array(ch2_areas) > (0)) #max of stack count when stack count has at least 1 pixel
        metadata.loc[r,'ch3_height_0'] = sum(np.array(ch3_areas) > (0)) #max of stack count when stack count has at least 1 pixel


metadata2= metadata
    


# ### Batch 3

# In[ ]:


# start building the sample processing loops

metadata= metadata3 #need to parse out because computer crashes if try to run all files at once

#to get all images in file folder
dir = '/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/CLSM_biofilm_python/'

metadata["stack_count_total"] = ""
metadata["ch1_area_max"] = ""
metadata["ch2_area_max"] = ""
metadata["ch3_area_max"] = ""
metadata["total_area_max"] = ""

metadata["ch1_vol"] = ""
metadata["ch2_vol"] = ""
metadata["ch3_vol"] = ""
metadata["EPS_vol"] = ""
metadata["total_vol"] = ""
metadata["ch2v1_vol"] = ""
metadata["ch2v3_vol"] = ""


metadata['ch1_spread']= ""
metadata['ch2_spread']= ""
metadata['ch3_spread']= ""

metadata['ch1_height']= ""
metadata['ch2_height']= ""
metadata['ch3_height']= ""

for row in metadata.itertuples(): #loop through each file
        r= row.Index
        file_name = (row.file_name)
        path= dir+ '/' + file_name
        m = bioformats.get_omexml_metadata(path)
        o = bioformats.OMEXML(m)
        stack_count_o = o.image().Pixels.SizeZ
        # stack_count_total.append(stack_count_o)
        metadata.loc[r,'stack_count_total'] = int(stack_count_o)
        
        if stack_count_o > 4 : #if file is tall enough, loop through each z stack in the file
            z_list = np.arange(4, stack_count_o) #cut off first 4 slices bc controls tended to fluoresce in first 4 slices
            
            image_list_ch1 = []
            image_list_ch2 = []
            image_list_ch3 = []
            image_list_EPS = []
            image_list_total = []
            image_list_2v1 = []
            image_list_2v3 = []
            
            ch1_areas = []
            ch2_areas = []
            ch3_areas = []
            areas_EPS = []
            areas_total = []
            areas_2v1 = []
            areas_2v3 = []

            for z in z_list: 
                #load image, filter, and threshold for each channel
                image_raw_ch1= bioformats.load_image(path, rescale = False, z = z, c = 0)
                image_binary_ch1= thresholding_function(image_raw_ch1)
                
                image_raw_ch2= bioformats.load_image(path, rescale = False, z = z, c = 1)
                image_binary_ch2= thresholding_function(image_raw_ch2)
                
                image_raw_ch3= bioformats.load_image(path, rescale = False, z = z, c = 2)
                image_binary_ch3= thresholding_function(image_raw_ch3)
                
                #combine channels that we want to combine
                image_binary_EPS = np.logical_or(image_binary_ch1, image_binary_ch3) #combine ch1 and ch3 to see EPS
                image_binary_total = np.logical_or(image_binary_EPS, image_binary_ch2) #combine ch1, ch2, and ch3 to capture total biovolume
                image_binary_2v1 = np.logical_and(image_binary_ch1, image_binary_ch2) #look at percent overlap of ch1 with ch2 (just relative biovolume)
                image_binary_2v3 = np.logical_and(image_binary_ch3, image_binary_ch2) #look at percent overlap of ch3 with ch2 (just relative biovolume)
                
                #calculate parameters 
                area_ch1 = area_function(image_binary_ch1)
                area_ch2 = area_function(image_binary_ch2)
                area_ch3 = area_function(image_binary_ch3)
                area_EPS = area_function(image_binary_EPS)
                area_total = area_function(image_binary_total)
                area_2v1 = area_function(image_binary_2v1)
                area_2v3 = area_function(image_binary_2v3)
                
                #append parameters to lists if needed for whole image
                # image_list_ch1.append(image_binary_ch1)
                # image_list_ch2.append(image_binary_ch2)
                # image_list_ch3.append(image_binary_ch3)
                # image_list_EPS.append(image_binary_EPS)
                # image_list_total.append(image_binary_total)
                # image_list_2v1.append(image_binary_2v1)
                # image_list_2v3.append(image_binary_2v3)
                
                ch1_areas.append(area_ch1)
                ch2_areas.append(area_ch2)
                ch3_areas.append(area_ch3)
                areas_EPS.append(area_EPS)
                areas_total.append(area_total)
                areas_2v1.append(area_2v1)
                areas_2v3.append(area_2v3)
            
        else:
            ch1_areas = [0]
            ch2_areas = [0]
            ch3_areas = [0]
            areas_EPS = [0]
            areas_total = [0]
            areas_2v1 = [0]
            areas_2v3 = [0]
        
        metadata.loc[r,'ch1_area_max'] = max(ch1_areas) #max area of biofilm in ch1
        metadata.loc[r,'ch2_area_max'] = max(ch2_areas) #max area of biofilm in ch2
        metadata.loc[r,'ch3_area_max'] = max(ch3_areas) #max area of biofilm in ch3
        metadata.loc[r,'total_area_max'] = max(areas_total) #max area of biofilm in combined channels
       
        
        metadata.loc[r,'ch1_vol'] = sum(ch1_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2_vol'] = sum(ch2_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch3_vol'] = sum(ch3_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'EPS_vol'] = sum(areas_EPS) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'total_vol'] = sum(areas_total) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2v1_vol'] = sum(areas_2v1) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2v3_vol'] = sum(areas_2v3) *1 #volume calc, each stack is 1 um thick
        
        if metadata.loc[r,'ch1_area_max'] != 0:
            metadata.loc[r,'ch1_spread'] = metadata.loc[r,'ch1_vol']/metadata.loc[r,'ch1_area_max'] #spread is a proxy for height in Fish et al
        if metadata.loc[r,'ch2_area_max'] != 0:
            metadata.loc[r,'ch2_spread'] = metadata.loc[r,'ch2_vol']/metadata.loc[r,'ch2_area_max'] #spread is a proxy for height in Fish et al
        if metadata.loc[r,'ch3_area_max'] != 0:
            metadata.loc[r,'ch3_spread'] = metadata.loc[r,'ch3_vol']/metadata.loc[r,'ch3_area_max'] #spread is a proxy for height in Fish et al
        
        metadata.loc[r,'ch1_height'] = sum(np.array(ch1_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        metadata.loc[r,'ch2_height'] = sum(np.array(ch2_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        metadata.loc[r,'ch3_height'] = sum(np.array(ch3_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        
        metadata.loc[r,'ch1_height_0'] = sum(np.array(ch1_areas) > (0)) #max of stack count when stack count has at least 1 pixel
        metadata.loc[r,'ch2_height_0'] = sum(np.array(ch2_areas) > (0)) #max of stack count when stack count has at least 1 pixel
        metadata.loc[r,'ch3_height_0'] = sum(np.array(ch3_areas) > (0)) #max of stack count when stack count has at least 1 pixel


metadata3= metadata
    


# ### Batch 4

# In[ ]:


# start building the sample processing loops

metadata= metadata4 #need to parse out because computer crashes if try to run all files at once

#to get all images in file folder
dir = '/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/CLSM_biofilm_python/'

metadata["stack_count_total"] = ""
metadata["ch1_area_max"] = ""
metadata["ch2_area_max"] = ""
metadata["ch3_area_max"] = ""
metadata["total_area_max"] = ""

metadata["ch1_vol"] = ""
metadata["ch2_vol"] = ""
metadata["ch3_vol"] = ""
metadata["EPS_vol"] = ""
metadata["total_vol"] = ""
metadata["ch2v1_vol"] = ""
metadata["ch2v3_vol"] = ""

metadata['ch1_spread']= ""
metadata['ch2_spread']= ""
metadata['ch3_spread']= ""

metadata['ch1_height']= ""
metadata['ch2_height']= ""
metadata['ch3_height']= ""

for row in metadata.itertuples(): #loop through each file
        r= row.Index
        file_name = (row.file_name)
        path= dir+ '/' + file_name
        m = bioformats.get_omexml_metadata(path)
        o = bioformats.OMEXML(m)
        stack_count_o = o.image().Pixels.SizeZ
        # stack_count_total.append(stack_count_o)
        metadata.loc[r,'stack_count_total'] = int(stack_count_o)
        
        if stack_count_o > 4 : #if file is tall enough, loop through each z stack in the file
            z_list = np.arange(4, stack_count_o) #cut off first 4 slices bc controls tended to fluoresce in first 4 slices
            
            image_list_ch1 = []
            image_list_ch2 = []
            image_list_ch3 = []
            image_list_EPS = []
            image_list_total = []
            image_list_2v1 = []
            image_list_2v3 = []
            
            ch1_areas = []
            ch2_areas = []
            ch3_areas = []
            areas_EPS = []
            areas_total = []
            areas_2v1 = []
            areas_2v3 = []

            for z in z_list: 
                #load image, filter, and threshold for each channel
                image_raw_ch1= bioformats.load_image(path, rescale = False, z = z, c = 0)
                image_binary_ch1= thresholding_function(image_raw_ch1)
                
                image_raw_ch2= bioformats.load_image(path, rescale = False, z = z, c = 1)
                image_binary_ch2= thresholding_function(image_raw_ch2)
                
                image_raw_ch3= bioformats.load_image(path, rescale = False, z = z, c = 2)
                image_binary_ch3= thresholding_function(image_raw_ch3)
                
                #combine channels that we want to combine
                image_binary_EPS = np.logical_or(image_binary_ch1, image_binary_ch3) #combine ch1 and ch3 to see EPS
                image_binary_total = np.logical_or(image_binary_EPS, image_binary_ch2) #combine ch1, ch2, and ch3 to capture total biovolume
                image_binary_2v1 = np.logical_and(image_binary_ch1, image_binary_ch2) #look at percent overlap of ch1 with ch2 (just relative biovolume)
                image_binary_2v3 = np.logical_and(image_binary_ch3, image_binary_ch2) #look at percent overlap of ch3 with ch2 (just relative biovolume)
                
                #calculate parameters 
                area_ch1 = area_function(image_binary_ch1)
                area_ch2 = area_function(image_binary_ch2)
                area_ch3 = area_function(image_binary_ch3)
                area_EPS = area_function(image_binary_EPS)
                area_total = area_function(image_binary_total)
                area_2v1 = area_function(image_binary_2v1)
                area_2v3 = area_function(image_binary_2v3)
                
                #append parameters to lists if needed for whole image
                # image_list_ch1.append(image_binary_ch1)
                # image_list_ch2.append(image_binary_ch2)
                # image_list_ch3.append(image_binary_ch3)
                # image_list_EPS.append(image_binary_EPS)
                # image_list_total.append(image_binary_total)
                # image_list_2v1.append(image_binary_2v1)
                # image_list_2v3.append(image_binary_2v3)
                
                ch1_areas.append(area_ch1)
                ch2_areas.append(area_ch2)
                ch3_areas.append(area_ch3)
                areas_EPS.append(area_EPS)
                areas_total.append(area_total)
                areas_2v1.append(area_2v1)
                areas_2v3.append(area_2v3)
            
        else:
            ch1_areas = [0]
            ch2_areas = [0]
            ch3_areas = [0]
            areas_EPS = [0]
            areas_total = [0]
            areas_2v1 = [0]
            areas_2v3 = [0]
        
        metadata.loc[r,'ch1_area_max'] = max(ch1_areas) #max area of biofilm in ch1
        metadata.loc[r,'ch2_area_max'] = max(ch2_areas) #max area of biofilm in ch2
        metadata.loc[r,'ch3_area_max'] = max(ch3_areas) #max area of biofilm in ch3
        metadata.loc[r,'total_area_max'] = max(areas_total) #max area of biofilm in combined channels
       
        metadata.loc[r,'ch1_vol'] = sum(ch1_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2_vol'] = sum(ch2_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch3_vol'] = sum(ch3_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'EPS_vol'] = sum(areas_EPS) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'total_vol'] = sum(areas_total) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2v1_vol'] = sum(areas_2v1) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2v3_vol'] = sum(areas_2v3) *1 #volume calc, each stack is 1 um thick
        
        if metadata.loc[r,'ch1_area_max'] != 0:
            metadata.loc[r,'ch1_spread'] = metadata.loc[r,'ch1_vol']/metadata.loc[r,'ch1_area_max'] #spread is a proxy for height in Fish et al
        if metadata.loc[r,'ch2_area_max'] != 0:
            metadata.loc[r,'ch2_spread'] = metadata.loc[r,'ch2_vol']/metadata.loc[r,'ch2_area_max'] #spread is a proxy for height in Fish et al
        if metadata.loc[r,'ch3_area_max'] != 0:
            metadata.loc[r,'ch3_spread'] = metadata.loc[r,'ch3_vol']/metadata.loc[r,'ch3_area_max'] #spread is a proxy for height in Fish et al
        
        metadata.loc[r,'ch1_height'] = sum(np.array(ch1_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        metadata.loc[r,'ch2_height'] = sum(np.array(ch2_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        metadata.loc[r,'ch3_height'] = sum(np.array(ch3_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        
        metadata.loc[r,'ch1_height_0'] = sum(np.array(ch1_areas) > (0)) #max of stack count when stack count has at least 1 pixel
        metadata.loc[r,'ch2_height_0'] = sum(np.array(ch2_areas) > (0)) #max of stack count when stack count has at least 1 pixel
        metadata.loc[r,'ch3_height_0'] = sum(np.array(ch3_areas) > (0)) #max of stack count when stack count has at least 1 pixel


metadata4= metadata
    


# In[79]:


metadata4


# ### Batch 5

# In[ ]:


# start building the sample processing loops

metadata= metadata5 #need to parse out because computer crashes if try to run all files at once

#to get all images in file folder
dir = '/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/CLSM_biofilm_python/'

metadata["stack_count_total"] = ""
metadata["ch1_area_max"] = ""
metadata["ch2_area_max"] = ""
metadata["ch3_area_max"] = ""
metadata["total_area_max"] = ""

metadata["ch1_vol"] = ""
metadata["ch2_vol"] = ""
metadata["ch3_vol"] = ""
metadata["EPS_vol"] = ""
metadata["total_vol"] = ""
metadata["ch2v1_vol"] = ""
metadata["ch2v3_vol"] = ""


metadata['ch1_spread']= ""
metadata['ch2_spread']= ""
metadata['ch3_spread']= ""

metadata['ch1_height']= ""
metadata['ch2_height']= ""
metadata['ch3_height']= ""

for row in metadata.itertuples(): #loop through each file
        r= row.Index
        file_name = (row.file_name)
        path= dir+ '/' + file_name
        m = bioformats.get_omexml_metadata(path)
        o = bioformats.OMEXML(m)
        stack_count_o = o.image().Pixels.SizeZ
        # stack_count_total.append(stack_count_o)
        metadata.loc[r,'stack_count_total'] = int(stack_count_o)
        
        if stack_count_o > 4 : #if file is tall enough, loop through each z stack in the file
            z_list = np.arange(4, stack_count_o) #cut off first 4 slices bc controls tended to fluoresce in first 4 slices
            
            image_list_ch1 = []
            image_list_ch2 = []
            image_list_ch3 = []
            image_list_EPS = []
            image_list_total = []
            image_list_2v1 = []
            image_list_2v3 = []
            
            ch1_areas = []
            ch2_areas = []
            ch3_areas = []
            areas_EPS = []
            areas_total = []
            areas_2v1 = []
            areas_2v3 = []

            for z in z_list: 
                #load image, filter, and threshold for each channel
                image_raw_ch1= bioformats.load_image(path, rescale = False, z = z, c = 0)
                image_binary_ch1= thresholding_function(image_raw_ch1)
                
                image_raw_ch2= bioformats.load_image(path, rescale = False, z = z, c = 1)
                image_binary_ch2= thresholding_function(image_raw_ch2)
                
                image_raw_ch3= bioformats.load_image(path, rescale = False, z = z, c = 2)
                image_binary_ch3= thresholding_function(image_raw_ch3)
                
                #combine channels that we want to combine
                image_binary_EPS = np.logical_or(image_binary_ch1, image_binary_ch3) #combine ch1 and ch3 to see EPS
                image_binary_total = np.logical_or(image_binary_EPS, image_binary_ch2) #combine ch1, ch2, and ch3 to capture total biovolume
                image_binary_2v1 = np.logical_and(image_binary_ch1, image_binary_ch2) #look at percent overlap of ch1 with ch2 (just relative biovolume)
                image_binary_2v3 = np.logical_and(image_binary_ch3, image_binary_ch2) #look at percent overlap of ch3 with ch2 (just relative biovolume)
                
                #calculate parameters 
                area_ch1 = area_function(image_binary_ch1)
                area_ch2 = area_function(image_binary_ch2)
                area_ch3 = area_function(image_binary_ch3)
                area_EPS = area_function(image_binary_EPS)
                area_total = area_function(image_binary_total)
                area_2v1 = area_function(image_binary_2v1)
                area_2v3 = area_function(image_binary_2v3)
                
                #append parameters to lists if needed for whole image
                # image_list_ch1.append(image_binary_ch1)
                # image_list_ch2.append(image_binary_ch2)
                # image_list_ch3.append(image_binary_ch3)
                # image_list_EPS.append(image_binary_EPS)
                # image_list_total.append(image_binary_total)
                # image_list_2v1.append(image_binary_2v1)
                # image_list_2v3.append(image_binary_2v3)
                
                ch1_areas.append(area_ch1)
                ch2_areas.append(area_ch2)
                ch3_areas.append(area_ch3)
                areas_EPS.append(area_EPS)
                areas_total.append(area_total)
                areas_2v1.append(area_2v1)
                areas_2v3.append(area_2v3)
            
        else:
            ch1_areas = [0]
            ch2_areas = [0]
            ch3_areas = [0]
            areas_EPS = [0]
            areas_total = [0]
            areas_2v1 = [0]
            areas_2v3 = [0]
        
        metadata.loc[r,'ch1_area_max'] = max(ch1_areas) #max area of biofilm in ch1
        metadata.loc[r,'ch2_area_max'] = max(ch2_areas) #max area of biofilm in ch2
        metadata.loc[r,'ch3_area_max'] = max(ch3_areas) #max area of biofilm in ch3
        metadata.loc[r,'total_area_max'] = max(areas_total) #max area of biofilm in combined channels
       
        metadata.loc[r,'ch1_vol'] = sum(ch1_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2_vol'] = sum(ch2_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch3_vol'] = sum(ch3_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'EPS_vol'] = sum(areas_EPS) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'total_vol'] = sum(areas_total) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2v1_vol'] = sum(areas_2v1) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2v3_vol'] = sum(areas_2v3) *1 #volume calc, each stack is 1 um thick
        
        if metadata.loc[r,'ch1_area_max'] != 0:
            metadata.loc[r,'ch1_spread'] = metadata.loc[r,'ch1_vol']/metadata.loc[r,'ch1_area_max'] #spread is a proxy for height in Fish et al
        if metadata.loc[r,'ch2_area_max'] != 0:
            metadata.loc[r,'ch2_spread'] = metadata.loc[r,'ch2_vol']/metadata.loc[r,'ch2_area_max'] #spread is a proxy for height in Fish et al
        if metadata.loc[r,'ch3_area_max'] != 0:
            metadata.loc[r,'ch3_spread'] = metadata.loc[r,'ch3_vol']/metadata.loc[r,'ch3_area_max'] #spread is a proxy for height in Fish et al
        
        metadata.loc[r,'ch1_height'] = sum(np.array(ch1_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        metadata.loc[r,'ch2_height'] = sum(np.array(ch2_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        metadata.loc[r,'ch3_height'] = sum(np.array(ch3_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        
        metadata.loc[r,'ch1_height_0'] = sum(np.array(ch1_areas) > (0)) #max of stack count when stack count has at least 1 pixel
        metadata.loc[r,'ch2_height_0'] = sum(np.array(ch2_areas) > (0)) #max of stack count when stack count has at least 1 pixel
        metadata.loc[r,'ch3_height_0'] = sum(np.array(ch3_areas) > (0)) #max of stack count when stack count has at least 1 pixel

        

metadata5= metadata
    


# In[81]:


metadata5


# ### Batch 6

# In[82]:


# start building the sample processing loops

metadata= metadata6 #need to parse out because computer crashes if try to run all files at once

#to get all images in file folder
dir = '/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/CLSM_biofilm_python/'

metadata["stack_count_total"] = ""
metadata["ch1_area_max"] = ""
metadata["ch2_area_max"] = ""
metadata["ch3_area_max"] = ""
metadata["total_area_max"] = ""

metadata["ch1_vol"] = ""
metadata["ch2_vol"] = ""
metadata["ch3_vol"] = ""
metadata["EPS_vol"] = ""
metadata["total_vol"] = ""
metadata["ch2v1_vol"] = ""
metadata["ch2v3_vol"] = ""


metadata['ch1_spread']= ""
metadata['ch2_spread']= ""
metadata['ch3_spread']= ""

metadata['ch1_height']= ""
metadata['ch2_height']= ""
metadata['ch3_height']= ""

for row in metadata.itertuples(): #loop through each file
        r= row.Index
        file_name = (row.file_name)
        path= dir+ '/' + file_name
        m = bioformats.get_omexml_metadata(path)
        o = bioformats.OMEXML(m)
        stack_count_o = o.image().Pixels.SizeZ
        # stack_count_total.append(stack_count_o)
        metadata.loc[r,'stack_count_total'] = int(stack_count_o)
        
        if stack_count_o > 4 : #if file is tall enough, loop through each z stack in the file
            z_list = np.arange(4, stack_count_o) #cut off first 4 slices bc controls tended to fluoresce in first 4 slices
            
            image_list_ch1 = []
            image_list_ch2 = []
            image_list_ch3 = []
            image_list_EPS = []
            image_list_total = []
            image_list_2v1 = []
            image_list_2v3 = []
            
            ch1_areas = []
            ch2_areas = []
            ch3_areas = []
            areas_EPS = []
            areas_total = []
            areas_2v1 = []
            areas_2v3 = []

            for z in z_list: 
                #load image, filter, and threshold for each channel
                image_raw_ch1= bioformats.load_image(path, rescale = False, z = z, c = 0)
                image_binary_ch1= thresholding_function(image_raw_ch1)
                
                image_raw_ch2= bioformats.load_image(path, rescale = False, z = z, c = 1)
                image_binary_ch2= thresholding_function(image_raw_ch2)
                
                image_raw_ch3= bioformats.load_image(path, rescale = False, z = z, c = 2)
                image_binary_ch3= thresholding_function(image_raw_ch3)
                
                #combine channels that we want to combine
                image_binary_EPS = np.logical_or(image_binary_ch1, image_binary_ch3) #combine ch1 and ch3 to see EPS
                image_binary_total = np.logical_or(image_binary_EPS, image_binary_ch2) #combine ch1, ch2, and ch3 to capture total biovolume
                image_binary_2v1 = np.logical_and(image_binary_ch1, image_binary_ch2) #look at percent overlap of ch1 with ch2 (just relative biovolume)
                image_binary_2v3 = np.logical_and(image_binary_ch3, image_binary_ch2) #look at percent overlap of ch3 with ch2 (just relative biovolume)
                
                #calculate parameters 
                area_ch1 = area_function(image_binary_ch1)
                area_ch2 = area_function(image_binary_ch2)
                area_ch3 = area_function(image_binary_ch3)
                area_EPS = area_function(image_binary_EPS)
                area_total = area_function(image_binary_total)
                area_2v1 = area_function(image_binary_2v1)
                area_2v3 = area_function(image_binary_2v3)
                
                #append parameters to lists if needed for whole image
                # image_list_ch1.append(image_binary_ch1)
                # image_list_ch2.append(image_binary_ch2)
                # image_list_ch3.append(image_binary_ch3)
                # image_list_EPS.append(image_binary_EPS)
                # image_list_total.append(image_binary_total)
                # image_list_2v1.append(image_binary_2v1)
                # image_list_2v3.append(image_binary_2v3)
                
                ch1_areas.append(area_ch1)
                ch2_areas.append(area_ch2)
                ch3_areas.append(area_ch3)
                areas_EPS.append(area_EPS)
                areas_total.append(area_total)
                areas_2v1.append(area_2v1)
                areas_2v3.append(area_2v3)
            
        else:
            ch1_areas = [0]
            ch2_areas = [0]
            ch3_areas = [0]
            areas_EPS = [0]
            areas_total = [0]
            areas_2v1 = [0]
            areas_2v3 = [0]
        
        metadata.loc[r,'ch1_area_max'] = max(ch1_areas) #max area of biofilm in ch1
        metadata.loc[r,'ch2_area_max'] = max(ch2_areas) #max area of biofilm in ch2
        metadata.loc[r,'ch3_area_max'] = max(ch3_areas) #max area of biofilm in ch3
        metadata.loc[r,'total_area_max'] = max(areas_total) #max area of biofilm in combined channels
       
        metadata.loc[r,'ch1_vol'] = sum(ch1_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2_vol'] = sum(ch2_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch3_vol'] = sum(ch3_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'EPS_vol'] = sum(areas_EPS) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'total_vol'] = sum(areas_total) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2v1_vol'] = sum(areas_2v1) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2v3_vol'] = sum(areas_2v3) *1 #volume calc, each stack is 1 um thick
        
        if metadata.loc[r,'ch1_area_max'] != 0:
            metadata.loc[r,'ch1_spread'] = metadata.loc[r,'ch1_vol']/metadata.loc[r,'ch1_area_max'] #spread is a proxy for height in Fish et al
        if metadata.loc[r,'ch2_area_max'] != 0:
            metadata.loc[r,'ch2_spread'] = metadata.loc[r,'ch2_vol']/metadata.loc[r,'ch2_area_max'] #spread is a proxy for height in Fish et al
        if metadata.loc[r,'ch3_area_max'] != 0:
            metadata.loc[r,'ch3_spread'] = metadata.loc[r,'ch3_vol']/metadata.loc[r,'ch3_area_max'] #spread is a proxy for height in Fish et al
        
        metadata.loc[r,'ch1_height'] = sum(np.array(ch1_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        metadata.loc[r,'ch2_height'] = sum(np.array(ch2_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        metadata.loc[r,'ch3_height'] = sum(np.array(ch3_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        
        metadata.loc[r,'ch1_height_0'] = sum(np.array(ch1_areas) > (0)) #max of stack count when stack count has at least 1 pixel
        metadata.loc[r,'ch2_height_0'] = sum(np.array(ch2_areas) > (0)) #max of stack count when stack count has at least 1 pixel
        metadata.loc[r,'ch3_height_0'] = sum(np.array(ch3_areas) > (0)) #max of stack count when stack count has at least 1 pixel


metadata6= metadata
    


# In[83]:


metadata6


# ### Batch 7

# In[84]:


# start building the sample processing loops

metadata= metadata7 #need to parse out because computer crashes if try to run all files at once

#to get all images in file folder
dir = '/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/CLSM_biofilm_python/'

metadata["stack_count_total"] = ""
metadata["ch1_area_max"] = ""
metadata["ch2_area_max"] = ""
metadata["ch3_area_max"] = ""
metadata["total_area_max"] = ""

metadata["ch1_vol"] = ""
metadata["ch2_vol"] = ""
metadata["ch3_vol"] = ""
metadata["EPS_vol"] = ""
metadata["total_vol"] = ""
metadata["ch2v1_vol"] = ""
metadata["ch2v3_vol"] = ""


metadata['ch1_spread']= ""
metadata['ch2_spread']= ""
metadata['ch3_spread']= ""

metadata['ch1_height']= ""
metadata['ch2_height']= ""
metadata['ch3_height']= ""

for row in metadata.itertuples(): #loop through each file
        r= row.Index
        file_name = (row.file_name)
        path= dir+ '/' + file_name
        m = bioformats.get_omexml_metadata(path)
        o = bioformats.OMEXML(m)
        stack_count_o = o.image().Pixels.SizeZ
        # stack_count_total.append(stack_count_o)
        metadata.loc[r,'stack_count_total'] = int(stack_count_o)
        
        if stack_count_o > 4 : #if file is tall enough, loop through each z stack in the file
            z_list = np.arange(4, stack_count_o) #cut off first 4 slices bc controls tended to fluoresce in first 4 slices
            
            image_list_ch1 = []
            image_list_ch2 = []
            image_list_ch3 = []
            image_list_EPS = []
            image_list_total = []
            image_list_2v1 = []
            image_list_2v3 = []
            
            ch1_areas = []
            ch2_areas = []
            ch3_areas = []
            areas_EPS = []
            areas_total = []
            areas_2v1 = []
            areas_2v3 = []

            for z in z_list: 
                #load image, filter, and threshold for each channel
                image_raw_ch1= bioformats.load_image(path, rescale = False, z = z, c = 0)
                image_binary_ch1= thresholding_function(image_raw_ch1)
                
                image_raw_ch2= bioformats.load_image(path, rescale = False, z = z, c = 1)
                image_binary_ch2= thresholding_function(image_raw_ch2)
                
                image_raw_ch3= bioformats.load_image(path, rescale = False, z = z, c = 2)
                image_binary_ch3= thresholding_function(image_raw_ch3)
                
                #combine channels that we want to combine
                image_binary_EPS = np.logical_or(image_binary_ch1, image_binary_ch3) #combine ch1 and ch3 to see EPS
                image_binary_total = np.logical_or(image_binary_EPS, image_binary_ch2) #combine ch1, ch2, and ch3 to capture total biovolume
                image_binary_2v1 = np.logical_and(image_binary_ch1, image_binary_ch2) #look at percent overlap of ch1 with ch2 (just relative biovolume)
                image_binary_2v3 = np.logical_and(image_binary_ch3, image_binary_ch2) #look at percent overlap of ch3 with ch2 (just relative biovolume)
                
                #calculate parameters 
                area_ch1 = area_function(image_binary_ch1)
                area_ch2 = area_function(image_binary_ch2)
                area_ch3 = area_function(image_binary_ch3)
                area_EPS = area_function(image_binary_EPS)
                area_total = area_function(image_binary_total)
                area_2v1 = area_function(image_binary_2v1)
                area_2v3 = area_function(image_binary_2v3)
                
                #append parameters to lists if needed for whole image
                # image_list_ch1.append(image_binary_ch1)
                # image_list_ch2.append(image_binary_ch2)
                # image_list_ch3.append(image_binary_ch3)
                # image_list_EPS.append(image_binary_EPS)
                # image_list_total.append(image_binary_total)
                # image_list_2v1.append(image_binary_2v1)
                # image_list_2v3.append(image_binary_2v3)
                
                ch1_areas.append(area_ch1)
                ch2_areas.append(area_ch2)
                ch3_areas.append(area_ch3)
                areas_EPS.append(area_EPS)
                areas_total.append(area_total)
                areas_2v1.append(area_2v1)
                areas_2v3.append(area_2v3)
            
        else:
            ch1_areas = [0]
            ch2_areas = [0]
            ch3_areas = [0]
            areas_EPS = [0]
            areas_total = [0]
            areas_2v1 = [0]
            areas_2v3 = [0]
        
        metadata.loc[r,'ch1_area_max'] = max(ch1_areas) #max area of biofilm in ch1
        metadata.loc[r,'ch2_area_max'] = max(ch2_areas) #max area of biofilm in ch2
        metadata.loc[r,'ch3_area_max'] = max(ch3_areas) #max area of biofilm in ch3
        metadata.loc[r,'total_area_max'] = max(areas_total) #max area of biofilm in combined channels
       
        metadata.loc[r,'ch1_vol'] = sum(ch1_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2_vol'] = sum(ch2_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch3_vol'] = sum(ch3_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'EPS_vol'] = sum(areas_EPS) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'total_vol'] = sum(areas_total) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2v1_vol'] = sum(areas_2v1) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2v3_vol'] = sum(areas_2v3) *1 #volume calc, each stack is 1 um thick
        
        if metadata.loc[r,'ch1_area_max'] != 0:
            metadata.loc[r,'ch1_spread'] = metadata.loc[r,'ch1_vol']/metadata.loc[r,'ch1_area_max'] #spread is a proxy for height in Fish et al
        if metadata.loc[r,'ch2_area_max'] != 0:
            metadata.loc[r,'ch2_spread'] = metadata.loc[r,'ch2_vol']/metadata.loc[r,'ch2_area_max'] #spread is a proxy for height in Fish et al
        if metadata.loc[r,'ch3_area_max'] != 0:
            metadata.loc[r,'ch3_spread'] = metadata.loc[r,'ch3_vol']/metadata.loc[r,'ch3_area_max'] #spread is a proxy for height in Fish et al
        
        metadata.loc[r,'ch1_height'] = sum(np.array(ch1_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        metadata.loc[r,'ch2_height'] = sum(np.array(ch2_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        metadata.loc[r,'ch3_height'] = sum(np.array(ch3_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        
        metadata.loc[r,'ch1_height_0'] = sum(np.array(ch1_areas) > (0)) #max of stack count when stack count has at least 1 pixel
        metadata.loc[r,'ch2_height_0'] = sum(np.array(ch2_areas) > (0)) #max of stack count when stack count has at least 1 pixel
        metadata.loc[r,'ch3_height_0'] = sum(np.array(ch3_areas) > (0)) #max of stack count when stack count has at least 1 pixel


metadata7= metadata
    


# ### Batch 8a

# In[85]:


# start building the sample processing loops

metadata= metadata8a #need to parse out because computer crashes if try to run all files at once

#to get all images in file folder
dir = '/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/CLSM_biofilm_python/'

metadata["stack_count_total"] = ""
metadata["ch1_area_max"] = ""
metadata["ch2_area_max"] = ""
metadata["ch3_area_max"] = ""
metadata["total_area_max"] = ""

metadata["ch1_vol"] = ""
metadata["ch2_vol"] = ""
metadata["ch3_vol"] = ""
metadata["EPS_vol"] = ""
metadata["total_vol"] = ""
metadata["ch2v1_vol"] = ""
metadata["ch2v3_vol"] = ""


metadata['ch1_spread']= ""
metadata['ch2_spread']= ""
metadata['ch3_spread']= ""

metadata['ch1_height']= ""
metadata['ch2_height']= ""
metadata['ch3_height']= ""

for row in metadata.itertuples(): #loop through each file
        r= row.Index
        file_name = (row.file_name)
        path= dir+ '/' + file_name
        m = bioformats.get_omexml_metadata(path)
        o = bioformats.OMEXML(m)
        stack_count_o = o.image().Pixels.SizeZ
        # stack_count_total.append(stack_count_o)
        metadata.loc[r,'stack_count_total'] = int(stack_count_o)
        
        if stack_count_o > 4 : #if file is tall enough, loop through each z stack in the file
            z_list = np.arange(4, stack_count_o) #cut off first 4 slices bc controls tended to fluoresce in first 4 slices
            
            image_list_ch1 = []
            image_list_ch2 = []
            image_list_ch3 = []
            image_list_EPS = []
            image_list_total = []
            image_list_2v1 = []
            image_list_2v3 = []
            
            ch1_areas = []
            ch2_areas = []
            ch3_areas = []
            areas_EPS = []
            areas_total = []
            areas_2v1 = []
            areas_2v3 = []

            for z in z_list: 
                #load image, filter, and threshold for each channel
                image_raw_ch1= bioformats.load_image(path, rescale = False, z = z, c = 0)
                image_binary_ch1= thresholding_function(image_raw_ch1)
                
                image_raw_ch2= bioformats.load_image(path, rescale = False, z = z, c = 1)
                image_binary_ch2= thresholding_function(image_raw_ch2)
                
                image_raw_ch3= bioformats.load_image(path, rescale = False, z = z, c = 2)
                image_binary_ch3= thresholding_function(image_raw_ch3)
                
                #combine channels that we want to combine
                image_binary_EPS = np.logical_or(image_binary_ch1, image_binary_ch3) #combine ch1 and ch3 to see EPS
                image_binary_total = np.logical_or(image_binary_EPS, image_binary_ch2) #combine ch1, ch2, and ch3 to capture total biovolume
                image_binary_2v1 = np.logical_and(image_binary_ch1, image_binary_ch2) #look at percent overlap of ch1 with ch2 (just relative biovolume)
                image_binary_2v3 = np.logical_and(image_binary_ch3, image_binary_ch2) #look at percent overlap of ch3 with ch2 (just relative biovolume)
                
                #calculate parameters 
                area_ch1 = area_function(image_binary_ch1)
                area_ch2 = area_function(image_binary_ch2)
                area_ch3 = area_function(image_binary_ch3)
                area_EPS = area_function(image_binary_EPS)
                area_total = area_function(image_binary_total)
                area_2v1 = area_function(image_binary_2v1)
                area_2v3 = area_function(image_binary_2v3)
                
                #append parameters to lists if needed for whole image
                # image_list_ch1.append(image_binary_ch1)
                # image_list_ch2.append(image_binary_ch2)
                # image_list_ch3.append(image_binary_ch3)
                # image_list_EPS.append(image_binary_EPS)
                # image_list_total.append(image_binary_total)
                # image_list_2v1.append(image_binary_2v1)
                # image_list_2v3.append(image_binary_2v3)
                
                ch1_areas.append(area_ch1)
                ch2_areas.append(area_ch2)
                ch3_areas.append(area_ch3)
                areas_EPS.append(area_EPS)
                areas_total.append(area_total)
                areas_2v1.append(area_2v1)
                areas_2v3.append(area_2v3)
            
        else:
            ch1_areas = [0]
            ch2_areas = [0]
            ch3_areas = [0]
            areas_EPS = [0]
            areas_total = [0]
            areas_2v1 = [0]
            areas_2v3 = [0]
        
        metadata.loc[r,'ch1_area_max'] = max(ch1_areas) #max area of biofilm in ch1
        metadata.loc[r,'ch2_area_max'] = max(ch2_areas) #max area of biofilm in ch2
        metadata.loc[r,'ch3_area_max'] = max(ch3_areas) #max area of biofilm in ch3
        metadata.loc[r,'total_area_max'] = max(areas_total) #max area of biofilm in combined channels
       
        metadata.loc[r,'ch1_vol'] = sum(ch1_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2_vol'] = sum(ch2_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch3_vol'] = sum(ch3_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'EPS_vol'] = sum(areas_EPS) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'total_vol'] = sum(areas_total) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2v1_vol'] = sum(areas_2v1) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2v3_vol'] = sum(areas_2v3) *1 #volume calc, each stack is 1 um thick
        
        if metadata.loc[r,'ch1_area_max'] != 0:
            metadata.loc[r,'ch1_spread'] = metadata.loc[r,'ch1_vol']/metadata.loc[r,'ch1_area_max'] #spread is a proxy for height in Fish et al
        if metadata.loc[r,'ch2_area_max'] != 0:
            metadata.loc[r,'ch2_spread'] = metadata.loc[r,'ch2_vol']/metadata.loc[r,'ch2_area_max'] #spread is a proxy for height in Fish et al
        if metadata.loc[r,'ch3_area_max'] != 0:
            metadata.loc[r,'ch3_spread'] = metadata.loc[r,'ch3_vol']/metadata.loc[r,'ch3_area_max'] #spread is a proxy for height in Fish et al
        
        metadata.loc[r,'ch1_height'] = sum(np.array(ch1_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        metadata.loc[r,'ch2_height'] = sum(np.array(ch2_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        metadata.loc[r,'ch3_height'] = sum(np.array(ch3_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        
        metadata.loc[r,'ch1_height_0'] = sum(np.array(ch1_areas) > (0)) #max of stack count when stack count has at least 1 pixel
        metadata.loc[r,'ch2_height_0'] = sum(np.array(ch2_areas) > (0)) #max of stack count when stack count has at least 1 pixel
        metadata.loc[r,'ch3_height_0'] = sum(np.array(ch3_areas) > (0)) #max of stack count when stack count has at least 1 pixel


metadata8a= metadata
    


# In[87]:


metadata8a


# ### Batch 8b

# In[88]:


# start building the sample processing loops

metadata= metadata8b #need to parse out because computer crashes if try to run all files at once

#to get all images in file folder
dir = '/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/CLSM_biofilm_python/'

metadata["stack_count_total"] = ""
metadata["ch1_area_max"] = ""
metadata["ch2_area_max"] = ""
metadata["ch3_area_max"] = ""
metadata["total_area_max"] = ""

metadata["ch1_vol"] = ""
metadata["ch2_vol"] = ""
metadata["ch3_vol"] = ""
metadata["EPS_vol"] = ""
metadata["total_vol"] = ""
metadata["ch2v1_vol"] = ""
metadata["ch2v3_vol"] = ""


metadata['ch1_spread']= ""
metadata['ch2_spread']= ""
metadata['ch3_spread']= ""

metadata['ch1_height']= ""
metadata['ch2_height']= ""
metadata['ch3_height']= ""

for row in metadata.itertuples(): #loop through each file
        r= row.Index
        file_name = (row.file_name)
        path= dir+ '/' + file_name
        m = bioformats.get_omexml_metadata(path)
        o = bioformats.OMEXML(m)
        stack_count_o = o.image().Pixels.SizeZ
        # stack_count_total.append(stack_count_o)
        metadata.loc[r,'stack_count_total'] = int(stack_count_o)
        
        if stack_count_o > 4 : #if file is tall enough, loop through each z stack in the file
            z_list = np.arange(4, stack_count_o) #cut off first 4 slices bc controls tended to fluoresce in first 4 slices
            
            image_list_ch1 = []
            image_list_ch2 = []
            image_list_ch3 = []
            image_list_EPS = []
            image_list_total = []
            image_list_2v1 = []
            image_list_2v3 = []
            
            ch1_areas = []
            ch2_areas = []
            ch3_areas = []
            areas_EPS = []
            areas_total = []
            areas_2v1 = []
            areas_2v3 = []

            for z in z_list: 
                #load image, filter, and threshold for each channel
                image_raw_ch1= bioformats.load_image(path, rescale = False, z = z, c = 0)
                image_binary_ch1= thresholding_function(image_raw_ch1)
                
                image_raw_ch2= bioformats.load_image(path, rescale = False, z = z, c = 1)
                image_binary_ch2= thresholding_function(image_raw_ch2)
                
                image_raw_ch3= bioformats.load_image(path, rescale = False, z = z, c = 2)
                image_binary_ch3= thresholding_function(image_raw_ch3)
                
                #combine channels that we want to combine
                image_binary_EPS = np.logical_or(image_binary_ch1, image_binary_ch3) #combine ch1 and ch3 to see EPS
                image_binary_total = np.logical_or(image_binary_EPS, image_binary_ch2) #combine ch1, ch2, and ch3 to capture total biovolume
                image_binary_2v1 = np.logical_and(image_binary_ch1, image_binary_ch2) #look at percent overlap of ch1 with ch2 (just relative biovolume)
                image_binary_2v3 = np.logical_and(image_binary_ch3, image_binary_ch2) #look at percent overlap of ch3 with ch2 (just relative biovolume)
                
                #calculate parameters 
                area_ch1 = area_function(image_binary_ch1)
                area_ch2 = area_function(image_binary_ch2)
                area_ch3 = area_function(image_binary_ch3)
                area_EPS = area_function(image_binary_EPS)
                area_total = area_function(image_binary_total)
                area_2v1 = area_function(image_binary_2v1)
                area_2v3 = area_function(image_binary_2v3)
                
                #append parameters to lists if needed for whole image
                # image_list_ch1.append(image_binary_ch1)
                # image_list_ch2.append(image_binary_ch2)
                # image_list_ch3.append(image_binary_ch3)
                # image_list_EPS.append(image_binary_EPS)
                # image_list_total.append(image_binary_total)
                # image_list_2v1.append(image_binary_2v1)
                # image_list_2v3.append(image_binary_2v3)
                
                ch1_areas.append(area_ch1)
                ch2_areas.append(area_ch2)
                ch3_areas.append(area_ch3)
                areas_EPS.append(area_EPS)
                areas_total.append(area_total)
                areas_2v1.append(area_2v1)
                areas_2v3.append(area_2v3)
            
        else:
            ch1_areas = [0]
            ch2_areas = [0]
            ch3_areas = [0]
            areas_EPS = [0]
            areas_total = [0]
            areas_2v1 = [0]
            areas_2v3 = [0]
        
        metadata.loc[r,'ch1_area_max'] = max(ch1_areas) #max area of biofilm in ch1
        metadata.loc[r,'ch2_area_max'] = max(ch2_areas) #max area of biofilm in ch2
        metadata.loc[r,'ch3_area_max'] = max(ch3_areas) #max area of biofilm in ch3
        metadata.loc[r,'total_area_max'] = max(areas_total) #max area of biofilm in combined channels
       
        metadata.loc[r,'ch1_vol'] = sum(ch1_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2_vol'] = sum(ch2_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch3_vol'] = sum(ch3_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'EPS_vol'] = sum(areas_EPS) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'total_vol'] = sum(areas_total) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2v1_vol'] = sum(areas_2v1) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2v3_vol'] = sum(areas_2v3) *1 #volume calc, each stack is 1 um thick
        
        if metadata.loc[r,'ch1_area_max'] != 0:
            metadata.loc[r,'ch1_spread'] = metadata.loc[r,'ch1_vol']/metadata.loc[r,'ch1_area_max'] #spread is a proxy for height in Fish et al
        if metadata.loc[r,'ch2_area_max'] != 0:
            metadata.loc[r,'ch2_spread'] = metadata.loc[r,'ch2_vol']/metadata.loc[r,'ch2_area_max'] #spread is a proxy for height in Fish et al
        if metadata.loc[r,'ch3_area_max'] != 0:
            metadata.loc[r,'ch3_spread'] = metadata.loc[r,'ch3_vol']/metadata.loc[r,'ch3_area_max'] #spread is a proxy for height in Fish et al
        
        metadata.loc[r,'ch1_height'] = sum(np.array(ch1_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        metadata.loc[r,'ch2_height'] = sum(np.array(ch2_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        metadata.loc[r,'ch3_height'] = sum(np.array(ch3_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        
        metadata.loc[r,'ch1_height_0'] = sum(np.array(ch1_areas) > (0)) #max of stack count when stack count has at least 1 pixel
        metadata.loc[r,'ch2_height_0'] = sum(np.array(ch2_areas) > (0)) #max of stack count when stack count has at least 1 pixel
        metadata.loc[r,'ch3_height_0'] = sum(np.array(ch3_areas) > (0)) #max of stack count when stack count has at least 1 pixel


metadata8b= metadata
    


# ### One file is acting up, image was not collected properly (skipped channels 2 and 3)

# In[31]:


# # process just the one weird file
# metadata= metadata8b

# dir = '/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/CLSM_biofilm_python/'

# metadata["stack_count_total"] = ""
# metadata["ch1_area_max"] = ""
# metadata["ch2_area_max"] = ""
# metadata["ch3_area_max"] = ""

# metadata["ch1_vol"] = ""
# metadata["ch2_vol"] = ""
# metadata["ch3_vol"] = ""
# metadata["EPS_vol"] = ""
# metadata["total_vol"] = ""
# metadata["ch2v1_vol"] = ""
# metadata["ch2v3_vol"] = ""


# metadata['ch1_spread']= ""
# metadata['ch2_spread']= ""
# metadata['ch3_spread']= ""

# metadata['ch1_height']= ""
# metadata['ch2_height']= ""
# metadata['ch3_height']= ""

# r=197
# file_name = '20_0104_AR3_2.czi'
# metadata.loc[r,'file_name'] = file_name
# path= dir + '/' + file_name
# m = bioformats.get_omexml_metadata(path)
# o = bioformats.OMEXML(m)
# stack_count_o = o.image().Pixels.SizeZ
# metadata.loc[r,'stack_count_total'] = int(stack_count_o)

# if stack_count_o > 4 : #if file is tall enough, loop through each z stack in the file
#     z_list = np.arange(4, stack_count_o) #cut off first 4 slices bc controls tended to fluoresce in first 4 slices

#     image_list_ch1 = []
#     image_list_ch2 = []
#     image_list_ch3 = []
#     image_list_EPS = []
#     image_list_total = []
#     image_list_2v1 = []
#     image_list_2v3 = []

#     ch1_areas = []
#     ch2_areas = []
#     ch3_areas = []
#     areas_EPS = []
#     areas_total = []
#     areas_2v1 = []
#     areas_2v3 = []

#     for z in z_list: 
#         #load image, filter, and threshold for each channel
#         image_raw_ch1= bioformats.load_image(path, rescale = False, z = z, c = 0)
#         image_binary_ch1= thresholding_function(image_raw_ch1)

#         # image_raw_ch2= bioformats.load_image(path, rescale = False, z = z, c = 1)
#         # image_binary_ch2= thresholding_function(image_raw_ch2)

#         # image_raw_ch3= bioformats.load_image(path, rescale = False, z = z, c = 2)
#         # image_binary_ch3= thresholding_function(image_raw_ch3)

# #         #combine channels that we want to combine
# #         image_binary_EPS = np.logical_or(image_binary_ch1, image_binary_ch3) #combine ch1 and ch3 to see EPS
# #         image_binary_total = np.logical_or(image_binary_EPS, image_binary_ch2) #combine ch1, ch2, and ch3 to capture total biovolume
# #         image_binary_2v1 = np.logical_and(image_binary_ch1, image_binary_ch2) #look at percent overlap of ch1 with ch2 (just relative biovolume)
# #         image_binary_2v3 = np.logical_and(image_binary_ch3, image_binary_ch2) #look at percent overlap of ch3 with ch2 (just relative biovolume)

#         #calculate parameters 
#         area_ch1 = area_function(image_binary_ch1)
# #         area_ch2 = area_function(image_binary_ch2)
# #         area_ch3 = area_function(image_binary_ch3)
# #         area_EPS = area_function(image_binary_EPS)
# #         area_total = area_function(image_binary_total)
# #         area_2v1 = area_function(image_binary_2v1)
# #         area_2v3 = area_function(image_binary_2v3)


#         ch1_areas.append(area_ch1)
# #         ch2_areas.append(area_ch2)
# #         ch3_areas.append(area_ch3)
# #         areas_EPS.append(area_EPS)
# #         areas_total.append(area_total)
# #         areas_2v1.append(area_2v1)
# #         areas_2v3.append(area_2v3)



# metadata.loc[r,'ch1_area_max'] = max(ch1_areas) #max area of biofilm in ch1
# # metadata.loc[r,'ch2_area_max'] = max(ch2_areas) #max area of biofilm in ch2
# # metadata.loc[r,'ch3_area_max'] = max(ch3_areas) #max area of biofilm in ch3

# metadata.loc[r,'ch1_vol'] = sum(ch1_areas) *1 #volume calc, each stack is 1 um thick
# # metadata.loc[r,'ch2_vol'] = sum(ch2_areas) *1 #volume calc, each stack is 1 um thick
# # metadata.loc[r,'ch3_vol'] = sum(ch3_areas) *1 #volume calc, each stack is 1 um thick
# # metadata.loc[r,'EPS_vol'] = sum(areas_EPS) *1 #volume calc, each stack is 1 um thick
# # metadata.loc[r,'total_vol'] = sum(areas_total) *1 #volume calc, each stack is 1 um thick
# # metadata.loc[r,'ch2v1_vol'] = sum(areas_2v1) *1 #volume calc, each stack is 1 um thick
# # metadata.loc[r,'ch2v3_vol'] = sum(areas_2v3) *1 #volume calc, each stack is 1 um thick

# if metadata.loc[r,'ch1_area_max'] != 0:
#     metadata.loc[r,'ch1_spread'] = metadata.loc[r,'ch1_vol']/metadata.loc[r,'ch1_area_max'] #spread is a proxy for height in Fish et al
# # if metadata.loc[r,'ch2_area_max'] != 0:
# #     metadata.loc[r,'ch2_spread'] = metadata.loc[r,'ch2_vol']/metadata.loc[r,'ch2_area_max'] #spread is a proxy for height in Fish et al
# # if metadata.loc[r,'ch3_area_max'] != 0:
# #     metadata.loc[r,'ch3_spread'] = metadata.loc[r,'ch3_vol']/metadata.loc[r,'ch3_area_max'] #spread is a proxy for height in Fish et al

# metadata.loc[r,'ch1_height'] = sum(np.array(ch1_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
# # metadata.loc[r,'ch2_height'] = sum(np.array(ch2_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
# # metadata.loc[r,'ch3_height'] = sum(np.array(ch3_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%

# metadata.loc[r,'ch1_height_0'] = sum(np.array(ch1_areas) > (0)) #max of stack count when stack count has at least 1 pixel
# # metadata.loc[r,'ch2_height_0'] = sum(np.array(ch2_areas) > (0)) #max of stack count when stack count has at least 1 pixel
# # metadata.loc[r,'ch3_height_0'] = sum(np.array(ch3_areas) > (0)) #max of stack count when stack count has at least 1 pixel


# metadata197 = metadata
# metadata197


# ### Batch 9

# In[89]:


# start building the sample processing loops

metadata= metadata9 #need to parse out because computer crashes if try to run all files at once

#to get all images in file folder
dir = '/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/CLSM_biofilm_python/'

metadata["stack_count_total"] = ""
metadata["ch1_area_max"] = ""
metadata["ch2_area_max"] = ""
metadata["ch3_area_max"] = ""
metadata["total_area_max"] = ""

metadata["ch1_vol"] = ""
metadata["ch2_vol"] = ""
metadata["ch3_vol"] = ""
metadata["EPS_vol"] = ""
metadata["total_vol"] = ""
metadata["ch2v1_vol"] = ""
metadata["ch2v3_vol"] = ""


metadata['ch1_spread']= ""
metadata['ch2_spread']= ""
metadata['ch3_spread']= ""

metadata['ch1_height']= ""
metadata['ch2_height']= ""
metadata['ch3_height']= ""

for row in metadata.itertuples(): #loop through each file
        r= row.Index
        file_name = (row.file_name)
        path= dir+ '/' + file_name
        m = bioformats.get_omexml_metadata(path)
        o = bioformats.OMEXML(m)
        stack_count_o = o.image().Pixels.SizeZ
        # stack_count_total.append(stack_count_o)
        metadata.loc[r,'stack_count_total'] = int(stack_count_o)
        
        if stack_count_o > 4 : #if file is tall enough, loop through each z stack in the file
            z_list = np.arange(4, stack_count_o) #cut off first 4 slices bc controls tended to fluoresce in first 4 slices
            
            image_list_ch1 = []
            image_list_ch2 = []
            image_list_ch3 = []
            image_list_EPS = []
            image_list_total = []
            image_list_2v1 = []
            image_list_2v3 = []
            
            ch1_areas = []
            ch2_areas = []
            ch3_areas = []
            areas_EPS = []
            areas_total = []
            areas_2v1 = []
            areas_2v3 = []

            for z in z_list: 
                #load image, filter, and threshold for each channel
                image_raw_ch1= bioformats.load_image(path, rescale = False, z = z, c = 0)
                image_binary_ch1= thresholding_function(image_raw_ch1)
                
                image_raw_ch2= bioformats.load_image(path, rescale = False, z = z, c = 1)
                image_binary_ch2= thresholding_function(image_raw_ch2)
                
                image_raw_ch3= bioformats.load_image(path, rescale = False, z = z, c = 2)
                image_binary_ch3= thresholding_function(image_raw_ch3)
                
                #combine channels that we want to combine
                image_binary_EPS = np.logical_or(image_binary_ch1, image_binary_ch3) #combine ch1 and ch3 to see EPS
                image_binary_total = np.logical_or(image_binary_EPS, image_binary_ch2) #combine ch1, ch2, and ch3 to capture total biovolume
                image_binary_2v1 = np.logical_and(image_binary_ch1, image_binary_ch2) #look at percent overlap of ch1 with ch2 (just relative biovolume)
                image_binary_2v3 = np.logical_and(image_binary_ch3, image_binary_ch2) #look at percent overlap of ch3 with ch2 (just relative biovolume)
                
                #calculate parameters 
                area_ch1 = area_function(image_binary_ch1)
                area_ch2 = area_function(image_binary_ch2)
                area_ch3 = area_function(image_binary_ch3)
                area_EPS = area_function(image_binary_EPS)
                area_total = area_function(image_binary_total)
                area_2v1 = area_function(image_binary_2v1)
                area_2v3 = area_function(image_binary_2v3)
                
                #append parameters to lists if needed for whole image
                # image_list_ch1.append(image_binary_ch1)
                # image_list_ch2.append(image_binary_ch2)
                # image_list_ch3.append(image_binary_ch3)
                # image_list_EPS.append(image_binary_EPS)
                # image_list_total.append(image_binary_total)
                # image_list_2v1.append(image_binary_2v1)
                # image_list_2v3.append(image_binary_2v3)
                
                ch1_areas.append(area_ch1)
                ch2_areas.append(area_ch2)
                ch3_areas.append(area_ch3)
                areas_EPS.append(area_EPS)
                areas_total.append(area_total)
                areas_2v1.append(area_2v1)
                areas_2v3.append(area_2v3)
            
        else:
            ch1_areas = [0]
            ch2_areas = [0]
            ch3_areas = [0]
            areas_EPS = [0]
            areas_total = [0]
            areas_2v1 = [0]
            areas_2v3 = [0]
        
        metadata.loc[r,'ch1_area_max'] = max(ch1_areas) #max area of biofilm in ch1
        metadata.loc[r,'ch2_area_max'] = max(ch2_areas) #max area of biofilm in ch2
        metadata.loc[r,'ch3_area_max'] = max(ch3_areas) #max area of biofilm in ch3
        metadata.loc[r,'total_area_max'] = max(areas_total) #max area of biofilm in combined channels
       
        metadata.loc[r,'ch1_vol'] = sum(ch1_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2_vol'] = sum(ch2_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch3_vol'] = sum(ch3_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'EPS_vol'] = sum(areas_EPS) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'total_vol'] = sum(areas_total) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2v1_vol'] = sum(areas_2v1) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2v3_vol'] = sum(areas_2v3) *1 #volume calc, each stack is 1 um thick
        
        if metadata.loc[r,'ch1_area_max'] != 0:
            metadata.loc[r,'ch1_spread'] = metadata.loc[r,'ch1_vol']/metadata.loc[r,'ch1_area_max'] #spread is a proxy for height in Fish et al
        if metadata.loc[r,'ch2_area_max'] != 0:
            metadata.loc[r,'ch2_spread'] = metadata.loc[r,'ch2_vol']/metadata.loc[r,'ch2_area_max'] #spread is a proxy for height in Fish et al
        if metadata.loc[r,'ch3_area_max'] != 0:
            metadata.loc[r,'ch3_spread'] = metadata.loc[r,'ch3_vol']/metadata.loc[r,'ch3_area_max'] #spread is a proxy for height in Fish et al
        
        metadata.loc[r,'ch1_height'] = sum(np.array(ch1_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        metadata.loc[r,'ch2_height'] = sum(np.array(ch2_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        metadata.loc[r,'ch3_height'] = sum(np.array(ch3_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        
        metadata.loc[r,'ch1_height_0'] = sum(np.array(ch1_areas) > (0)) #max of stack count when stack count has at least 1 pixel
        metadata.loc[r,'ch2_height_0'] = sum(np.array(ch2_areas) > (0)) #max of stack count when stack count has at least 1 pixel
        metadata.loc[r,'ch3_height_0'] = sum(np.array(ch3_areas) > (0)) #max of stack count when stack count has at least 1 pixel


metadata9= metadata
    


# ### Batch 10

# In[90]:


# start building the sample processing loops

metadata= metadata10 #need to parse out because computer crashes if try to run all files at once

#to get all images in file folder
dir = '/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/CLSM_biofilm_python/'

metadata["stack_count_total"] = ""
metadata["ch1_area_max"] = ""
metadata["ch2_area_max"] = ""
metadata["ch3_area_max"] = ""
metadata["total_area_max"] = ""

metadata["ch1_vol"] = ""
metadata["ch2_vol"] = ""
metadata["ch3_vol"] = ""
metadata["EPS_vol"] = ""
metadata["total_vol"] = ""
metadata["ch2v1_vol"] = ""
metadata["ch2v3_vol"] = ""


metadata['ch1_spread']= ""
metadata['ch2_spread']= ""
metadata['ch3_spread']= ""

metadata['ch1_height']= ""
metadata['ch2_height']= ""
metadata['ch3_height']= ""

for row in metadata.itertuples(): #loop through each file
        r= row.Index
        file_name = (row.file_name)
        path= dir+ '/' + file_name
        m = bioformats.get_omexml_metadata(path)
        o = bioformats.OMEXML(m)
        stack_count_o = o.image().Pixels.SizeZ
        # stack_count_total.append(stack_count_o)
        metadata.loc[r,'stack_count_total'] = int(stack_count_o)
        
        if stack_count_o > 4 : #if file is tall enough, loop through each z stack in the file
            z_list = np.arange(4, stack_count_o) #cut off first 4 slices bc controls tended to fluoresce in first 4 slices
            
            image_list_ch1 = []
            image_list_ch2 = []
            image_list_ch3 = []
            image_list_EPS = []
            image_list_total = []
            image_list_2v1 = []
            image_list_2v3 = []
            
            ch1_areas = []
            ch2_areas = []
            ch3_areas = []
            areas_EPS = []
            areas_total = []
            areas_2v1 = []
            areas_2v3 = []

            for z in z_list: 
                #load image, filter, and threshold for each channel
                image_raw_ch1= bioformats.load_image(path, rescale = False, z = z, c = 0)
                image_binary_ch1= thresholding_function(image_raw_ch1)
                
                image_raw_ch2= bioformats.load_image(path, rescale = False, z = z, c = 1)
                image_binary_ch2= thresholding_function(image_raw_ch2)
                
                image_raw_ch3= bioformats.load_image(path, rescale = False, z = z, c = 2)
                image_binary_ch3= thresholding_function(image_raw_ch3)
                
                #combine channels that we want to combine
                image_binary_EPS = np.logical_or(image_binary_ch1, image_binary_ch3) #combine ch1 and ch3 to see EPS
                image_binary_total = np.logical_or(image_binary_EPS, image_binary_ch2) #combine ch1, ch2, and ch3 to capture total biovolume
                image_binary_2v1 = np.logical_and(image_binary_ch1, image_binary_ch2) #look at percent overlap of ch1 with ch2 (just relative biovolume)
                image_binary_2v3 = np.logical_and(image_binary_ch3, image_binary_ch2) #look at percent overlap of ch3 with ch2 (just relative biovolume)
                
                #calculate parameters 
                area_ch1 = area_function(image_binary_ch1)
                area_ch2 = area_function(image_binary_ch2)
                area_ch3 = area_function(image_binary_ch3)
                area_EPS = area_function(image_binary_EPS)
                area_total = area_function(image_binary_total)
                area_2v1 = area_function(image_binary_2v1)
                area_2v3 = area_function(image_binary_2v3)
                
                #append parameters to lists if needed for whole image
                # image_list_ch1.append(image_binary_ch1)
                # image_list_ch2.append(image_binary_ch2)
                # image_list_ch3.append(image_binary_ch3)
                # image_list_EPS.append(image_binary_EPS)
                # image_list_total.append(image_binary_total)
                # image_list_2v1.append(image_binary_2v1)
                # image_list_2v3.append(image_binary_2v3)
                
                ch1_areas.append(area_ch1)
                ch2_areas.append(area_ch2)
                ch3_areas.append(area_ch3)
                areas_EPS.append(area_EPS)
                areas_total.append(area_total)
                areas_2v1.append(area_2v1)
                areas_2v3.append(area_2v3)
            
        else:
            ch1_areas = [0]
            ch2_areas = [0]
            ch3_areas = [0]
            areas_EPS = [0]
            areas_total = [0]
            areas_2v1 = [0]
            areas_2v3 = [0]
        
        metadata.loc[r,'ch1_area_max'] = max(ch1_areas) #max area of biofilm in ch1
        metadata.loc[r,'ch2_area_max'] = max(ch2_areas) #max area of biofilm in ch2
        metadata.loc[r,'ch3_area_max'] = max(ch3_areas) #max area of biofilm in ch3
        metadata.loc[r,'total_area_max'] = max(areas_total) #max area of biofilm in combined channels
       
        metadata.loc[r,'ch1_vol'] = sum(ch1_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2_vol'] = sum(ch2_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch3_vol'] = sum(ch3_areas) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'EPS_vol'] = sum(areas_EPS) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'total_vol'] = sum(areas_total) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2v1_vol'] = sum(areas_2v1) *1 #volume calc, each stack is 1 um thick
        metadata.loc[r,'ch2v3_vol'] = sum(areas_2v3) *1 #volume calc, each stack is 1 um thick
        
        if metadata.loc[r,'ch1_area_max'] != 0:
            metadata.loc[r,'ch1_spread'] = metadata.loc[r,'ch1_vol']/metadata.loc[r,'ch1_area_max'] #spread is a proxy for height in Fish et al
        if metadata.loc[r,'ch2_area_max'] != 0:
            metadata.loc[r,'ch2_spread'] = metadata.loc[r,'ch2_vol']/metadata.loc[r,'ch2_area_max'] #spread is a proxy for height in Fish et al
        if metadata.loc[r,'ch3_area_max'] != 0:
            metadata.loc[r,'ch3_spread'] = metadata.loc[r,'ch3_vol']/metadata.loc[r,'ch3_area_max'] #spread is a proxy for height in Fish et al
        
        metadata.loc[r,'ch1_height'] = sum(np.array(ch1_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        metadata.loc[r,'ch2_height'] = sum(np.array(ch2_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        metadata.loc[r,'ch3_height'] = sum(np.array(ch3_areas) > (0.01*212.5*212.5)) #max of stack count when stack count has fraction coverage > 1%
        
        metadata.loc[r,'ch1_height_0'] = sum(np.array(ch1_areas) > (0)) #max of stack count when stack count has at least 1 pixel
        metadata.loc[r,'ch2_height_0'] = sum(np.array(ch2_areas) > (0)) #max of stack count when stack count has at least 1 pixel
        metadata.loc[r,'ch3_height_0'] = sum(np.array(ch3_areas) > (0)) #max of stack count when stack count has at least 1 pixel


metadata10= metadata
    


# ## export dataframe as csv then plot in R

# In[91]:


#download the outputs of parameters after subtracting first 4 z frames

metadata1.to_csv('/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/analyzed_clsm_outputs/analyzed_biofilm_clsm1.csv', index=False, quoting=csv.QUOTE_NONE, header=True)
metadata2.to_csv('/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/analyzed_clsm_outputs/analyzed_biofilm_clsm2.csv', index=False, quoting=csv.QUOTE_NONE, header=True)
metadata3.to_csv('/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/analyzed_clsm_outputs/analyzed_biofilm_clsm3.csv', index=False, quoting=csv.QUOTE_NONE, header=True)
metadata4.to_csv('/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/analyzed_clsm_outputs/analyzed_biofilm_clsm4.csv', index=False, quoting=csv.QUOTE_NONE, header=True)
metadata5.to_csv('/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/analyzed_clsm_outputs/analyzed_biofilm_clsm5.csv', index=False, quoting=csv.QUOTE_NONE, header=True)
metadata6.to_csv('/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/analyzed_clsm_outputs/analyzed_biofilm_clsm6.csv', index=False, quoting=csv.QUOTE_NONE, header=True)
metadata7.to_csv('/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/analyzed_clsm_outputs/analyzed_biofilm_clsm7.csv', index=False, quoting=csv.QUOTE_NONE, header=True)
metadata8a.to_csv('/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/analyzed_clsm_outputs/analyzed_biofilm_clsm8a.csv', index=False, quoting=csv.QUOTE_NONE, header=True)
metadata8b.to_csv('/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/analyzed_clsm_outputs/analyzed_biofilm_clsm8b.csv', index=False, quoting=csv.QUOTE_NONE, header=True)
metadata9.to_csv('/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/analyzed_clsm_outputs/analyzed_biofilm_clsm9.csv', index=False, quoting=csv.QUOTE_NONE, header=True)
metadata10.to_csv('/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/analyzed_clsm_outputs/analyzed_biofilm_clsm10.csv', index=False, quoting=csv.QUOTE_NONE, header=True)
metadata11.to_csv('/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/analyzed_clsm_outputs/analyzed_biofilm_clsm11.csv', index=False, quoting=csv.QUOTE_NONE, header=True)
metadata12.to_csv('/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/analyzed_clsm_outputs/analyzed_biofilm_clsm12.csv', index=False, quoting=csv.QUOTE_NONE, header=True)
# metadata197.to_csv('/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/analyzed_clsm_outputs/analyzed_biofilm_clsm197.csv', index=False, quoting=csv.QUOTE_NONE, header=True)
metadata13.to_csv('/Users/hannahgreenwald/Documents/Documents/Berkeley_Research/DPR_AR_Research/CLSM_Analysis/analyzed_clsm_outputs/analyzed_biofilm_clsm13.csv', index=False, quoting=csv.QUOTE_NONE, header=True)



# ## Kill the Virtual Machine. End Session

# In[93]:


javabridge.kill_vm()

