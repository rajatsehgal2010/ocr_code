
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
from PIL import Image,ImageDraw
import pytesseract as pt
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
from shapely.geometry import Polygon
from shapely.geometry import LineString  
from math import cos  
from math import sin  
from math import pi 
import math
import glob
import subprocess
from autocorrect import spell
from datetime import datetime
from dateutil.parser import parse
import json
import concurrent.futures
import time
import multiprocessing
#import cython
#%load_ext Cython



# In[2]:


def clean_image(path,path_out):
    image_name = path.split("/")[-1].split(".")[0]
    print(image_name)
    path_out = path_out + image_name+".jpg"
    print(path_out)
    command = "./textcleaner -g -e stretch -f 25 -o 10 -u -s 1 -T -p 10 {} {}".format(path,path_out)
    returnvalue = subprocess.call(command,shell=True)


# In[3]:


def east_detect():
    subprocess.call('./east.sh',shell=True)


# In[4]:


def read_image(image_name):
    image=cv2.imread("/home/ansul/EAST/merge/{}.jpg".format(image_name)) 
    imgray = cv2.cvtColor(image.copy(),cv2.COLOR_BGR2GRAY)
    ret,thr = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return image,thr


# In[5]:


def df_org(east_boxtxt_path):
    text_bb = pd.read_csv(east_boxtxt_path, sep=",", header=None)
    text_bb.columns = ["a", "b", "c", "d","e","f","g","h"]
    text_bb
    x1=[]
    x2=[]
    y1=[]
    y2=[]
    x_c=[]
    y_c=[]
    box=[]
    polygons = []
    center=[]
    for i in text_bb.index:
        x_mi=min(text_bb.iloc[i]['a'],text_bb.iloc[i]['c'],text_bb.iloc[i]['e'],text_bb.iloc[i]['g'])
        x_ma=max(text_bb.iloc[i]['a'],text_bb.iloc[i]['c'],text_bb.iloc[i]['e'],text_bb.iloc[i]['g'])
        y_mi=min(text_bb.iloc[i]['b'],text_bb.iloc[i]['d'],text_bb.iloc[i]['f'],text_bb.iloc[i]['h'])
        y_ma=max(text_bb.iloc[i]['b'],text_bb.iloc[i]['d'],text_bb.iloc[i]['f'],text_bb.iloc[i]['h'])
        x1.append(x_mi)
        x2.append(x_ma)
        y1.append(y_mi)
        y2.append(y_ma)
        xc=int((x_mi+x_ma)/2)
        yc=int((x_mi+x_ma)/2)
        x_c.append(int((x_mi+x_ma)/2))
        y_c.append(int((y_mi+y_ma)/2))
        box.append((x_mi,x_ma,y_mi,y_ma))
        center.append([xc,yc])
        #center.append(((x1+x2)/2,(y1+y2)/2))
        polygons.append(Polygon([(x_mi,y_ma), (x_mi,y_mi), (x_ma,y_mi), (x_ma,y_ma)])) 
    x1 = pd.Series(x1)
    x2 = pd.Series(x2)
    y1 = pd.Series(y1)
    y2 = pd.Series(y2)
    x_c = pd.Series(x_c)
    y_c = pd.Series(y_c)
    cordinate=pd.DataFrame(data=dict(x1=x1,x2=x2,y1=y1,y2=y2,x_c=x_c,y_c=y_c), index=x1.index)
    return cordinate,polygons


# In[6]:


def closest_polygon(x, y, angle, polygons, dist = 10000):  

    angle = angle * pi / 180.0  
    line = LineString([(x, y), (x + dist * sin(angle), y + dist * cos(angle))])  

    dist_min = None  
    closest_polygon = None  
    for i in range(len(polygons)):  
        difference = line.difference(polygons[i])  
        if difference.geom_type == 'MultiLineString':  
            dist = list(difference.geoms)[0].length  
            if dist_min is None or dist_min > dist:  
                dist_min = dist  
                closest_polygon = i  



    return {'closest_polygon': closest_polygon, 'distance': dist_min}  


# In[7]:


def merge_overlap(cordinate,polygons):
    for i in cordinate.index:
        if cordinate.iloc[i]['x1'] != -1:
        #print("-----out---")
            close_r=closest_polygon(cordinate.iloc[i]['x_c'],cordinate.iloc[i]['y_c'],90,polygons)
       #close_l=closest_polygon(cordinate.iloc[i]['x_c'],cordinate.iloc[i]['y_c'],270,polygons)
            i_r=close_r["closest_polygon"]
       #i_l=close_l["closest_polygon"]
            if (i_r != None) :
                while (i_r!=None) and ((0 < cordinate.iloc[i_r]['x1']-cordinate.iloc[i]['x2']  < 20) or ((cordinate.iloc[i]['x2'] > cordinate.iloc[i_r]['x1']) and (cordinate.iloc[i]['y2'] > cordinate.iloc[i_r]['y1']))):
                    cordinate.iloc[i]['x2']=cordinate.iloc[i_r]['x2']
                    if (cordinate.iloc[i_r]['y2'] > cordinate.iloc[i]['y2']):
                        cordinate.iloc[i]['y2']=cordinate.iloc[i_r]['y2']
                    if (cordinate.iloc[i_r]['y1'] < cordinate.iloc[i]['y1']):
                        cordinate.iloc[i]['y1']=cordinate.iloc[i_r]['y1']
               
                    polygons[i]=Polygon([(cordinate.iloc[i]['x1'],cordinate.iloc[i]['y2']), (cordinate.iloc[i]['x1'],cordinate.iloc[i]['y1']), (cordinate.iloc[i]['x2'],cordinate.iloc[i]['y1']), (cordinate.iloc[i]['x2'],cordinate.iloc[i]['y2'])])
                    polygons[i_r]=Polygon([(5000,5000),(5000,4900),(5100,4900),(5100,5000)])
              
                    xt,yt,xb,yb=cordinate.iloc[i]['x1'],cordinate.iloc[i]['y1'],cordinate.iloc[i]['x2'],cordinate.iloc[i]['y2']
                    cordinate.iloc[i_r]['x1']=-1
                    xc=(xb+xt)/2
                    yc=(yt+yb)/2
                    close_r=closest_polygon(xc,yc,90,polygons)
               
               #close_l=closest_polygon(cordinate.iloc[i]['x_c'],cordinate.iloc[i]['y_c'],270,polygons)
                    i_r=close_r["closest_polygon"]
    return cordinate
               


# In[8]:


def box_inten(cordinate):
    text_bb1 = cordinate
    poly = []
    polygons = []
    centres = []
    for i in text_bb1.index:
        if text_bb1.iloc[i]['x1'] != -1:
            x1=text_bb1.iloc[i]['x1']
            x2=text_bb1.iloc[i]['x2']
            y1=text_bb1.iloc[i]['y1']
            y2=text_bb1.iloc[i]['y2']
            poly.append((x1,x2,y1,y2))
            polygons.append(Polygon([(x1, y2), (x1, y1), (x2, y1), (x2, y2)]))
            centres.append(((x1+x2)/2,(y1+y2)/2))
    
    new_poly = poly[:]
    return new_poly,poly,polygons,centres
    


# In[9]:


def set_polygons(centres):
    centre = []
    left = []
    right = []
    for i in range(len(centres)) :
        centre.append(1)
        left.append(1)
        right.append(1)
    return centre,left,right


# In[10]:


def compute_intensity(centre_index,left,right,thr,poly):
    intensity=[]
    
    y = int((poly[centre_index][2]+poly[centre_index][3])/2)
    
    #print(y)
    
    intensity = thr[y,left:right] 

    #for i in range(left,right,1):
        #intensity.append(thr[y,i])
        
    return intensity      


# In[11]:


def compute_ends(intensity):
    length = len(intensity)
    max_count = 0
    count = 0
    left =0
    right = 0
    for i in range(length):
        if(intensity[i]==0): 
            if(max_count<count):
                max_count = count
                left = i - count
                right = i-1
            count = 0
        elif(intensity[i]==255):
            count+=1
    if(max_count<count):
        max_count = count
        left = i - count
        right = i-1
    return left,right,length


# In[12]:


def compute_ave_intensity(centre_index,left,right,thr,poly):
    intensity=[]
    
    y1 , y2= poly[centre_index][2],poly[centre_index][3]
    
    #print(y)
    
    #box = thr[y1:y2,left:right] 
    
    

    for i in range(left,right,1):
        avg = np.mean(thr[y1:y2,i])
        intensity.append(avg)
        #intensity.append(thr[y,i])
        
    return intensity  


# In[13]:


def compute_ends_none_left(intensity):
    #length = len(intensity)
    length = len(intensity)
    max_count = 0
    count = 0
    left =0
    right = 0
    for i in range(length-1,-1,-1):
        if(intensity[i]==0): 
            if(max_count<count):
                max_count = count
                left = i + count
                right = i+1
                if(max_count>=50):
                    break
            count = 0
        elif(intensity[i]==255):
            count+=1
    if(max_count<count):
        max_count = count
        left = i + count
        right = i+1
        if(max_count>=50):
            return left,right,length
            
    return left,right,length
    


# In[14]:


def compute_ends_none_right(intensity):
    #length = len(intensity)
    length = len(intensity)
    max_count = 0
    count = 0
    left =0
    right = 0
    for i in range(length):
        if(intensity[i]==0): 
            if(max_count<count):
                max_count = count
                left = i - count
                right = i-1
                if(max_count>=50):
                    break
            count = 0
        elif(intensity[i]==255):
            count+=1
    if(max_count<count):
        max_count = count
        left = i - count
        right = i-1
        if(max_count>=50):
            return left,right,length
            
    return left,right,length
    


# In[15]:


def compute_ends_right(intensity):
    length = len(intensity)
    max_count = 0
    count = 0
    left =0
    right = 0
    for i in range(length):
        if(intensity[i]!=255.0): 
            if(max_count<count):
                max_count = count
                left = i - count
                right = i-1
                if(max_count>=50):
                    break
            count = 0
        elif(intensity[i]==255.0):
            count+=1
    if(max_count<count):
        max_count = count
        left = i - count
        right = i-1
        if(max_count>=50):
            return left,right,length
            
    return left,right,length


# In[16]:


def compute_ends_left(intensity):
    length = len(intensity)
    max_count = 0
    count = 0
    left = length
    right = 0
    for i in range(length-1,-1,-1):
        if(intensity[i]!=255.0): 
            if(max_count<count):
                max_count = count
                left = i + count
                right = i+1
                if(max_count>=50):
                    break
            count = 0
        elif(intensity[i]==255.0):
            count+=1
    if(max_count<count):
        max_count = count
        left = i + count
        right = i + 1
        if(max_count>=50):
            return left,right,length
    return left,right,length


# In[17]:


def correctbox_intensity(new_poly,poly,polygons,centres,thr):
    centre,left,right = set_polygons(centres)
    for i,point in enumerate(centres):
        if(centre[i]==1):
            if(right[i]==1):
                closest_right = closest_polygon(point[0],point[1],90,polygons)
                j_r = closest_right["closest_polygon"]
                if(j_r != None and left[j_r]==1):
                    intensity = compute_ave_intensity(i,poly[i][1],poly[j_r][0],thr,poly)
                    peak_left,peak_right,length = compute_ends_right(intensity)
                    intensity1 = compute_ave_intensity(j_r,poly[i][1],poly[j_r][0],thr,poly)
                    peak_left1,peak_right1,length1 = compute_ends_left(intensity1)
                    mid = (peak_left+peak_right)/2
                    left[j_r]=0
                    right[i]=0
                    new_poly[i] = new_poly[i][0], new_poly[i][1]+ peak_left,new_poly[i][2],new_poly[i][3]
                    new_poly[j_r] = new_poly[j_r][0]-(length1-peak_left1),new_poly[j_r][1],new_poly[j_r][2],new_poly[j_r][3]

                else:
                
                    intensity = compute_intensity(i,poly[i][1],thr.shape[1],thr,poly)
                    peak_left,peak_right,length = compute_ends_none_right(intensity)
                    mid = (peak_left+peak_right)/2
                
                    new_poly[i] = new_poly[i][0],new_poly[i][1]+peak_left,new_poly[i][2],new_poly[i][3]
                    
                    right[i]=0
                
                
            if(left[i]==1):
                closest_left = closest_polygon(point[0],point[1],270,polygons)
                j_l = closest_left["closest_polygon"]
               
            
                if(j_l!= None and right[j_l]==1):
                    intensity = compute_ave_intensity(i,poly[j_l][1],poly[i][0],thr,poly)
                    peak_left,peak_right,length = compute_ends_left(intensity)
                    intensity1 = compute_ave_intensity(j_l,poly[j_l][1],poly[i][0],thr,poly)
                    peak_left1,peak_right1,length1 = compute_ends_right(intensity1)
                    mid = (peak_left+peak_right)/2
                    right[j_l]=0
                    left[i]=0
                   
                    new_poly[i] = new_poly[i][0]-(length-peak_left),new_poly[i][1],new_poly[i][2],new_poly[i][3]
                    
                    new_poly[j_l] = new_poly[j_l][0],new_poly[j_l][1]+peak_left1,new_poly[j_l][2],new_poly[j_l][3] 
                    
                else:
                    intensity = compute_intensity(i,0,poly[i][0],thr,poly)
                    peak_left,peak_right,length= compute_ends_none_left(intensity)
                    mid = (peak_left+peak_right)/2
                    new_poly[i] = peak_left-1,new_poly[i][1],new_poly[i][2],new_poly[i][3]
                   
                    left[i]=0
                
        centre[i]=0
        
    return new_poly


# In[18]:


def merge_after_intensity(new_poly):
    x1=[]
    x2=[]
    y1=[]
    y2=[]
    x_c=[]
    y_c=[]
    polygons = []
    for i in new_poly:
        x1.append(i[0])
        x2.append(i[1])
        y1.append(i[2])
        y2.append(i[3])
        xc=int((i[0]+i[1])/2)
        yc=int((i[2]+i[3])/2)
        x_c.append(xc)
        y_c.append(yc)
        polygons.append(Polygon([(i[0],i[3]), (i[0],i[2]), (i[1],i[2]), (i[1],i[3])])) 
    x1 = pd.Series(x1)
    x2 = pd.Series(x2)
    y1 = pd.Series(y1)
    y2 = pd.Series(y2)
    x_c = pd.Series(x_c)
    y_c = pd.Series(y_c)
    new_df=pd.DataFrame(data=dict(x1=x1,x2=x2,y1=y1,y2=y2,x_c=x_c,y_c=y_c), index=x1.index)
   
    for i in new_df.index:
        if new_df.iloc[i]['x1'] != -1:
            close_r=closest_polygon(new_df.iloc[i]['x_c'],new_df.iloc[i]['y_c'],90,polygons)
            i_r=close_r["closest_polygon"]
            if (i_r != None) :
                while (i_r!=None) and ((0 < new_df.iloc[i_r]['x1']-new_df.iloc[i]['x2']  < 20)):
                    new_df.iloc[i]['x2']=new_df.iloc[i_r]['x2']
                    if (new_df.iloc[i_r]['y2'] > new_df.iloc[i]['y2']):
                        new_df.iloc[i]['y2']=new_df.iloc[i_r]['y2']
                    if (new_df.iloc[i_r]['y1'] < new_df.iloc[i]['y1']):
                        new_df.iloc[i]['y1']=new_df.iloc[i_r]['y1']
              
                    polygons[i]=Polygon([(new_df.iloc[i]['x1'],new_df.iloc[i]['y2']), (new_df.iloc[i]['x1'],new_df.iloc[i]['y1']), (new_df.iloc[i]['x2'],new_df.iloc[i]['y1']), (new_df.iloc[i]['x2'],new_df.iloc[i]['y2'])])
                    polygons[i_r]=Polygon([(5000,5000),(5000,4900),(5100,4900),(5100,5000)])
             
                    xt,yt,xb,yb=new_df.iloc[i]['x1'],new_df.iloc[i]['y1'],new_df.iloc[i]['x2'],new_df.iloc[i]['y2']
                    new_df.iloc[i_r]['x1']=-1
                    xc=(xb+xt)/2
                    yc=(yt+yb)/2
                    close_r=closest_polygon(xc,yc,90,polygons)
                    i_r=close_r["closest_polygon"]
   
    final_df=new_df[new_df['x1'] != -1]
    final_df.index= range(len(final_df.index))
    return final_df


# In[19]:



def extract_mask(img):


    gray_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) 


    thr = cv2.adaptiveThreshold(gray_img.copy(),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,-2)
    
   
    
    horizontal_img = thr.copy()
    vertical_img = thr.copy()

    size1 = int(img.shape[0]/5)
    size2 = int(img.shape[1]/5)
  

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (70,1))
    horizontal_img = cv2.erode(horizontal_img, kernel, iterations=1)
    horizontal_img = cv2.dilate(horizontal_img, kernel, iterations=1)


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,70))
    vertical_img = cv2.erode(vertical_img, kernel, iterations=1)
    vertical_img = cv2.dilate(vertical_img, kernel, iterations=1)

    mask_img = horizontal_img + vertical_img
    
    
    return gray_img,mask_img


# In[20]:


def compute_intensity_line(img):
    intensity=[]
    
    y = int(img.shape[0]/2)
    
    #print(y)
    
    intensity = img[y,:] 

    #for i in range(left,right,1):
        #intensity.append(thr[y,i])
        
    return intensity      


# In[21]:


def extract_borders1(intensity):
    borders_l=[]
    borders_r=[]
    for i in range(len(intensity)):
        if(intensity[i]==255 and i!=len(intensity)-1 and intensity[i+1]==0):
            borders_l.append(i)
        if(intensity[i]==0 and i!=len(intensity)-1 and intensity[i+1]==255):
            borders_r.append(i+1)
    return borders_l,borders_r


# In[22]:


def extract_borders(intensity):
    left =-1
    right = -1
    borders =[]
    if(intensity[0]==0):
            left = 0
    for i in range(len(intensity)):
        if(intensity[i]==0 and i!=0 and intensity[i-1]==255):
            if(left==-1):
                left = i-1
                right = -1
        if(i!=left and intensity[i]==0 and i!=len(intensity)-1 and intensity[i+1]==255):
            if(right==-1 and left!=-1):
                right = i+1
                borders.append((left,right))
                left = -1
    if(intensity[len(intensity)-1]==0 and left!=-1):
        right = len(intensity)-1
        borders.append((left,right))
    
    return borders  


# In[23]:


def remove_borders(gray_img,mask_img,borders): 
    for i in range(len(borders)):
        if(abs(borders[i][0]-borders[i][1])<=10):
            if(borders[i][1]==mask_img.shape[1]-1):
                mask_img[:,borders[i][0]:borders[i][1]+1]=255
            else:
                mask_img[:,borders[i][0]:borders[i][1]]=255
    
    img =  np.bitwise_or(gray_img,mask_img)
    
    return img


# In[24]:


def process_image(img_path):
    im = Image.open(img_path)
    im.save("/home/ansul/300/out30.jpg",dpi=(300,300))
    img = cv2.imread("/home/ansul/300/out30.jpg")
    return img    


# In[25]:


def frequent_prediction(predict):
    frequency =[]
    for i in range(len(predict)):
        count =1
        if(predict[i]!=""):
            for j in range(i,len(predict),1):
                if(predict[i]==predict[j]):
                    count+=1
        frequency.append(count)
            
    max_value = max(frequency)
    max_index = frequency.index(max_value)
    final = predict[max_index]
    return final


# In[26]:


def recognize_dir(path_crop1):
    for i,img_path in enumerate(glob.iglob(path_crop1)):
        
        #img = cv2.imread(img_path)
        file_name = img_path.split('/')[-1]
        
        #subprocess.call("python /home/crisp/Downloads/text-skew-correction/correct_skew.py --image {}".format(img_path),shell=True)
        
        
        
        for j in range(1,2,1):
            img = process_image(img_path)
            img = cv2.resize(img, None, fx=j,fy=j,interpolation=cv2.INTER_CUBIC)
           #Convert to gray
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

             #Aly dilation and erosion to remove some noise
            #kernel = np.ones((1, 1), np.uint8)
            #img = cv2.dilate(img, kernel, iterations=1)
            #img = cv2.erode(img, kernel, iterations=1)
            # Apply threshold to get image with only black and white
            #img = apply_threshold(img, method)
   
            #ret,thr = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

   
            result = pt.image_to_string(img, lang="eng",config='--psm 7')

            
            
        print(file_name + " " +result+"      "+finalauto)
        print("......................................................................")
        
       
        
        
        #with open("/home/crisp/Downloads/allcrop/before/po_no.txt", "a+") as text_file:
        #    text_file.write(file_name + " " +result+"   "+finalauto +"\n")


# In[27]:


#recognize_dir("/home/crisp/Downloads/allcrop/before/*.png")


# In[57]:



def recognize(img_path):
    #predict =[]
    img = cv2.imread(img_path)
    
    file_name = img_path.split('/')[-1]
    
        
        #cv2.imwrite("/home/crisp/Downloads/Test/train_data/{}.jpg".format(i+114),img)    
        
    #for j in range(1,,1):
    im = process_image(img_path)
    im = cv2.resize(img, None, fx=1,fy=1,interpolation=cv2.INTER_CUBIC)
   
    result = pt.image_to_string(im, lang="eng",config='--psm 7')
        #predict.append(result)
          
    #final = frequent_prediction(predict)
    finalauto = result
       
    if("|" in finalauto):
        finalauto = finalauto.replace("|", "")
     
    
    #print(file_name + "   " +result+"   "+finalauto)
    
    #print("....................................................")

    #with open("/home/crisp/Downloads/Test/after/out20_cleaned/recognised.txt", "a+") as text_file:
        #text_file.write(final + "    " +finalauto + "\n")
        
    return finalauto


# In[29]:


#recognize("/home/crisp/Downloads/Test/before/out21/58.jpg")


# In[30]:


#recognize("/home/crisp/Downloads/Test/before/out21/109.jpg")


# In[31]:


#recognize("/home/crisp/Downloads/Test/before/out21/114.jpg")


# In[32]:


#recognize("/home/crisp/Downloads/Test/before/out00/5.jpg")


# In[33]:


#recognize("/home/crisp/Downloads/allcrop/gst/jfh.jpg")


# In[34]:



def recognize_single(image_filename):
    file_name = image_filename.split('/')[-1]
    text = recognize(image_filename)
    return file_name,text
    


# In[63]:


os.environ['OMP_THREAD_LIMIT'] = '1'
def recognize_crop(cropimg_path):
    recognised = {}
    r=[]
    start=time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:

        image_files=glob.glob(cropimg_path)
   
       
        r.append(list(executor.map(recognize_single, image_files)))
    
    
    end=time.time()
    print("parllel",end-start)    
    
    for i in r[0]:
        recognised[i[0]]=i[1]
    return recognised

    
    


# In[36]:


def text2int(textnum, numwords={}):
    if not numwords:
        units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",]
        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        scales = ["hundred", "thousand", "lakh", "crores"]
        numwords["and"] = (1, 0)
        numwords["only"]=(1,0)
        numwords["paise"]=(1,0)
        for idx, word in enumerate(units):    numwords[word] = (1, idx)
        for idx, word in enumerate(tens):     numwords[word] = (1, idx * 10)
        for idx, word in enumerate(scales):
            if (idx ==0 or idx==1):
                numwords[word] = (10 ** ((idx * 3) or 2), 0)
            else:
                numwords[word] = (10 ** ((idx * 2)+ 1), 0)
   # print(numwords.keys())
    current = result = 0
    scales = ["hundred", "thousand", "lakh", "crores"]
    for word in textnum.split():
        
        if word not in numwords:
            
            return word
        else:
            if word in scales:
                if current != 0:
                    continue
                else :
                    current =1
            scale, increment = numwords[word]
            
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0
        #print(result,current)    
                
   
    return result + current


# In[37]:


def text2int2(textnum, numwords={}):
    if not numwords:
        units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",]
        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        scales = ["hundred", "thousand", "lakh", "crores"]
        numwords["and"] = (1, 0)
        numwords["only"]=(1,0)
        numwords["paise"]=(1,0)
        numwords["rupees"]=(1,0)
        numwords["inr"]=(1,0)
        numwords["indian"]=(1,0)
        numwords["paiseonl"]=(1,0)
        
        for idx, word in enumerate(units):    numwords[word] = (1, idx)
        for idx, word in enumerate(tens):     numwords[word] = (1, idx * 10)
        for idx, word in enumerate(scales):
            if (idx ==0 or idx==1):
                numwords[word] = (10 ** ((idx * 3) or 2), 0)
            else:
                numwords[word] = (10 ** ((idx * 2)+ 1), 0)

    current = result = 0
    for word in textnum.split():
        if (word =="lakhs") or (word =="lac") or (word =="lacs"):
            word ="lakh"
        if word not in numwords:
            return word
        scale, increment = numwords[word]
        #print("scale:",scale,"current",current,"increment",increment)
        current = current * scale + increment
        if scale > 100:
            result += current
            current = 0
       # print(current,result)
    return result + current


# In[38]:


def extract_list_amount(recognised,finalbox_df,image):

    out=[]
    all_text=''

    for key in recognised.keys():
    
    #file_name="out{}.jpg".format(i)
    #text=recognize(image_path)
        text = recognised[key]
    
        PERMITTED_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ " 
        clean_text = "".join(c for c in text if c in PERMITTED_CHARS)
        all_text=all_text + text
        if  (not re.match(r'\s+',clean_text)) and (clean_text != ""):
        
            text_int=text2int(clean_text.lower())
        
        else:
            text_int="invalid"
        if (text_int == "inr" or text_int == "indian" or text_int == "rupees" or str(text_int).isnumeric()):
            out.append(key.split('.')[0])



    if (len(out) == 1):        
        x1,y1,x2,y2=(finalbox_df.iloc[int(out[0])]['x1'],finalbox_df.iloc[int(out[0])]['y1'],finalbox_df.iloc[int(out[0])]['x2'],finalbox_df.iloc[int(out[0])]['y2'])
        roi=image.copy()[y1:y2,x1:x2]
        cv2.imwrite("/home/ansul/300/roi/roi0.jpg",roi)

    amount_list=[]    
    if (len(out) >= 2):
        for i in range(0,len(out)):
            x1,y1,x2,y2=(finalbox_df.iloc[int(out[i])]['x1'],finalbox_df.iloc[int(out[i])]['y1'],finalbox_df.iloc[int(out[i])]['x2'],finalbox_df.iloc[int(out[i])]['y2'])
            amount_list.append((x1,y1,x2,y2))
        amount_df=pd.DataFrame(amount_list,columns=['x1','y1','x2','y2'])
        l=len(amount_df.index)
    
        for i in amount_df.index:
   
             if (i != l-1 and (abs(amount_df.iloc[i]['y1']-amount_df.iloc[i+1]['y1']) < 20)):
                amount_df.iloc[i]['x2']=amount_df.iloc[i+1]['x2']
                if amount_df.iloc[i]['y1'] > amount_df.iloc[i+1]['y1'] :
                    amount_df.iloc[i]['y1'] = amount_df.iloc[i+1]['y1']
                if amount_df.iloc[i]['y2'] < amount_df.iloc[i+1]['y2'] :
                    amount_df.iloc[i]['y2']=amount_df.iloc[i+1]['y2']
                amount_df.iloc[i+1]['x1']=-1
        
        for i in amount_df.index:
            if amount_df.iloc[i]['x1'] != -1: 
                x1,y1,x2,y2=(amount_df.iloc[i]['x1'],amount_df.iloc[i]['y1'],amount_df.iloc[i]['x2'],amount_df.iloc[i]['y2'])
                roi=image.copy()[y1:y2,x1:x2]
           
                cv2.imwrite("/home/ansul/300/roi/roi{}.jpg".format(i),roi) 
            
    int_amt=[]            
    for i in glob.iglob("/home/ansul/300/roi/*.jpg"):    
    
        amount_img=cv2.imread(i)
        amount_text=recognize(i)
    
        clean_text2 = "".join(c for c in amount_text if c in PERMITTED_CHARS)
        clean_text2=clean_text2.lower()
        x=[m.start() for m in re.finditer('rupees', clean_text2)]
        if len(x) == 2:
            clean_text2=clean_text2[0:x[1]]
        number=text2int2(clean_text2)
        rup_and=re.search(r'.+?(RUPEES| and) (.+?) paise only',clean_text2)
    
        if rup_and:
            pais=rup_and.group(2)
            dec=text2int2(pais)
            number = number - dec
            number =number + dec*0.01
        else:
            number =number
        int_amt.append(number)
        
    return int_amt,all_text
    


# In[39]:


def invoice_no(recognised,finalbox_df,new_polygons):
    keys = ["Invoice No","Bill No","GST INV No","GST Invoice No","BILL NO","Transaction ID"]
    for key in recognised.keys():
        
        text = recognised[key]
        #text = pt.image_to_string(im, lang="eng",config='--oem 0 --psm 7')
        text = text.replace(".","")
        if('|' in text):
            text = text.replace('|',"")
            
        split = text.split(":")
        text = split[0]
        if(text in keys):
            if(len(split)>1):
                result = split[1]
                result = result.replace(" ","")
                return split[1]
            index = int(key.split('.')[0])
            x_c = (finalbox_df.iloc[index]['x1']+finalbox_df.iloc[index]['x2'])/2
            y_c = (finalbox_df.iloc[index]['y1']+finalbox_df.iloc[index]['y2'])/2
            right = closest_polygon(x_c,y_c,90,new_polygons)
            bottom = closest_polygon(x_c,y_c,0,new_polygons)
            right_i,bottom_i=right['closest_polygon'],bottom['closest_polygon']
            if(right_i!=None):
                right_text = recognised['{}.jpg'.format(right_i)]
                if(not (re.match(r'^[A-Za-z\s*]+$',right_text))):
                    right_text = right_text.replace(" ","")
                    return right_text
            if(bottom_i!=None):
                bottom_text=recognised['{}.jpg'.format(bottom_i)]
                if(not(re.match(r'^[A-Za-z\s*]+$',bottom_text))):
                    bottom_text = bottom_text.replace(" ","")
                    return bottom_text
            
            
    return " "


# In[40]:


def tot_amount(list_amt,all_text):
    try:
        tot_amt=max(list_amt)
    except:
        return " "
    tot_amt=str(tot_amt)
    PERMITTED_CHARS_NUMS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
    all_text= "".join(c for c in all_text if c in PERMITTED_CHARS_NUMS)
    
    #count=all_text.count(tot_amt)
    #if count != 0:
    return tot_amt


# In[41]:


def tax_amount(list_amt,all_text):
    taxes ={"CGST":"","SGST":""}
    
    if len(list_amt) > 1:
        tax_amt=min(list_amt)
        cgst=tax_amt/2
        sgst=tax_amt/2
        tot_amt=max(list_amt)
       # cgst_perc = int((cgst/(tot_amt-tax_amt))*100)
        #sgst_perc = int((cgst/(tot_amt-tax_amt))*100)
        taxes["CGST"]=cgst
        taxes["SGST"]=sgst
    else:
        return taxes
   
    #print(cgst)
    #cgst=str(cgst)
    #sgst=str(sgst)
    #count=all_text.count(cgst)
    #if count > 2:
        #return cgst
    
    return taxes


# In[42]:


def pan_no(recognised):
    pan_dict={"pan_tvs":"","supplier":""}
    pan_list=[]
    pan_tvs="AADCT0724A"
    for key in recognised.keys():
        text=recognised[key]
        pan=re.findall(r'[A-Za-z]{5}\d{4}[A-Za-z]{1}',text)
        if pan:
            pan_list.append(pan)   
    for pan in pan_list:
        if pan[0] == pan_tvs:
            pan_dict["pan_tvs"] = pan[0]    
        else:
            pan_dict["supplier"] = pan[0]
    return pan_dict


# In[43]:


def gst_no(recognised):
    pan_tvs="AADCT0724A"
    gst_dict={"gst_tvs":"","supplier":""}
    gst_list=[]
    for key in recognised.keys():
        text=recognised[key]
        gst=re.findall(r'(?:[0]{1}[1-9]{1}|[1-2]{1}[0-9]{1}|[3]{1}[0-7]{1})(?:[a-zA-Z]{5}[0-9]{4}[a-zA-Z]{1}[1-9a-zA-Z]{1}[zZ]{1}[0-9a-zA-Z]{1})+',text)
        if gst:
            gst_list.append(gst)
    for gst in gst_list:
        if pan_tvs in gst[0]:
            gst_dict["gst_tvs"]=gst[0]
        else:
            gst_dict["supplier"]=gst[0]
    return gst_dict


# In[44]:


def invoice_date(recognised):
    dates=[]
    for key in recognised.keys():
        text=recognised[key]
        form1=re.findall(r'\d{1,2}[\.]\d{1,2}[\.]\d{2,4}',text)
        form2=re.findall(r'\d{1,2}[\s]\d{1,2}[\s]\d{2,4}',text)
        form3=re.findall(r'\d{1,2}[\/]\d{1,2}[\/]\d{2,4}',text)
        form4=re.findall(r'\d{1,2}[-]\d{1,2}[-]\d{2,4}',text)
        form5=re.findall(r'\d{1,2}(th)?[\.][a-zA-Z]{3,9}[\.]\d{2,4}',text)
        form6=re.findall(r'\d{1,2}(th)?[\s][a-zA-Z]{3,9}[\s]\d{2,4}',text)
        form7=re.findall(r'\d{1,2}(th)?[\/][a-zA-Z]{3,9}[\/]\d{2,4}',text)
        form8=re.findall(r'\d{1,2}(th)?[-][a-zA-Z]{3,9}[-]\d{2,4}',text)
      
        if form1:
            dates.append(form1)
        if form2:
            dates.append(form2)
        if form3:
            dates.append(form3)
        if form4:
            dates.append(form4)
        if form5:
            dates.append(form5)
        if form6:
            dates.append(form6)
        if form7:
            dates.append(form7)
        if form8:
            dates.append(form8)
      
    converted=[]
    for date in dates:
        try:
            converted.append(parse(date, dayfirst=True))
        except:
            continue
    if(converted):
        return max(converted).strftime("%Y-%m-%d %H:%M:%S").split(' ')[0]
    else:
        return " "


# In[45]:


def po_no():
    
    return ""


# In[46]:


def po_date():
    
    return ""


# In[47]:


def supplier_name():
    
    return ""


# In[48]:


def fields_json(store_json_path,recognised,int_amt,all_text,finalbox_df,new_polygons):
    field_dict = {}
    field_dict["Invoice_No"]=invoice_no(recognised,finalbox_df,new_polygons)
    field_dict["Invoice_Date"]=invoice_date(recognised)
    field_dict["Supplier_Name"]=supplier_name()
    field_dict["Amount"]=tot_amount(int_amt,all_text)
    field_dict["CGST"]=dict(tax_amount(int_amt,all_text))["CGST"]
    field_dict["SGST"]=dict(tax_amount(int_amt,all_text))["SGST"]
    field_dict["GST_No_TVS"]=dict(gst_no(recognised))["gst_tvs"]
    field_dict["GST_No_SUPPLIER"]=dict(gst_no(recognised))["supplier"]
    field_dict["PAN_No_SUPPLIER"]=dict(pan_no(recognised))["supplier"]
    field_dict["PAN_NO_TVS"]=dict(pan_no(recognised))["pan_tvs"]
    field_dict["PO_No"]=po_no()
    field_dict["PO_Date"]=po_date()
    with open(store_json_path,"w+") as f:
        json.dump(field_dict,f)
    
    return field_dict


# In[61]:


def main(image_path):
    image_name = image_path.split("/")[-1].split(".")[0]
    clean_image(image_path,"/home/ansul/EAST/merge/")
    east_detect()
    image,thr = read_image(image_name)
    cordinate,polygons = df_org("/home/ansul/EAST/res/{}.txt".format(image_name))
    cordinate = merge_overlap(cordinate,polygons)
    new_poly,poly,polygons,centres = box_inten(cordinate)
    new_poly = correctbox_intensity(new_poly,poly,polygons,centres,thr)
    finalbox_df = merge_after_intensity(new_poly)
    new_polygons=[]
    out = image.copy()
    for i in finalbox_df.index:
        x1,x2,y1,y2 = finalbox_df.iloc[i]['x1'],finalbox_df.iloc[i]['x2'],finalbox_df.iloc[i]['y1'],finalbox_df.iloc[i]['y2']
        roi=image[y1:y2,x1:x2]
        cv2.imwrite("/home/ansul/300/crop/{}.jpg".format(i),roi)
        new_polygons.append(Polygon([(x1, y2), (x1, y1), (x2, y1), (x2, y2)]))
        cv2.rectangle(out,(x1,y1),(x2,y2),(0,255,0),1)
    cv2.imwrite("/home/ansul/300/Int{}.jpg".format(image_name,i),out)
    for i,img_path in enumerate(glob.iglob("/home/ansul/300/crop/*.jpg")):
        crop_img_name=img_path.split('/')[-1]
        img = cv2.imread(img_path)
        gray_img,mask_img=extract_mask(img)
        intensity = compute_intensity_line(mask_img)
        borders = extract_borders(intensity)
        final_img = remove_borders(gray_img,mask_img,borders)
        cv2.imwrite("/home/ansul/300/{}".format(crop_img_name),final_img)
    recognised = recognize_crop("/home/ansul/300/*.jpg")
    int_amt,all_text = extract_list_amount(recognised,finalbox_df,image)
    json_path ="/home/ansul/{}.json".format(image_name)
    fields=fields_json(json_path,recognised,int_amt,all_text,finalbox_df,new_polygons)
    return fields
    
    
    
    
    


# In[64]:

def integrate(path):
    start = time.time()
    field=main(path)
    end = time.time()-start
    print(end)
    return field
