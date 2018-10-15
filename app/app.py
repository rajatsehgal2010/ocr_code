from flask import Flask, render_template,request,jsonify,request
import glob2
import cv2
import pandas as pd
from PIL import Image
import pytesseract as pt
import argparse
import re
from bs4 import BeautifulSoup

from matplotlib import pyplot as plt
import pdf2image

from werkzeug.utils import secure_filename
from flask import Flask, redirect, render_template, request, session, url_for
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class,DOCUMENTS,TEXT

import os
import xlwt
import int_ocr
app = Flask(__name__)
dropzone = Dropzone(app)
from random import randint


# app.config['SECRET_KEY'] = 'supersecretkeygoeshere'

# # Dropzone settings
# app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
# app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
# app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*, .pdf, .txt'
# app.config['DROPZONE_REDIRECT_VIEW'] = 'extract'

# # Uploads settings
# app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/uploads'


# print app.config['UPLOADED_PHOTOS_DEST']
# # files = UploadSet('photos', IMAGES)
# # configure_uploads(app, files)
# # patch_request_class(app)  # set maximum file size, default is 16MB

app.config['SECRET_KEY'] = 'supersecretkeygoeshere'

# Dropzone settings
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*, .pdf, .txt'
app.config['DROPZONE_REDIRECT_VIEW'] = 'extract'

# Uploads settings
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/static/img/'

uploadSetFiles = UploadSet('photos', IMAGES+DOCUMENTS+TEXT)
configure_uploads(app, uploadSetFiles)
patch_request_class(app)  # set maximum file size, default is 16MB




def extractbb(bb):
	coords_w = {'x':[],'y':[],'w':[],'h':[]}
	for i in bb:
		#print(i)
		X = i['title'].split(';')[0].split()
		x,y,w,h =(int(X[1]),int(X[2]),(int(X[3])-int(X[1])),(int(X[4])-int(X[2])))
		#coords = {'x':[],'y':[],'w':[],'h':[]}
		coords_w['x'].append(x)
		coords_w['y'].append(y)
		coords_w['w'].append(w)
		coords_w['h'].append(h)
	return pd.DataFrame(coords_w)


# In[3]:


def drawbboxes(bb,im,path):
	
	for i in bb.index:
		x, y, w, h = (bb.iloc[i]['x'],bb.iloc[i]['y'],bb.iloc[i]['w'],bb.iloc[i]['h'])
 
	
		roi = im[y:y+h, x:x+w]
	
		#path = ('/home/aditya/Downloads/AA/{}.jpg').format(i)
	
		#cv2.imwrite(path,roi)
 
	
		cv2.rectangle(ima,(x,y),( x + w, y + h ),(0,255,0),2)
		#cv2.waitKey(0)
		
	cv2.imwrite(path,im)


# In[4]:


def company(areas):
	

	x = areas[0].text

	return x.strip()


# In[5]:


def billno(areas):
	
	for i in areas:
		
		x = i.text.strip()
		result=re.search(r'Bill',x)
		if(result!=None):
			i=result.start()
			j=result.end()
			if(i==0 and j==4):
				return x
			


# In[6]:


def date(areas):
	
	for i in areas: 
		x = i.text.strip()
		result=re.search(r'Date',x)
		if(result!=None):
			return x


# In[7]:




def receiver(lines):
	x =lines[4].text
	return x.strip().replace('\n','')


		


# In[8]:


def text2int(textnum, numwords={}):
	if not numwords:
		units = [
		"zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
		"nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
		"sixteen", "seventeen", "eighteen", "nineteen",]
		tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
		scales = ["hundred", "thousand", "lakh", "crores"]
		numwords["and"] = (1, 0)
		for idx, word in enumerate(units):    numwords[word] = (1, idx)
		for idx, word in enumerate(tens):     numwords[word] = (1, idx * 10)
		for idx, word in enumerate(scales):
			if (idx ==0 or idx==1):
				numwords[word] = (10 ** ((idx * 3) or 2), 0)
			else:
				numwords[word] = (10 ** ((idx * 2)+ 1), 0)

	current = result = 0
	for word in textnum.split():
		if word not in numwords:
			raise Exception("Illegal word: " + word)

		scale, increment = numwords[word]
		current = current * scale + increment
		if scale > 100:
			result += current
			current = 0

	return result + current


# In[9]:



def total(areas):
	for i in areas:
		x=i.text.strip()
		result=re.search(r'INR',x)
		if(result != None):
			tot = x.replace('\n',' ').split('R')
			s = text2int(tot[1].lower())
			return tot[0]+'R '+str(s)
	


# In[10]:


def ph_fax(lines):
	ph_fa=[]
	for i in lines:
		x=i.text.strip()
		result=re.search(r'Ph:',x)
		if(result != None):
			y=x.split(',')
			ph_fa.append(y[0])
			ph_fa.append(y[1])
	return ph_fa


# In[11]:


def email(lines):
	for i in lines:
		x=i.text.strip()
		result=re.search(r'Email:',x)
		if(result != None):
			return x


# In[12]:


def receiver_address(areas):
	
	address= areas[4].text

	return address.strip().replace('\n',' ')


# In[13]:


def address_company(area):
	x=area[-3].text.strip()
	return x


def extractfields(path):
	
	pt.pytesseract.run_tesseract(path,'imgout',extension="hocr",lang=None,config="hocr")

	
	soup = BeautifulSoup(open('imgout.hocr'),'html.parser')

	lines = soup.findAll('span', attrs={'class':'ocr_line'})

	words = soup.findAll('span', attrs={'class':'ocrx_word'})

	areas = soup.findAll('div', attrs={'class':'ocr_carea'})
	
	Company =company(areas)
	
	BillNo = billno(areas)
	
	Date = date(areas)
	
	recipient=receiver(lines)
	
	tot = total(areas)
	
	Ph,Fax = ph_fax(lines)
	
	Email = email(lines)
	
	address_comp = address_company(areas)
	
	rec_address = receiver_address(areas)
	
	result ={"company":Company,"billno":BillNo,"date":Date,"receiver":recipient,"total":tot,"phone":Ph,"fax":Fax,"email":Email,"address_comp":address_comp,"receiveraddress":rec_address}
	
	return result


global counter

counter=0

images =list(glob2.iglob("static/img/*.jpg"))
print (images)






@app.route('/', methods=['GET', 'POST'])
def index():
    
    # set session for image results
    if "file_urls" not in session:
        session['file_urls'] = []
    # list to hold our uploaded image urls
    file_urls = session['file_urls']

    # handle image upload from Dropszone
    if request.method == 'POST':
        file_obj = request.files
        for f in file_obj:
            file = request.files.get(f)
            n = file.filename
            if n.endswith('.pdf'):
            	print("PDF Found")
            	file.save(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], file.filename))
            	#file.save(file.filename)

            print (n)
            # save the file with to our photos folder
            filename = uploadSetFiles.save(
                file,
                name=file.filename    
            )

            # append image urls
            file_urls.append(uploadSetFiles.url(filename))
            
        session['file_urls'] = file_urls
        return "uploading..."
    convertpdftoimage()
    # return dropzone template on GET request    
    return render_template('index.html')


# @app.route('/results')
# def results():
    
#     # redirect to home if no images to display
#     if "file_urls" not in session or session['file_urls'] == []:
#         return redirect(url_for('index'))
        
#     # set the file_urls and remove the session variable
#     file_urls = session['file_urls']
#     session.pop('file_urls', None)
    
#     return render_template('results.html', file_urls=file_urls)
def convertpdftoimage():
		for i,pages in enumerate(glob2.iglob('uploads/*.pdf')):
		   print(pages)
		   pg = pdf2image.convert_from_path(pages, 300)
		   os.remove(pages)
		   for j in range(len(pg)):
		       pg[j].save('uploads/skp{}{}.jpg'.format(i,j), 'JPEG')

@app.route('/extract',methods=['GET'])
def extract():
	images =list(glob2.iglob("static/img/*.*"))
	print(images)

	

	return render_template("extract.html",image_path=images[counter])
	#return "success"

def directoryM(path,resp):
	Iname = path.split('/')[2].split('.')[0]
	path = 'uploads/machine_response/'+Iname
	# if not os.path.isfile(path):
	# 	os.mkdir(path)
	#uuid = randint(0, 99)
	print("Path is :::: ",path)
	book = xlwt.Workbook(encoding="utf-8") 
	sheet1 = book.add_sheet("sheet1")
	colunm_count = 0
	data = resp
	print("datatagadghsbfksebnfkjrngkjrnf:",data)
	for title, value in data.items():
	    sheet1.write(0, colunm_count, title)
	    sheet1.write(1, colunm_count, value)
	    colunm_count += 1
	    file_name = path+".xls"%()
	    book.save(file_name)
	return "Done"

@app.route('/uploaded_images',methods=['GET'])
def showfiles():
	images =list(glob2.iglob("static/img/*.*"))
	print(images)
	return render_template("uploaded_files.html",Image=images)

@app.route('/custom',methods=['POST'])
def directoryU():
	path = images[counter]
	Invoice_No = request.form['Invoice_No']
	Invoice_Date = request.form['Invoice_Date']
	Supplier_Name = request.form['Supplier_Name']
	Amount = request.form['Amount']
	CGST = request.form['CGST']
	SGST = request.form['SGST']
	GST_No_TVS = request.form['GST_No_TVS']
	GST_No_SUPPLIER = request.form['GST_No_SUPPLIER']
	PAN_No_SUPPLIER = request.form['PAN_No_SUPPLIER']
	PAN_NO_TVS = request.form['PAN_NO_TVS']
	PO_No = request.form['PO_No']
	PO_Date = request.form['PO_Date']
	sample = {"Invoice_No": Invoice_No, "Invoice_Date": Invoice_Date, "Supplier_Name": Supplier_Name, "Amount": Amount, "CGST": CGST, "SGST": SGST, "GST_No_TVS": GST_No_TVS, "GST_No_SUPPLIER": GST_No_SUPPLIER, "PAN_No_SUPPLIER": PAN_No_SUPPLIER, "PAN_NO_TVS": PAN_NO_TVS, "PO_No": PO_No, "PO_Date": PO_Date}

	#{"Invoice_No","Invoice_Date","Supplier_Name",Amount","CGST","SGST","GST_No_TVS","GST_No_SUPPLIER","PAN_No_SUPPLIER","PAN_NO_TVS","PO_No","PO_Date"}
	

	print("In custom : ",sample)
	Iname = path.split('/')[2].split('.')[0]
	path = 'uploads/user_response/'+Iname
	# if not os.path.isfile(path):
	# 	os.mkdir(path)
	#uuid = randint(0, 99)
	book = xlwt.Workbook(encoding="utf-8") 
	sheet1 = book.add_sheet("sheet1")
	colunm_count = 0
	data = sample
	for title, value in data.items():
	    sheet1.write(0, colunm_count, title)
	    sheet1.write(1, colunm_count, value)
	    colunm_count += 1
	    file_name = path+".xls"%()
	    book.save(file_name)
	return "Done"








#d.split('/')[2].split('.')[0]

@app.route('/parse',methods=['POST'])
def parse():
	global counter
	data=request.get_json()
	
	path=data['path']
	# result = extractfields(path)
	print ("Path : ",path)
	#return jsonify(result)
	#return path
	#field = int_ocr.integrate("/home/ansul/EAST/merge/out00.jpg")
	print (images[counter])
	sample  = {"Invoice_No": "HKS/118/2017-18", "Invoice_Date": " ", "Supplier_Name": "", "Amount": "162000", "CGST": 17718.75, "SGST": 17718.75, "GST_No_TVS": "", "GST_No_SUPPLIER": "", "PAN_No_SUPPLIER": "ADCTO0724A", "PAN_NO_TVS": "", "PO_No": "", "PO_Date": ""}
	directoryM(images[counter],sample)
	return jsonify(sample)


@app.route('/next',methods=['GET'])
def next():
	global counter
	images=list(glob2.iglob("static/img/*.*"))
	if counter>=len(images)-1:
		counter=0
	else:
		counter+=1
	print (counter,len(images))
	print (counter,"  ",images[counter])
	return render_template("extract.html",image_path=images[counter])


if __name__ == '__main__':
	app.run('0.0.0.0',port=5051)
	app.debug = True
