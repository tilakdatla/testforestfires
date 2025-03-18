from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

ridge_model=pickle.load(open('ridge.pkl','rb'))
standard_scalar=pickle.load(open("scaler.pkl",'rb'))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata",methods=['GET','POST'])
def helper():
    if request.method=='POST':
        Temperature=float(request.form.get("Temperature"))
        rh = float(request.form.get("RH"))
        ws = float(request.form.get("Ws"))
        rain = float(request.form.get("Rain"))
        ffmc = float(request.form.get("FFMC"))
        dmc = float(request.form.get("DMC"))
        isi = float(request.form.get("ISI"))
        classes = request.form.get("Classes")  # Assuming string input
        region = request.form.get("Region") 
        
        new_data=standard_scalar.transform([[Temperature,rh,ws,rain,ffmc,
                                    dmc,isi,classes,region]])
        result=ridge_model.predict(new_data)
        return render_template("home.html",result=result[0])
    else:
        return render_template("home.html")
