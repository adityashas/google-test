from flask import Flask
from flask_mongoengine import MongoEngine
from flask import render_template, request, json, jsonify, Response, redirect, flash, url_for, session
import flask.scaffold
flask.helpers._endpoint_from_view_func = flask.scaffold._endpoint_from_view_func
from flask_restplus import Resource
from flask_restplus import Api
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField, IntegerField
from wtforms.validators import DataRequired, Email, Length, EqualTo, ValidationError
from bson import ObjectId # For ObjectId to work
import os#, cv2, secrets
#from pymongo import MongoClient
import numpy as np
import ctypes
import validators
#from google.cloud import storage
from flask import Flask
#from flask_ngrok import run_with_ngrok


from taxonomy_mapper import TaxonomyPredictor
#secret_key = secrets.token_hex(16)

api = Api()
app = Flask(__name__)
#run_with_ngrok(app)

#app.config['SECRET_KEY'] = secret_key

#app.config['MONGODB_SETTINGS'] = {
    #'db': 'ninademo',
    #'host': 'localhost',
  #  #'port': 27017
#}


#db = MongoEngine()
#db.init_app(app)
#api.init_app(app)

#client = MongoClient("mongodb://127.0.0.1:27017") #host uri
#dbs = client.ninademo    #Select the database
#todos = dbs.urls #Select the collection name


#from werkzeug.security import generate_password_hash, check_password_hash

############################### Forms Page ######################################



############################### Home Login/Logout/Register Page ##########################





###################################### Text or Url Query Input Page ##########
@app.route("/", methods=['POST','GET'])
def url():
        global name
        if request.method == 'GET':               
            name = request.args.get("name")          
            if name is None:   
                return render_template('url.html') 
            if not validators.url(name):
                flash("Sorry, something went wrong.","danger")
                return render_template('user.html')
            return redirect(url_for('predict'))
        else: 
            name = request.form['name']
            if not validators.url(name):
                flash("Sorry, something went wrong.","danger")
                return render_template('user.html')
            return redirect(url_for('predict'))

@app.route("/text", methods=['POST','GET'])
def text():
     global name
     if request.method == 'GET':                   
            name = request.args.get("name")        
            if name is None:
                return render_template('text.html')
            if validators.url(name):
                flash("Sorry, something went wrong.","danger")
                return render_template('user.html')
            return redirect(url_for('predict'))
     else: 
            name = request.form['name']
            if validators.url(name):
                flash("Sorry, something went wrong.","danger")
                return render_template('user.html')
            return redirect(url_for('predict'))

##################### Prediction Starts Here ########################
import pandas as pd

csv_file_path = 'gs://seshaditya-datastore/dataset_v3.1.1.csv'

df=pd.read_csv(csv_file_path, encoding='ISO-8859-1')

#df = pd.read_csv("dataset_v3.1.1.csv", encoding='ISO-8859-1')

@app.route('/predict')
def predict():    
    if name is None:
        flash("Oops!! you have entered a wrong query.","danger") 
        return redirect(url_for('user'))
    else:  
        tp = TaxonomyPredictor(name)
        preds = tp.taxonomy_predictor()
        desc = preds['description']
        keys = preds['keywords']
        longkeys = preds['long keywords']
        tax = preds['taxonomy']
        tdict = {}
        for i in range(len(tax)):
            tdict[tax[i]] = i + 1
        urllist = df['url'].to_list()
        urldict = {}
        n = 0
        for i in range(len(urllist)):
            if tax[0] == df['class'][i] or tax[1] == df['class'][i]:
                n = n+1
                urldict[df['url'][i]] = n
        
        return render_template("predict.html",title="prediction", desc=desc,keys=keys, longkeys=longkeys, tdict = tdict, urldict = urldict)
app.run()
