from flask import Flask,request
from main import generateAI
import pickle
generateAI()
ai=pickle.load(open('model.pkl','rb')) #unpickling

app=Flask(__name__)

#for APIs we use decorater @,and decorater function(def ) should follow immediately after decorater(@)
@app.route('/')
def home():
    return('AI model server is running')

#main API
@app.route('/predict',methods=['GET']) # '/predict':-decorater name 
def predict():
    temp=request.args.get('temp') #temp :input value,request-IoT takes through requests
    temp=float(temp)              #input always return str ,convert to float
    data=[[temp]]                 #preparing data i.e. in 2D array [[]]
    result=ai.predict(data)       #ai predicts an array
    #api always have string format as return type
    result=result[0]              #result[0] is str:on/off
    return (result)

#starting the server
if(__name__=="__main__"):
    #host:IPv4 i.e.a.b.c.d,0.0.0.0 means take respective websites/machine's IP address
    app.run(host='0.0.0.0',port=5000,debug=True) 





