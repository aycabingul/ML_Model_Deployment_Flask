import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template

# Load ML models
model6 = pickle.load(open('models\ort6.pickle', 'rb'))
model7 = pickle.load(open('models\ort7.pickle', 'rb'))
model9 = pickle.load(open('models\ort9.pickle', 'rb')) 
model10 = pickle.load(open('models\ort10.pickle', 'rb'))
model11 = pickle.load(open('models\ort11.pickle', 'rb'))

# Create application
app = Flask(__name__)

#DEBUG MODE#########
app.config['ENV'] = 'development'
app.config['DEBUG'] = True
app.config['TESTING'] = True
############


# Bind home function to URL
@app.route('/', methods=['POST', 'GET'])
def home():
    return render_template('homepage.html')

# Bind predict function to URL
@app.route('/ort6.html', methods=['POST', 'GET'])
def ort6():
    return render_template('ort6.html')

@app.route('/predict6', methods=['POST', 'GET'])
def predict6():
    
    # Put all form entries values in a list 

    genel_list = [int(i) for i in request.form.values()]
    comment_list = []
    comment_list.extend(genel_list[5:10])

    indices = [5, 6, 7, 9]
    girdiler = [i for j, i in enumerate(genel_list) if j not in indices]
    
    # Convert features to DataFrame
    data_girdi=pd.DataFrame(columns=['Bsag','Aoz','Abirlikte','Aogrenim','Bogrenim','ders_calisma','ozel_kurs','sosyal_kulturel','ort5'])
    data_girdi = data_girdi.append({'Bsag': girdiler[0],'Aoz': girdiler[1], 'Abirlikte': girdiler[2], 'Aogrenim': girdiler[3], 'Bogrenim': girdiler[4], 'ders_calisma': girdiler[5],'ozel_kurs': girdiler[6], 'sosyal_kulturel': girdiler[7],
                                'ort5': girdiler[8]}, ignore_index=True)

    #data_girdi=data_girdi.iloc[:,:].astype("int64")

    # Predict features
    prediction = model6.predict(data_girdi)
    
    output = int(prediction)
    
    # Check the output values and retrive the result with html tag based on the value

    return render_template('ort6.html', 
                           result = output,
                           uyku_comment = comment_list[0],
                           internet_comment = comment_list[1],
                           televizyon_comment = comment_list[2],
                           ders_calisma_comment = comment_list[3],
                           okul_dyk_comment = comment_list[4]
                           )

# Bind predict function to URL
@app.route('/ort7.html', methods=['POST', 'GET'])
def ort7():
    return render_template('ort7.html')

@app.route('/predict7', methods=['POST', 'GET'])
def predict7():
    
    # Put all form entries values in a list 
    genel_list = [int(i) for i in request.form.values()]
    comment_list = []
    comment_list.extend(genel_list[1:4])
    comment_list.extend(genel_list[5:7])

    indices = [1, 3, 6]
    girdiler = [i for j, i in enumerate(genel_list) if j not in indices]
    
    # Convert features to DataFrame
    data_girdi=pd.DataFrame(columns=['Bsag','internet','oyun','ders_calisma','ort5','ort6'])
    data_girdi = data_girdi.append({'Bsag': girdiler[0], 'internet': girdiler[1], 
                                'oyun': girdiler[2] ,'ders_calisma': girdiler[3], 
                                'ort5': girdiler[4], 'ort6': girdiler[5]}, ignore_index=True)

    data_girdi=data_girdi.iloc[:,:].astype("int64")

    # Predict features
    prediction = model7.predict(data_girdi)
    
    output = int(prediction)
    
    # Check the output values and retrive the result with html tag based on the value

    return render_template('ort7.html', 
                           result = output,
                           uyku_comment = comment_list[0],
                           internet_comment = comment_list[1],
                           televizyon_comment = comment_list[2],
                           ders_calisma_comment = comment_list[3],
                           okul_dyk_comment = comment_list[4])


# Bind predict function to URL
@app.route('/ort9.html', methods=['POST', 'GET'])
def ort9():
    return render_template('ort9.html')

@app.route('/predict9', methods=['POST', 'GET'])
def predict9():
    
    # Put all form entries values in a list 
    genel_list = [int(i) for i in request.form.values()]
    comment_list = []
    comment_list.extend(genel_list[2:7])

    indices = [2, 4, 5]
    girdiler = [i for j, i in enumerate(genel_list) if j not in indices]
    
    # Convert features to DataFrame
    data_girdi=pd.DataFrame(columns=['Asag','Aogrenim','internet','okul_dyk','lgs_puani','turkce9','mat9'])
    data_girdi = data_girdi.append({'Asag': girdiler[0],
                                'Aogrenim': girdiler[1],
                                'internet': girdiler[2],
                                'okul_dyk': girdiler[3],
                                'lgs_puani': girdiler[4],
                                'turkce9': girdiler[5],
                                'mat9': girdiler[6]},
                                ignore_index=True)

    data_girdi=data_girdi.iloc[:,:].astype("int64")

    # Predict features
    prediction = model9.predict(data_girdi)
    
    output = int(prediction)
    
    # Check the output values and retrive the result with html tag based on the value

    return render_template('ort9.html', 
                           result = output,
                           uyku_comment = comment_list[0],
                           internet_comment = comment_list[1],
                           televizyon_comment = comment_list[2],
                           ders_calisma_comment = comment_list[3],
                           okul_dyk_comment = comment_list[4])


# Bind predict function to URL
@app.route('/ort10.html', methods=['POST', 'GET'])
def ort10():
    return render_template('ort10.html')

@app.route('/predict10', methods=['POST', 'GET'])
def predict10():
    
    # Put all form entries values in a list 
    genel_list = [int(i) for i in request.form.values()]
    comment_list = []
    comment_list.extend(genel_list[2:7])

    indices = [2, 4]
    girdiler = [i for j, i in enumerate(genel_list) if j not in indices]
    
    # Convert features to DataFrame
    data_girdi=pd.DataFrame(columns=['cinsiyet','Boz','internet','ders_calisma','okul_dyk','sosyal_kulturel','ortaokul_turu',"lgs_puani",'ort9'])
    data_girdi = data_girdi.append({'cinsiyet': girdiler[0],
                                'Boz': girdiler[1],
                                'internet': girdiler[2],
                                'ders_calisma': girdiler[3],
                                'okul_dyk': girdiler[4],
                                'sosyal_kulturel': girdiler[5],
                                'ortaokul_turu': girdiler[6],
                                'lgs_puani': girdiler[7],
                                'ort9': girdiler[8]},
                                ignore_index=True)

    data_girdi=data_girdi.iloc[:,:].astype("int64")

    # Predict features
    prediction = model10.predict(data_girdi)
    
    output = int(prediction)
    
    # Check the output values and retrive the result with html tag based on the value

    return render_template('ort10.html', 
                           result = output,
                           uyku_comment = comment_list[0],
                           internet_comment = comment_list[1],
                           televizyon_comment = comment_list[2],
                           ders_calisma_comment = comment_list[3],
                           okul_dyk_comment = comment_list[4])

# Bind predict function to URL
@app.route('/ort11.html', methods=['POST', 'GET'])
def ort11():
    return render_template('ort11.html')

@app.route('/predict11', methods=['POST', 'GET'])
def predict11():
    
    # Put all form entries values in a list 
    genel_list = [int(i) for i in request.form.values()]
    comment_list = []
    comment_list.extend(genel_list[2:7])

    indices = [2, 3, 4]
    girdiler = [i for j, i in enumerate(genel_list) if j not in indices]
    
    # Convert features to DataFrame
    data_girdi=pd.DataFrame(columns=["cinsiyet","ABayri","okul_dyk","ders_calisma","sosyal_kulturel","turkce9","ort10"])
    data_girdi = data_girdi.append({'cinsiyet': girdiler[0],                              
                                'ABayri': girdiler[1],
                                'okul_dyk': girdiler[2],
                                'ders_calisma': girdiler[3],
                                'sosyal_kulturel': girdiler[4],
                                'turkce9': girdiler[5],
                                'ort10': girdiler[6]},
                                ignore_index=True)

    data_girdi=data_girdi.iloc[:,:].astype("int64")

    # Predict features
    prediction = model11.predict(data_girdi)
    
    output = int(prediction)
    
    # Check the output values and retrive the result with html tag based on the value

    return render_template('ort11.html', 
                           result = output,
                           uyku_comment = comment_list[0],
                           internet_comment = comment_list[1],
                           televizyon_comment = comment_list[2],
                           ders_calisma_comment = comment_list[3],
                           okul_dyk_comment = comment_list[4])

if __name__ == '__main__':
#Run the application
    app.run()