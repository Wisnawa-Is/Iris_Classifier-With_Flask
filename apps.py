#!flask/bin/python
from flask import Flask, jsonify, request
import pickle
app = Flask(__name__)

@app.route('/todo/api/v1.0/tasks', methods=['GET'])
def get_tasks():
        #program inference ML/DL
        loaded_model = pickle.load(open('model.sav', 'rb'))
        #program input dari frontend
        SepalLengthCm = float(request.args.get('SepalLengthCm'))
        SepalWidthCm = float(request.args.get('SepalWidthCm'))
        PetalLengthCm = float(request.args.get('PetalLengthCm'))
        PetalWidthCm = float(request.args.get('PetalWidthCm'))
        data_baru2 = [[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]]
        #program input dari frontend
        hasil = loaded_model.predict(data_baru2)[0]
        return str(hasil)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='5678')