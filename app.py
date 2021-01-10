from flask import Flask, render_template, request, flash
from keras.models import model_from_json
from keras.optimizers import Adam
from wtforms import Form, SelectField, IntegerField, validators, DecimalField, SubmitField
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import tensorflow as tf


app = Flask(__name__, template_folder='templates')
app.secret_key = 'a secret key'
# model = load_model('model.h5')
# graph = tf.get_default_graph()

data = pd.read_csv("Global_Wine_Points_Formatted.csv", engine='python')
data.dropna()
province = data['Province']
vintage_pre = np.array(data['IntVintage'])
price_pre = np.array(data['IntPrice'])

vintage_pre = vintage_pre.astype(np.float)
min_vintage = np.min(vintage_pre)
max_vintage = np.max(vintage_pre)

price_pre = price_pre.astype(np.float)
min_price = np.min(price_pre)
max_price = np.max(price_pre)

province = np.unique(province)
indices = pd.DataFrame(province)
one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_encoder.fit(indices)
provinces_encoded = one_hot_encoder.transform(indices)
a_list = list(range(len(province)))
tuple_list = list(zip(a_list,province))


json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()

# use Keras model_from_json to make a loaded model

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("model.h5")
print("Loaded Model from disk")

# compile and evaluate loaded model
opt = Adam(learning_rate=0.00001)
loaded_model.compile(optimizer=opt, loss="mean_squared_error",
                   metrics=['mean_absolute_percentage_error'])

class ReusableForm(Form):
    vintage = IntegerField('Vintage:', validators=[validators.required()])
    province = SelectField('Province:',choices = tuple_list, coerce = int,validate_choice=False)
    price = DecimalField('Price:', validators=[validators.required()])

    @app.route("/", methods=['GET', 'POST'])
    def predict():
        prediction = ''
        form = ReusableForm(request.form)

        print(form.errors)
        if request.method == 'POST':
            vintage=request.form['vintage']
            province=request.form['province']
            price=request.form['price']


        if form.validate():
            # Save the comment here.
            #process data
            vintage = int(vintage)
            price = float(price)
            province_enc = provinces_encoded[int(province)]
            vintage_norm = (vintage - min_vintage)/(max_vintage-min_vintage)
            prices_norm = (price-min_price)/(max_price-min_price)
            inputs = np.zeros(2+len(province_enc))
            inputs[0] = prices_norm
            inputs[1] = vintage_norm
            for i in range(len(province_enc)):
                inputs[i+2]=province_enc[i]
            inputs = inputs.reshape(1,-1)
            raw_prediction = loaded_model.predict(inputs)
            raw_prediction = float(raw_prediction)*100
            if raw_prediction < 70:
                prediction = "D+"
            if raw_prediction < 75 and raw_prediction >= 70:
                prediction = "C"
            if raw_prediction < 80 and raw_prediction >= 75:
                prediction = "C+"
            if raw_prediction < 85 and raw_prediction >= 80:
                prediction = "B"
            if raw_prediction < 90 and raw_prediction >= 85:
                prediction = "B+"
            if raw_prediction >= 90:
                prediction = "A"
            flash("This wine recieves an "+prediction)

        else:
            print('All the fields are required. ')

        return render_template('predict.html', form=form, prediction=prediction)

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
"please wait until server has fully started"))
    #init()
    app.run(debug=True)
