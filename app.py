from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('model2.pkl', 'rb'))
# Load LabelEncoders for object-type columns
with open('label_encoder_gender.pkl', 'rb') as file:
    encoder_gender = pickle.load(file)
with open('label_encoder_profession.pkl', 'rb') as file:
    encoder_profession = pickle.load(file)
with open('label_encoder_spendTimeOnProf.pkl', 'rb') as file:
    encoder_spendTimeOnProf = pickle.load(file)
with open('label_encoder_whatSNSUse.pkl', 'rb') as file:
    encoder_whatSNSUse = pickle.load(file)
with open('label_encoder_spendTimeOnSNS.pkl', 'rb') as file:
    encoder_spendTimeOnSNS = pickle.load(file)
with open('label_encoder_spendTimeWithFamily.pkl', 'rb') as file:
    encoder_spendTimeWithFamily = pickle.load(file)
with open('label_encoder_whyUseSNS.pkl', 'rb') as file:
    encoder_whyUseSNS = pickle.load(file)
with open('label_encoder_personalBenifitUseSNS.pkl', 'rb') as file:
    encoder_personalBenifitUseSNS = pickle.load(file)
with open('label_encoder_whenAccessSNS.pkl', 'rb') as file:
    encoder_whenAccessSNS = pickle.load(file)
with open('label_encoder_policyAffectSNS.pkl', 'rb') as file:
    encoder_policyAffectSNS = pickle.load(file)
with open('label_encoder_adInfluencePurch.pkl', 'rb') as file:
    encoder_adInfluencePurch = pickle.load(file)
with open('label_encoder_forgetTime.pkl', 'rb') as file:
    encoder_forgetTime = pickle.load(file)


print("Labels in encoder_gender:", encoder_forgetTime.classes_)


app = Flask(__name__)

@app.route('/')
def home():
    return "Help World"

@app.route('/predict', methods=['POST'])
def predict():
    age = request.form.get('age').lower()
    gender = request.form.get('gender').lower()
    profession = request.form.get('profession').lower()
    spendTimeOnProf = request.form.get('spendTimeOnProf').lower()
    whatSNSUse = request.form.get('whatSNSUse').lower()
    spendTimeOnSNS = request.form.get('spendTimeOnSNS').lower()
    spendTimeWithFamily = request.form.get('spendTimeWithFamily').lower()
    whyUseSNS = request.form.get('whyUseSNS').lower()
    personalBenifitUseSNS = request.form.get('personalBenifitUseSNS').lower()
    whenAccessSNS = request.form.get('whenAccessSNS').lower()
    policyAffectSNS = request.form.get('policyAffectSNS').lower()
    adInfluencePurch = request.form.get('adInfluencePurch').lower()
    forgetTime = request.form.get('forgetTime').lower()

    encoded_gender = encoder_gender.transform([gender])[0]
    encoded_profession = encoder_profession.transform([profession])[0]
    encoded_spendTimeOnProf = encoder_spendTimeOnProf.transform([spendTimeOnProf])[0]
    encoded_whatSNSUse = encoder_whatSNSUse.transform([whatSNSUse])[0]
    encoded_spendTimeOnSNS = encoder_spendTimeOnSNS.transform([spendTimeOnSNS])[0]
    encoded_spendTimeWithFamily = encoder_spendTimeWithFamily.transform([spendTimeWithFamily])[0]
    encoded_whyUseSNS = encoder_whyUseSNS.transform([whyUseSNS])[0]
    encoded_personalBenifitUseSNS = encoder_personalBenifitUseSNS.transform([personalBenifitUseSNS])[0]
    encoded_whenAccessSNS = encoder_whenAccessSNS.transform([whenAccessSNS])[0]
    encoded_policyAffectSNS = encoder_policyAffectSNS.transform([policyAffectSNS])[0]
    encoded_adInfluencePurch = encoder_adInfluencePurch.transform([adInfluencePurch])[0]
    encoded_forgetTime = encoder_forgetTime.transform([forgetTime])[0]

    input_data = [age, encoded_gender, encoded_profession, encoded_spendTimeOnProf, encoded_whatSNSUse,
                  encoded_spendTimeOnSNS, encoded_spendTimeWithFamily, encoded_whyUseSNS,
                  encoded_personalBenifitUseSNS, encoded_whenAccessSNS, encoded_policyAffectSNS,
                  encoded_adInfluencePurch, encoded_forgetTime]

    result = model.predict([input_data])
    prediction = int(result[0])  # Assuming a binary outcome (0 or 1)
    return jsonify({'addicted': prediction})

  #  result = model.predict(input_data)
  #  return jsonify({'adicted': result})
'''
    input_query = np.array([[age, encoded_gender, encoded_profession, encoded_spendTimeOnProf, encoded_whatSNSUse,
                  encoded_spendTimeOnSNS, encoded_spendTimeWithFamily, encoded_whyUseSNS,
                  encoded_personalBenifitUseSNS, encoded_whenAccessSNS, encoded_policyAffectSNS,
                  encoded_adInfluencePurch, encoded_forgetTime]])

#    input_query = np.array([[age, gender, profession, spendTimeOnProf, whatSNSUse, spendTimeOnSNS, spendTimeWithFamily, whyUseSNS, personalBenifitUseSNS, whenAccessSNS, policyAffectSNS, adInfluencePurch, forgetTime]])

  #  result = {'age': age, 'gender': gender, 'profession': profession}
  #  return jsonify(result)

    result = model.predict(input_query)[0]
    return jsonify({'adicted' :result})
'''


if __name__ == '__main__':
    app.run(debug=True)
