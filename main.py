from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle
import ast

#Load Datasets
symtoms= pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df _Hindi.csv")
workout = pd.read_csv("datasets/workout_Hindi.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv("datasets/medications.csv")
diets = pd.read_csv("datasets/diets_Hindi.csv")
#load models
load_svc = pickle.load(open("models/svc.pkl",'rb'))

app=Flask(__name__)

# helper function
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    pre_h = precautions[precautions['Disease'] == dis][['Precaution1_Hindi', 'Precaution2_Hindi', 'Precaution3_Hindi', 'Precaution4_Hindi']]
    pre_h = [col for col in pre_h.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med]

    die = diets[diets['Disease'] == dis]['Diet']
    die_h = diets[diets['Disease'] == dis]['Diet_Hindi']

    die = [die for die in die.values]
    die_h = [die_h for die_h in die_h.values]


    wrkout = workout[workout['disease'] == dis]['workout']
    wrkout_h = workout[workout['disease'] == dis]['workout_Hindi']

    return desc, pre, med, die, wrkout, die_h,wrkout_h,pre_h


# model prediction


symptoms_dict_hi_en= {
    'itching': 'खुजली',
    'skin_rash': 'त्वचा पर चकत्ते',
    'nodal_skin_eruptions': 'त्वचा पर गांठदार फोड़े',
    'continuous_sneezing': 'लगातार छींक आना',
    'shivering': 'काँपना',
    'chills': 'सर्दी लगना',
    'joint_pain': 'जोड़ों का दर्द',
    'stomach_pain': 'पेट दर्द',
    'acidity': 'अम्लता/एसिडिटी',
    'ulcers_on_tongue': 'जीभ पर छाले',
    'muscle_wasting': 'मांसपेशियों का क्षय',
    'vomiting': 'उल्टी',
    'burning_micturition': 'पेशाब में जलन',
    'spotting_ urination': 'पेशाब में धब्बे / रक्त',
    'fatigue': 'थकान',
    'weight_gain': 'वजन बढ़ना',
    'anxiety': 'चिंता',
    'cold_hands_and_feets': 'हाथ-पैर ठंडे होना',
    'mood_swings': 'मिज़ाज में उतार-चढ़ाव',
    'weight_loss': 'वजन कम होना',
    'restlessness': 'बेचैनी',
    'lethargy': 'सुस्ती',
    'patches_in_throat': 'गले में चकत्ते',
    'irregular_sugar_level': 'अनियमित शुगर स्तर',
    'cough': 'खांसी',
    'high_fever': 'तेज़ बुखार',
    'sunken_eyes': 'धंसी हुई आँखें',
    'breathlessness': 'सांस फूलना',
    'sweating': 'पसीना आना',
    'dehydration': 'निर्जलीकरण',
    'indigestion': 'अपच',
    'headache': 'सिर दर्द',
    'yellowish_skin': 'पीली त्वचा',
    'dark_urine': 'गहरा रंग का पेशाब',
    'nausea': 'मतली',
    'loss_of_appetite': 'भूख न लगना',
    'pain_behind_the_eyes': 'आँखों के पीछे दर्द',
    'back_pain': 'कमर दर्द',
    'constipation': 'कब्ज',
    'abdominal_pain': 'पेट में दर्द',
    'diarrhoea': 'दस्त',
    'mild_fever': 'हल्का बुखार',
    'yellow_urine': 'पीला पेशाब',
    'yellowing_of_eyes': 'आँखें पीली होना',
    'acute_liver_failure': 'गंभीर लिवर फेल होना',
    'fluid_overload': 'शरीर में तरल का जमा होना',
    'swelling_of_stomach': 'पेट में सूजन',
    'swelled_lymph_nodes': 'लसीका ग्रंथियों की सूजन',
    'malaise': 'अस्वस्थता',
    'blurred_and_distorted_vision': 'धुंधली और विकृत दृष्टि',
    'phlegm': 'बलगम',
    'throat_irritation': 'गले में जलन',
    'redness_of_eyes': 'आँखों की लाली',
    'sinus_pressure': 'साइनस दबाव',
    'runny_nose': 'बहती नाक',
    'congestion': 'जकड़न',
    'chest_pain': 'सीने में दर्द',
    'weakness_in_limbs': 'हाथ-पैर में कमजोरी',
    'fast_heart_rate': 'तेज़ धड़कन',
    'pain_during_bowel_movements': 'शौच के दौरान दर्द',
    'pain_in_anal_region': 'मलाशय क्षेत्र में दर्द',
    'bloody_stool': 'खून वाली मल',
    'irritation_in_anus': 'गुदा में जलन',
    'neck_pain': 'गर्दन दर्द',
    'dizziness': 'चक्कर आना',
    'cramps': 'ऐंठन',
    'bruising': 'नील पड़ना',
    'obesity': 'मोटापा',
    'swollen_legs': 'सूजे हुए पैर',
    'swollen_blood_vessels': 'सूजी हुई रक्त वाहिकाएं',
    'puffy_face_and_eyes': 'चेहरे और आंखों में सूजन',
    'enlarged_thyroid': 'बड़ा थायरॉयड',
    'brittle_nails': 'कमजोर नाखून',
    'swollen_extremeties': 'हाथ-पैरों में सूजन',
    'excessive_hunger': 'अत्यधिक भूख',
    'extra_marital_contacts': 'अवैध यौन संपर्क',
    'drying_and_tingling_lips': 'होंठों का सूखना और झुनझुनी',
    'slurred_speech': 'तोतली वाणी',
    'knee_pain': 'घुटने का दर्द',
    'hip_joint_pain': 'कूल्हे के जोड़ में दर्द',
    'muscle_weakness': 'मांसपेशियों की कमजोरी',
    'stiff_neck': 'गर्दन अकड़ना',
    'swelling_joints': 'जोड़ों में सूजन',
    'movement_stiffness': 'हिलने-डुलने में अकड़न',
    'spinning_movements': 'घूमने जैसा लगना',
    'loss_of_balance': 'संतुलन खोना',
    'unsteadiness': 'अस्थिरता',
    'weakness_of_one_body_side': 'शरीर के एक तरफ कमजोरी',
    'loss_of_smell': 'गंध की क्षमता खोना',
    'bladder_discomfort': 'मूत्राशय में तकलीफ',
    'foul_smell_of urine': 'पेशाब से बदबू आना',
    'continuous_feel_of_urine': 'बार-बार पेशाब का एहसास',
    'passage_of_gases': 'गैस पास होना',
    'internal_itching': 'आंतरिक खुजली',
    'toxic_look_(typhos)': 'ज़हरीला चेहरा (टाइफस)',
    'depression': 'अवसाद',
    'irritability': 'चिड़चिड़ापन',
    'muscle_pain': 'मांसपेशियों में दर्द',
    'altered_sensorium': 'होश की गड़बड़ी',
    'red_spots_over_body': 'शरीर पर लाल धब्बे',
    'belly_pain': 'पेट दर्द',
    'abnormal_menstruation': 'असामान्य माहवारी',
    'dischromic _patches': 'त्वचा पर रंग बदलने वाले धब्बे',
    'watering_from_eyes': 'आँखों से पानी आना',
    'increased_appetite': 'भूख बढ़ना',
    'polyuria': 'बार-बार पेशाब आना',
    'family_history': 'पारिवारिक इतिहास',
    'mucoid_sputum': 'बलगमी थूक',
    'rusty_sputum': 'जंग जैसा थूक',
    'lack_of_concentration': 'एकाग्रता की कमी',
    'visual_disturbances': 'दृष्टि संबंधी समस्या',
    'receiving_blood_transfusion': 'रक्त चढ़ाना',
    'receiving_unsterile_injections': 'गंदे इंजेक्शन लगना',
    'coma': 'कोमा',
    'stomach_bleeding': 'पेट से खून आना',
    'distention_of_abdomen': 'पेट फूलना',
    'history_of_alcohol_consumption': 'शराब पीने का इतिहास',
    'fluid_overload.1': 'शरीर में अतिरिक्त तरल',
    'blood_in_sputum': 'थूक में खून आना',
    'prominent_veins_on_calf': 'पिंडलियों पर उभरी नसें',
    'palpitations': 'दिल की धड़कन बढ़ना',
    'painful_walking': 'चलते समय दर्द',
    'pus_filled_pimples': 'पीप से भरे फोड़े',
    'blackheads': 'काले दाने',
    'scurring': 'निशान/दाग',
    'skin_peeling': 'त्वचा का छिलना',
    'silver_like_dusting': 'चांदी जैसे धब्बे',
    'small_dents_in_nails': 'नाखूनों में गड्ढे',
    'inflammatory_nails': 'सूजे हुए नाखून',
    'blister': 'छाले',
    'red_sore_around_nose': 'नाक के आसपास लाल घाव',
    'yellow_crust_ooze': 'पीला पस/पपड़ी निकलना'
}

diseases_dict_en_hi = {
    'Fungal infection': 'फंगल संक्रमण',
    'Allergy': 'एलर्जी',
    'GERD': 'जीईआरडी (GERD)',
    'Chronic cholestasis': 'क्रोनिक कोलेस्टेसिस',
    'Drug Reaction': 'दवा से प्रतिक्रिया',
    'Peptic ulcer diseae': 'पेप्टिक अल्सर रोग',
    'AIDS': 'एड्स (AIDS)',
    'Diabetes': 'मधुमेह',
    'Gastroenteritis': 'आंत्रशोथ (गैस्ट्रोएंटेराइटिस)',
    'Bronchial Asthma': 'ब्रोंकियल अस्थमा',
    'Hypertension': 'उच्च रक्तचाप',
    'Migraine': 'माइग्रेन',
    'Cervical spondylosis': 'सर्वाइकल स्पॉन्डिलोसिस',
    'Paralysis (brain hemorrhage)': 'लकवा (मस्तिष्क रक्तस्राव)',
    'Jaundice': 'पीलिया',
    'Malaria': 'मलेरिया',
    'Chicken pox': 'चेचक (चिकनपॉक्स)',
    'Dengue': 'डेंगू',
    'Typhoid': 'टाइफॉइड',
    'hepatitis A': 'हेपेटाइटिस A',
    'Hepatitis B': 'हेपेटाइटिस B',
    'Hepatitis C': 'हेपेटाइटिस C',
    'Hepatitis D': 'हेपेटाइटिस D',
    'Hepatitis E': 'हेपेटाइटिस E',
    'Alcoholic hepatitis': 'अल्कोहलिक हेपेटाइटिस',
    'Tuberculosis': 'क्षय रोग (टीबी)',
    'Common Cold': 'सामान्य जुकाम',
    'Pneumonia': 'निमोनिया',
    'Dimorphic hemmorhoids(piles)': 'बवासीर (डिमॉर्फिक हेमोरॉइड्स)',
    'Heart attack': 'दिल का दौरा',
    'Varicose veins': 'वैरिकाज़ नसें',
    'Hypothyroidism': 'हाइपोथायरायडिज्म',
    'Hyperthyroidism': 'हाइपरथायरायडिज्म',
    'Hypoglycemia': 'हाइपोग्लाइसीमिया',
    'Osteoarthristis': 'ऑस्टियोआर्थराइटिस',
    'Arthritis': 'गठिया',
    '(vertigo) Paroymsal  Positional Vertigo': 'वर्टिगो (परॉक्सिस्मल पोज़िशनल वर्टिगो)',
    'Acne': 'मुंहासे (एक्ने)',
    'Urinary tract infection': 'मूत्र मार्ग संक्रमण',
    'Psoriasis': 'सोरायसिस',
    'Impetigo': 'इम्पेटिगो'
}

description_hi = {
    "Fungal infection": "फंगल संक्रमण एक सामान्य त्वचा की स्थिति है जो कवक के कारण होती है।",
    "Allergy": "एलर्जी प्रतिरक्षा प्रणाली की प्रतिक्रिया है जो किसी पर्यावरणीय पदार्थ के कारण होती है।",
    "GERD": "जीईआरडी (गैस्ट्रोएसोफेजियल रिफ्लक्स डिजीज) एक पाचन विकार है जो निचले अन्नप्रणाली स्फिंक्टर को प्रभावित करता है।",
    "Chronic cholestasis": "क्रॉनिक कोलेस्टेसिस एक स्थिति है जिसमें लंबे समय तक जिगर से पित्त का प्रवाह कम हो जाता है।",
    "Drug Reaction": "दवा प्रतिक्रिया तब होती है जब शरीर किसी दवा पर प्रतिकूल प्रतिक्रिया करता है।",
    "Peptic ulcer disease": "पेप्टिक अल्सर रोग में पेट और छोटी आंत की परत पर घाव विकसित हो जाते हैं।",
    "AIDS": "एड्स (अक्वायर्ड इम्यूनोडेफिशिएंसी सिंड्रोम) एचआईवी से होने वाला रोग है जो प्रतिरक्षा प्रणाली को कमजोर करता है।",
    "Diabetes": "मधुमेह एक पुरानी स्थिति है जो शरीर में रक्त शर्करा को संसाधित करने के तरीके को प्रभावित करती है।",
    "Gastroenteritis": "गैस्ट्रोएंटेराइटिस पेट और आंतों की सूजन है, जो आमतौर पर वायरस या बैक्टीरिया से होती है।",
    "Bronchial Asthma": "ब्रॉन्कियल अस्थमा एक श्वसन रोग है जिसमें श्वसन नलिकाओं में सूजन हो जाती है।",
    "Hypertension": "हाई ब्लड प्रेशर या उच्च रक्तचाप एक सामान्य हृदय संबंधी समस्या है।",
    "Migraine": "माइग्रेन एक प्रकार का सिरदर्द है जिसमें तीव्र दर्द और रोशनी व आवाज के प्रति संवेदनशीलता होती है।",
    "Cervical spondylosis": "सर्वाइकल स्पॉन्डिलोसिस गर्दन की रीढ़ की हड्डी का अपक्षयी रोग है।",
    "Paralysis (brain hemorrhage)": "पक्षाघात (मस्तिष्क रक्तस्राव) मस्तिष्क में रक्तस्राव के कारण मांसपेशियों की कार्यक्षमता की हानि है।",
    "Jaundice": "पीलिया त्वचा और आंखों का पीला पड़ना है, जो आमतौर पर जिगर की समस्या का संकेत देता है।",
    "Malaria": "मलेरिया एक मच्छर जनित संक्रामक रोग है।",
    "Chicken pox": "चेचक एक अत्यधिक संक्रामक वायरल संक्रमण है जिसमें खुजलीदार चकत्ते होते हैं।",
    "Dengue": "डेंगू एक मच्छर जनित वायरल संक्रमण है जिसमें फ्लू जैसे लक्षण होते हैं।",
    "Typhoid": "टाइफाइड एक जीवाणुजनित संक्रमण है जिसमें तेज बुखार और पाचन संबंधी लक्षण होते हैं।",
    "Hepatitis A": "हेपेटाइटिस ए एक वायरल यकृत रोग है।",
    "Hepatitis B": "हेपेटाइटिस बी एक वायरल संक्रमण है जो यकृत पर हमला करता है।",
    "Hepatitis C": "हेपेटाइटिस सी एक वायरल संक्रमण है जो यकृत में सूजन पैदा करता है।",
    "Hepatitis D": "हेपेटाइटिस डी हेपेटाइटिस डी वायरस के कारण होने वाला गंभीर यकृत रोग है।",
    "Hepatitis E": "हेपेटाइटिस ई एक वायरल संक्रमण है जो यकृत में सूजन पैदा करता है।",
    "Alcoholic hepatitis": "अल्कोहलिक हेपेटाइटिस शराब के सेवन से होने वाली यकृत की सूजन है।",
    "Tuberculosis": "टीबी (क्षय रोग) एक जीवाणुजनित संक्रमण है जो मुख्यतः फेफड़ों को प्रभावित करता है।",
    "Common Cold": "साधारण जुकाम ऊपरी श्वसन तंत्र का वायरल संक्रमण है।",
    "Pneumonia": "निमोनिया एक सूजन संबंधी स्थिति है जो फेफड़ों की वायु थैलियों को प्रभावित करती है।",
    "Dimorphic hemmorhoids (piles)": "बवासीर एक स्थिति है जिसमें मलाशय की रक्त वाहिकाएं सूज जाती हैं।",
    "Heart attack": "दिल का दौरा तब होता है जब हृदय की मांसपेशी तक रक्त प्रवाह अचानक और गंभीर रूप से कम हो जाता है।",
    "Varicose veins": "वैरिकोज नसें बड़ी और मुड़ी हुई नसें होती हैं, जो आमतौर पर पैरों में दिखाई देती हैं।",
    "Hypothyroidism": "हाइपोथायरायडिज्म वह स्थिति है जब थायरॉयड ग्रंथि पर्याप्त हार्मोन का उत्पादन नहीं करती।",
    "Hyperthyroidism": "हाइपरथायरायडिज्म वह स्थिति है जब थायरॉयड ग्रंथि अत्यधिक हार्मोन का उत्पादन करती है।",
    "Hypoglycemia": "हाइपोग्लाइसीमिया असामान्य रूप से कम रक्त शर्करा स्तर की स्थिति है।",
    "Osteoarthritis": "ऑस्टियोआर्थराइटिस एक अपक्षयी जोड़ रोग है जो उपास्थि को प्रभावित करता है।",
    "Arthritis": "आर्थराइटिस एक या अधिक जोड़ों में सूजन है, जिससे दर्द और अकड़न होती है।",
    "Vertigo": "वर्टिगो (पॉजिशनल पैरॉक्सिस्मल वर्टिगो) सिर की कुछ विशेष हरकतों से होने वाली चक्कर की स्थिति है।",
    "Acne": "मुँहासे एक त्वचा की स्थिति है जिसमें रोमछिद्र तेल और मृत कोशिकाओं से भर जाते हैं।",
    "Urinary tract infection": "मूत्र मार्ग संक्रमण मूत्र प्रणाली के किसी भी भाग में संक्रमण है।",
    "Psoriasis": "सोरायसिस एक पुरानी त्वचा रोग है जिसमें लाल, खुजलीदार और परतदार धब्बे हो जाते हैं।",
    "Impetigo": "इंपेटिगो एक अत्यधिक संक्रामक त्वचा संक्रमण है जिसमें लाल फोड़े होते हैं।"
}
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4,
                 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9,
                 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13,
                 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18,
                 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22,
                 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27,
                 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
                 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37,
                 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42,
                 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46,
                 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50,
                 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54,
                 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58,
                 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61,
                 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66,
                 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70,
                 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74,
                 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78,
                 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82,
                 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86,
                 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89,
                 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92,
                 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96,
                 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100,
                 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103,
                 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107,
                 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110,
                 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113,
                 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116,
                 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120,
                 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124,
                 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127,
                 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}

diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction',
                 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma',
                 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)',
                 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A',
                 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis',
                 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)',
                 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism',
                 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis',
                 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection',
                 35: 'Psoriasis', 27: 'Impetigo'}


def get_prediction_value(patient_symtomps):
    input_vector = np.zeros(len(symptoms_dict))

    for item in patient_symtomps:
        input_vector[symptoms_dict[item]] = 1

    return diseases_list[load_svc.predict([input_vector])[0]]


combine = (symtoms['Symptom_1'] + ", " + symtoms['Symptom_2'] + ", " + symtoms['Symptom_3'] + ", " + symtoms[
    'Symptom_4'])
combine = combine.dropna().unique()
#home route
@app.route('/')
def index():
    return  render_template('index.html',combine=combine,symptoms_dict=symptoms_dict.keys(),symptoms_dict_hindi=symptoms_dict_hi_en.items())

@app.route('/predict', methods=['POST', 'GET'])
def predict():
  try:
    symptom1 = request.form.get("symptom1", "")
    symptom2 = request.form.get("symptom2", "")
    symptom3 = request.form.get("symptom3", "")
    symptom4 = request.form.get("symptom4", "")
    message = 'Click Any Above Button to Show Result .'

    if request.method == 'POST':
        # Get up to 4 symptoms
        user_symptoms = []
        for i in range(1, 5):
            sym = request.form.get(f'symptom{i}')
            if sym and sym.strip():
                user_symptoms.append(sym.strip().lower().replace(" ", "_"))

        if not user_symptoms:
            message = "Please enter at least one symptom."
            return render_template('index.html', message=message, combine=combine,symptoms_dict=symptoms_dict.keys(),symptoms_dict_hindi=symptoms_dict_hi_en .items())

        # check unknown symptoms
        unknown_symptoms = [sym for sym in user_symptoms if sym not in symptoms_dict]
        if unknown_symptoms:
            message = f"Unknown or misspelled symptoms: {', '.join(unknown_symptoms)}"
            return render_template('index.html', message=message, combine=combine,symptoms_dict=symptoms_dict.keys(),symptoms_dict_hindi=symptoms_dict_hi_en .items())

        # prediction
        pred_disease = get_prediction_value(user_symptoms)

        desc, pre, med, die, wrkout ,die_h,wrkout_h,pre_h= helper(pred_disease)


        my_pre = list(pre[0])
        my_pre_h = list(pre_h[0])
        my_med = ast.literal_eval(med[0]) if med else ["No medication available"]
        my_die = ast.literal_eval(die[0]) if die else ["No diet info available"]
        my_die_h = ast.literal_eval(die_h[0]) if die_h else ["No diet info available"]

        die_combined = list(zip(my_die, my_die_h))
        workout_combined = list(zip(wrkout,wrkout_h))
        pre_combined = list(zip(my_pre,my_pre_h))
        return render_template(
            'index.html',
            message=message,
            user_symptoms=user_symptoms,
            combine=combine,
            symptoms_dict=symptoms_dict.keys(),
            pred_disease=pred_disease,
            disease_hindi =diseases_dict_en_hi[pred_disease],
            desc=desc,
            description_hi=description_hi[pred_disease],
            my_med=my_med,
            die_combined = die_combined,
            workout_combined=workout_combined,
            pre_combined=pre_combined,
            selected_symptom1=symptom1,
            selected_symptom2=symptom2,
            selected_symptom3=symptom3,
            selected_symptom4=symptom4,
            symptoms_dict_hindi=symptoms_dict_hi_en.items()
        )
  except:
      message = f"Unknown or misspelled symptoms: {', '.join(unknown_symptoms)}"
      return render_template('index.html', message=message, combine=combine, symptoms_dict=symptoms_dict.keys(),
                             symptoms_dict_hindi=symptoms_dict_hi_en.items())
@app.route('/index')
def home():
    return  render_template('index.html',combine=combine,symptoms_dict=symptoms_dict.keys(),symptoms_dict_hindi=symptoms_dict_hi_en .items())
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/developer')
def developer():
    return render_template('developer.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')
#python main
if __name__=="__main__":
    app.run(debug=True)