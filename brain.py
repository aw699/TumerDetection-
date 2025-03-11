import os
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image  # type: ignore

# إنشاء تطبيق Flask
app = Flask(__name__)

# تحميل نموذج Keras
MODEL_PATH = "D:/Brain_mobilenetv2_94.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# الفئات المتوقعة حسب الموديل
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# الصفحة الرئيسية لرفع الصور
@app.route('/')
def home():
    return render_template('page.html')

# تحميل الصورة ومعالجتها
def preprocess_image(img_path, target_size=(224, 224)):  # استخدم الحجم المناسب لموديلك
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype(np.float32) / 255.0  # تأكد أن البيانات من النوع float32
    return img_array

# API لاستقبال الصورة والتنبؤ بالفئة
# نصائح طبية لكل نوع ورم
advice = {
    "glioma": "استشر طبيب أورام مخ وأعصاب. العلاج يشمل الجراحة والعلاج الإشعاعي وربما الكيميائي.",
    "meningioma": "عادةً يكون ورم حميد، لكن يفضل مراجعة الطبيب لمتابعة النمو. يمكن علاجه بالجراحة أو الإشعاع.",
    "notumor": "لا يوجد ورم، لكن إذا كنت تشعر بأعراض غير طبيعية، استشر الطبيب للاطمئنان.",
    "pituitary": "ورم الغدة النخامية قد يؤثر على الهرمونات. يفضل استشارة طبيب غدد صماء وأعصاب."
}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = "uploaded_image.jpg"
    file.save(file_path)  # حفظ الصورة مؤقتًا

    # معالجة الصورة
    img_array = preprocess_image(file_path)

    # تشغيل التنبؤ باستخدام Keras
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # استخراج النصيحة المناسبة
    predicted_label = class_names[predicted_class]
    medical_advice = advice.get(predicted_label, "يرجى استشارة الطبيب للحصول على معلومات أكثر.")

    # حذف الصورة بعد التنبؤ لتوفير المساحة
    os.remove(file_path)

    result = {
        "prediction": predicted_label,
        "confidence": float(predictions[0][predicted_class]) * 100,
        "advice": medical_advice
    }

    return jsonify(result)

# تشغيل السيرفر على البورت 5001
if __name__ == '__main__':
    app.run(debug=True, port=5001)
