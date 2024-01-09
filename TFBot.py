import telebot
import traceback
from PIL import Image, ImageOps
import tensorflow as tf
import config

bot = telebot.TeleBot(config.TOKEN)
classes = ['американский бульдог', 'американский питбуль терьер', 'бассет-хаунд', 'бигль', 'боксер', 'чихуахуа',
           'английский кокер-спаниель', 'английский сеттер', 'немецкий курцхаар', 'великий пиреней', 'хаванез',
           'японский хин', 'кизхонд', 'леонбергер', 'миниатюрный пинчер', 'ньюфаундленд', 'померанский', 'мопс',
           'сенбернар', 'самоед', 'шотландский терьер', 'шиба ину', 'стаффордширский бультерьер', 'уитен терьер',
           'йоркширский терьер']

model = tf.keras.models.load_model('dogs_v3.h5')


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Привет! Пришли фото сюда, в ответ получи породу собаки 🐕🐕🐕')


@bot.message_handler(content_types=['photo'])
def repeat_all_messages(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        image_path = "image.jpg"

        with open(image_path, 'wb') as new_file:
            new_file.write(downloaded_file)

        image = Image.open(image_path).convert("RGB")
        size = (332, 332)
        image = ImageOps.fit(image, size, Image.LANCZOS)
        img_array = tf.keras.preprocessing.image.img_to_array(image)[tf.newaxis, ...]

        predictions = model.predict(img_array)
        predicted_class_index = tf.argmax(predictions, axis=1)[0]
        predicted_class = classes[predicted_class_index]

        bot.send_message(message.chat.id, f'На этом фото порода собаки под названием "{predicted_class}" 🐕🐕🐕')

    except Exception as e:
        traceback.print_exc()
        bot.send_message(message.chat.id, 'Упс, что-то пошло не так :(')


bot.polling(none_stop=True)
