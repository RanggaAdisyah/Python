import nltk
from nltk.chat.util import Chat, reflections

pola = [
    ['(hai|halo|hey)', ['Halo!', 'Hai, ada yang bisa saya bantu?']],
    ['(saya|aku) ingin (.*)', ['Kenapa Anda ingin %2?', 'Anda ingin %2 karena apa?']],
    ['(saya|aku) (.*)', ['Mengapa Anda %2?', 'Apakah ada alasan mengapa Anda %2?']],
    ['siapa kamu?', ['Saya adalah bot sederhana.', 'Anda bisa memanggil saya bot.', 'ini bapak budi', 'ini ibu budi']],
    ['(apa kabar?|)', ['Saya hanyalah sebuah program, jadi saya tidak punya perasaan.']],
    ['(.*) berapa (.*)(harga|biaya)(.*)', ['Maaf, saya tidak bisa memberikan informasi tentang harga.']],
    ['buwung apa tu man', ['buwung puyuh']],
    ['(.*)', ['Maaf, saya tidak mengerti.', 'Maaf, saya tidak bisa membantu dengan itu.']]

]

chatbot = Chat(pola, reflections)

def mulai_chat():
    print("Selamat datang! Silakan mulai berbicara dengan bot. Ketik 'keluar' untuk mengakhiri percakapan.")
    while True:
        user_input = input("Anda: ")
        if user_input == 'keluar':
            print("Percakapan telah diakhiri")
            break
        else:
            response = chatbot.respond(user_input)
            print("Bot:", response)

mulai_chat()