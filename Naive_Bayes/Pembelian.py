import math

# Data buatan
data = [
    {'hari': 'senin', 'promosi': 'ya', 'cuaca': 'cerah', 'pembelian': 'banyak'},
    {'hari': 'senin', 'promosi': 'tidak', 'cuaca': 'mendung', 'pembelian': 'sedikit'},
    {'hari': 'selasa', 'promosi': 'ya', 'cuaca': 'cerah', 'pembelian': 'banyak'},
    {'hari': 'selasa', 'promosi': 'tidak', 'cuaca': 'hujan', 'pembelian': 'sedikit'},
    {'hari': 'rabu', 'promosi': 'ya', 'cuaca': 'mendung', 'pembelian': 'banyak'},
    {'hari': 'rabu', 'promosi': 'tidak', 'cuaca': 'cerah', 'pembelian': 'sedikit'},
    {'hari': 'kamis', 'promosi': 'ya', 'cuaca': 'hujan', 'pembelian': 'banyak'},
    {'hari': 'kamis', 'promosi': 'tidak', 'cuaca': 'cerah', 'pembelian': 'sedikit'},
    {'hari': 'jumat', 'promosi': 'ya', 'cuaca': 'cerah', 'pembelian': 'banyak'},
    {'hari': 'jumat', 'promosi': 'tidak', 'cuaca': 'mendung', 'pembelian': 'sedikit'},
]

# Fungsi untuk menghitung probabilitas
def hitung_probabilitas(data, fitur, nilai_fitur, target, nilai_target):
    count_target = 0
    count_fitur_target = 0
    for d in data:
        if d[target] == nilai_target:
            count_target += 1
            if d[fitur] == nilai_fitur:
                count_fitur_target += 1
    return count_fitur_target / count_target if count_target != 0 else 0

# Fungsi untuk menghitung probabilitas target
def hitung_probabilitas_target(data, target, nilai_target):
    count_target = sum(1 for d in data if d[target] == nilai_target)
    return count_target / len(data)

# Fungsi untuk prediksi menggunakan Naive Bayes
def prediksi(data, input_data):
    targets = list(set(d['pembelian'] for d in data))
    probabilitas_target = {target: hitung_probabilitas_target(data, 'pembelian', target) for target in targets}
    probabilitas_input_target = {target: 1 for target in targets}
    
    for target in targets:
        for fitur, nilai in input_data.items():
            probabilitas_fitur_target = hitung_probabilitas(data, fitur, nilai, 'pembelian', target)
            probabilitas_input_target[target] *= probabilitas_fitur_target
    
    for target in targets:
        probabilitas_input_target[target] *= probabilitas_target[target]
    
    return max(probabilitas_input_target, key=probabilitas_input_target.get)

# Mendapatkan input dari pengguna
input_data = {}
fitur_yang_digunakan = ['hari', 'promosi', 'cuaca']

for fitur in fitur_yang_digunakan:
    nilai = input(f"Masukkan {fitur}: ")
    input_data[fitur] = nilai

# Lakukan prediksi
hasil_prediksi = prediksi(data, input_data)
print(f"Hasil prediksi untuk input {input_data}: {hasil_prediksi}")
