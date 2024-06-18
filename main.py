import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

filename = 'input.txt'

sample_rate = None
fragment_duration = None
num_samples = None

lead_I = []
lead_II = []
lead_III = []
avR = []
avL = []
avF = []

with open(filename, 'r', encoding='utf-8') as file:
    lines = file.readlines()

current_section = None

for i, line in enumerate(lines):
    line = line.strip()
    if "Частота дискретизации (Гц):" in line:
        sample_rate = int(lines[i + 1].strip())
    elif "Длительность фрагмента (сек):" in line:
        fragment_duration = int(lines[i + 1].strip())
    elif "Количество экспортированных сэмплов по каждому отведению:" in line:
        num_samples = int(lines[i + 1].strip())
    elif line.startswith("#"):
        current_section = line[1:]
    elif current_section:
        if current_section == "I":
            lead_I.extend(map(int, line.split(';')[:-1]))
        elif current_section == "II":
            lead_II.extend(map(int, line.split(';')[:-1]))
        elif current_section == "III":
            lead_III.extend(map(int, line.split(';')[:-1]))
        elif current_section == "avR":
            avR.extend(map(int, line.split(';')[:-1]))
        elif current_section == "avL":
            avL.extend(map(int, line.split(';')[:-1]))
        elif current_section == "avF":
            avF.extend(map(int, line.split(';')[:-1]))

# t = np.linspace(0, 851, len(lead))
# plt.plot(t, lead)
# plt.xlim([2, 10])
# plt.ylim([-200, 600])
# plt.show()

ecg_signal = np.array(lead_III)

fs = 100  # Частота дискретизации в Гц
duration = 351  # Длительность фрагмента в секундах
num_samples = 35096  # Количество сэмплов

assert len(
    ecg_signal) == num_samples, "Количество сэмплов не соответствует ожиданиям"

# Найдём пики (зубцы R)
# Минимальное расстояние между пиками (примерно 60% от 1 секунды, чтобы избежать ложных срабатываний)
distance = int(fs * 0.6)
r_peaks, _ = find_peaks(ecg_signal, distance=distance)

# Поиск зубцов Q и S
# Зубцы Q находятся перед зубцами R, ищем локальные минимумы
# Зубцы S находятся после зубцов R, ищем локальные минимумы
q_peaks = []
s_peaks = []

search_window_q = int(0.1 * fs)  # 100 мс до R-зубца
search_window_s = int(0.1 * fs)  # 100 мс после R-зубца

for r in r_peaks:
    if r > search_window_q:
        q_window = ecg_signal[r - search_window_q:r]
        q_peak = np.argmin(q_window) + (r - search_window_q)
        q_peaks.append(q_peak)

    if r + search_window_s < len(ecg_signal):
        s_window = ecg_signal[r:r + search_window_s]
        s_peak = np.argmin(s_window) + r
        s_peaks.append(s_peak)

q_peaks = np.array(q_peaks)
s_peaks = np.array(s_peaks)

# Поиск зубцов P
# Зубцы P находятся перед зубцами Q, ищем локальные максимумы
p_peaks = []

search_window_p = int(0.2 * fs)  # 200 мс до Q-зубца

for q in q_peaks:
    if q > search_window_p:
        p_window = ecg_signal[q - search_window_p:q]
        p_peak = np.argmax(p_window) + (q - search_window_p)
        p_peaks.append(p_peak)

p_peaks = np.array(p_peaks)

# Визуализация сигналов и найденных пиков
t = np.arange(len(ecg_signal)) / fs  # Создаём временной массив

plt.figure(figsize=(15, 6))
plt.plot(t, ecg_signal, label='ECG Signal')
plt.plot(t[r_peaks], ecg_signal[r_peaks], 'ro', label='R Peaks')
plt.plot(t[q_peaks], ecg_signal[q_peaks], 'go', label='Q Peaks')
plt.plot(t[s_peaks], ecg_signal[s_peaks], 'bo', label='S Peaks')
plt.plot(t[p_peaks], ecg_signal[p_peaks], 'mo', label='P Peaks')
plt.title('ECG Signal with P, Q, R, and S Peaks')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.legend()
plt.grid(True)
plt.xlim([2, 10])
plt.ylim([-600, 600])
plt.show()

# Сохранение найденных пиков в файл txt
output_file = 'ecg_peaks.txt'

with open(output_file, 'w') as file:
    file.write("Peak Type\tTime (s)\tAmplitude (µV)\n")

    for r_peak in r_peaks:
        file.write(f"R\t{t[r_peak]}\t{ecg_signal[r_peak]}\n")

    for q_peak in q_peaks:
        file.write(f"Q\t{t[q_peak]}\t{ecg_signal[q_peak]}\n")

    for s_peak in s_peaks:
        file.write(f"S\t{t[s_peak]}\t{ecg_signal[s_peak]}\n")

    for p_peak in p_peaks:
        file.write(f"P\t{t[p_peak]}\t{ecg_signal[p_peak]}\n")

print(f"Найденные пики сохранены в файле: {output_file}")

rr_intervals = np.diff(r_peaks) / fs  # Вычисляем интервалы RR в секундах
rr_std = np.std(rr_intervals)  # Стандартное отклонение интервалов RR
print(f"Стандартное отклонение интервалов RR: {rr_std:.2f} сек")

# Визуализация интервалов RR
plt.figure(figsize=(10, 4))
plt.plot(rr_intervals, marker='o', linestyle='-', color='b')
plt.title('RR Intervals')
plt.xlabel('Index')
plt.ylabel('RR Interval (s)')
plt.grid(True)
plt.show()

# Предполагая, что у нас есть массив с интервалами RR rr_intervals

# SDNN (стандартное отклонение всех NN интервалов)
sdnn = np.std(rr_intervals)
print(f"SDNN: {sdnn:.2f} сек")

# RMSSD (квадратный корень из среднеквадратической разности NN интервалов)
nn_diff = np.diff(rr_intervals)
rmssd = np.sqrt(np.mean(nn_diff ** 2))
print(f"RMSSD: {rmssd:.2f} сек")

# Оценка регулярности сердечного ритма и других особенностей
with open('ecg_analysis_results.txt', 'w') as file:
    file.write(f"Стандартное отклонение интервалов RR: {rr_std:.2f} сек\n")
    file.write(f"SDNN: {sdnn:.2f} сек\n")
    file.write(f"RMSSD: {rmssd:.2f} сек\n")
    file.write("\nДополнительные особенности анализа:\n")
    file.write("- Визуальный анализ: оценка формы QRS комплексов, наличие аритмий.\n")
    file.write("- Частотные характеристики: оценка частоты сердечных сокращений.\n")

print("Результаты анализа сохранены в файле 'ecg_analysis_results.txt'")

# Оцениваем среднюю ЧСС
heart_rate = 60 / np.mean(rr_intervals)  # Средний интервал RR -> ЧСС в ударах в минуту

# Оцениваем возможное наличие тахикардии
if heart_rate > 100:
    print(f"Обнаружена тахикардия. Частота сердечных сокращений: {heart_rate:.2f} уд/мин")
else:
    print(f"Частота сердечных сокращений: {heart_rate:.2f} уд/мин")

# Оценка наличия аритмий
if sdnn < 0.05 or rmssd < 0.05:
    print("Обнаружены аритмии")
else:
    print("Аритмии не обнаружены")
