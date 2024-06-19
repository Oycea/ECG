import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


def draw_pqrs():
    t = np.arange(len(ecg_signal)) / sample_rate  # Создаём временной массив
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
    plt.ylim([-1000, 1000])
    plt.show()


def draw_rr():
    rr_intervals, _, _, _, _, _ = get_analysis()
    plt.plot(rr_intervals, marker='o', linestyle='-', color='black')
    plt.title('RR Intervals')
    plt.xlabel('Index')
    plt.ylabel('RR Interval (s)')
    plt.grid(True)
    plt.show()


def get_analysis():
    rr_intervals = np.diff(r_peaks) / sample_rate
    nn_diff = np.diff(rr_intervals)

    sdnn = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(nn_diff ** 2))
    heart_rate = 60 / np.mean(rr_intervals)
    t_flag = False
    a_flag = False

    # Оцениваем возможное наличие тахикардии
    if heart_rate > 100:
        t_flag = True

    # Оценка наличия аритмий
    if sdnn < 0.05 or rmssd < 0.05:
        a_flag = True

    return rr_intervals, sdnn, rmssd, heart_rate, t_flag, a_flag


def save_info():
    _, sdnn, rmssd, heart_rate, t_flag, a_flag = get_analysis()

    peaks_file = 'ecg_peaks.txt'
    t = np.arange(len(ecg_signal)) / sample_rate
    with open(peaks_file, 'w', encoding='utf-8') as w_file:
        w_file.write("Peak Type\tTime (s)\tAmplitude (µV)\n")
        for r_peak in r_peaks:
            w_file.write(f"R\t{t[r_peak]}\t{ecg_signal[r_peak]}\n")
        for q_peak in q_peaks:
            w_file.write(f"Q\t{t[q_peak]}\t{ecg_signal[q_peak]}\n")
        for s_peak in s_peaks:
            w_file.write(f"S\t{t[s_peak]}\t{ecg_signal[s_peak]}\n")
        for p_peak in p_peaks:
            w_file.write(f"P\t{t[p_peak]}\t{ecg_signal[p_peak]}\n")
    print(f"Найденные пики сохранены в файле: {peaks_file}")

    print("Введите название файла для сохранения результатов:")
    res_file = input()
    with open(res_file, 'w', encoding='utf-8') as w_file:
        w_file.write(f"SDNN (стандартное отклонение интервалов RR): {sdnn:.2f} сек\n")
        w_file.write(f"RMSSD (стандартная статистическая мера ВСР): {rmssd:.2f} сек\n")
        w_file.write(f"Частота сердечных сокращений: {heart_rate:.2f} уд/мин\n")
        if t_flag:
            w_file.write(f"Обнаружена тахикардия.\n")
        if a_flag:
            w_file.write(f"Обнаружены аритмии\n")
    print(f"Результаты ЭКГ сохранены в файле: {res_file}")


print("Введите название файла с ЭКГ:")
filename = input()

sample_rate = None
fragment_duration = None
num_samples = None

lead_I = []
lead_II = []
lead_III = []
avR = []
avL = []
avF = []

current_section = None
separator = ';'

# ЧТЕНИЕ КАРДИОГРАММЫ

with open(filename, 'r', encoding='utf-8') as file:
    lines = file.readlines()

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
            lead_I.extend(map(int, line.split(separator)[:-1]))
        elif current_section == "II":
            lead_II.extend(map(int, line.split(separator)[:-1]))
        elif current_section == "III":
            lead_III.extend(map(int, line.split(separator)[:-1]))
        elif current_section == "avR":
            avR.extend(map(int, line.split(separator)[:-1]))
        elif current_section == "avL":
            avL.extend(map(int, line.split(separator)[:-1]))
        elif current_section == "avF":
            avF.extend(map(int, line.split(separator)[:-1]))

ecg_signal = np.array(lead_III)

# ЗУБЦЫ R

# Минимальное расстояние между пиками (примерно 60% от 1 секунды, чтобы избежать ложных срабатываний)
distance = int(sample_rate * 0.6)
r_peaks, _ = find_peaks(ecg_signal, distance=distance)

# ЗУБЦЫ Q и S

# Зубцы Q находятся перед зубцами R, ищем локальные минимумы
q_peaks = []
# Зубцы S находятся после зубцов R, ищем локальные минимумы
s_peaks = []

start_area_q = int(0.1 * sample_rate)  # 100 мс до R-зубца
start_area_s = int(0.1 * sample_rate)  # 100 мс после R-зубца

for r in r_peaks:
    if r > start_area_q:
        area_q = ecg_signal[r - start_area_q:r]
        q_peak = np.argmin(area_q) + (r - start_area_q)
        q_peaks.append(q_peak)

    if r + start_area_s < len(ecg_signal):
        area_s = ecg_signal[r:r + start_area_s]
        s_peak = np.argmin(area_s) + r
        s_peaks.append(s_peak)

q_peaks = np.array(q_peaks)
s_peaks = np.array(s_peaks)

# ЗУБЦЫ P

# Зубцы P находятся перед зубцами Q, ищем локальные максимумы
p_peaks = []

start_area_p = int(0.2 * sample_rate)  # 200 мс до Q-зубца

for q in q_peaks:
    if q > start_area_p:
        area_p = ecg_signal[q - start_area_p:q]
        p_peak = np.argmax(area_p) + (q - start_area_p)
        p_peaks.append(p_peak)

p_peaks = np.array(p_peaks)

# ВИЗУАЛИЗАЦИЯ

draw_pqrs()
draw_rr()

# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ

save_info()
