import matplotlib.pyplot as plt
import numpy as np

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

lead = lead_I[10:]
t = np.linspace(0, 351, len(lead))
plt.plot(t, lead)
plt.xlim([2, 10])
plt.ylim([-200, 600])
plt.show()