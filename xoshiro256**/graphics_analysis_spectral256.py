import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('./data/processed_spectral_analysis256.csv')

# Проверка на наличие пустых строк и их удаление
data = data.dropna(how='all')  # Удаляет строки, где все значения NaN или пустые

# Настройка графиков
fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

# Первый график: только Spectral_0
axes[0].plot(data.index, data['Spectral_0'], label='Spectral_0', color='blue', alpha=0.3)
axes[0].set_title('Спектральный анализ: Spectral_0')
axes[0].set_ylabel('Амплитуда спектра')
axes[0].legend(loc='upper right')

# Второй график: Spectral_1 ... Spectral_511
for column in data.columns[2:]:
    axes[1].plot(data.index, data[column], alpha=0.2)  # Полупрозрачные линии , marker='o', markersize=1
axes[1].set_title('Спектральный анализ: Spectral_1 ... Spectral_511')
axes[1].set_ylabel('Амплитуда спектра')
axes[1].set_xlabel('Блоки')

# Сохранение графика
#plt.savefig("./data/averaged_analysis_spectral256.png")
plt.savefig("./picture256/averaged_analysis_spectral256.png")

# Настройка внешнего вида
plt.tight_layout()
plt.show()
