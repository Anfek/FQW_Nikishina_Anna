import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

# Загрузка данных
file_path = './data/processed_data128.csv'
data = pd.read_csv(file_path)

# Проверка на наличие пустых строк и их удаление
data = data.dropna(how='all')  # Удаляет строки, где все значения NaN или пустые

# Универсальная функция для построения графиков
def plot_group(
    data,
    columns,
    title,
    xlabel,
    ylabel,
    window_title,
    save_as,
    legend_labels=None,
    show_legend=True,
    plot_type="line",
    alpha_num=0.5,
    bins=None,
):
    """
    Универсальная функция для построения графиков.

    :param data: DataFrame с данными
    :param columnslegend_labels: Список колонок для построения
    :param title: Заголовок графика
    :param xlabel: Название оси X
    :param ylabel: Название оси Y
    :param window_title: Заголовок окна
    :param save_as: Название файла для сохранения
    :param legend_labels: Пользовательские названия для легенды
    :param plot_type: Тип графика ("line", "bar", "hist")
    :param bins: Количество интервалов (для гистограммы)
    """
    plt.figure(figsize=(12, 6))

    for i, col in enumerate(columns):
        label = legend_labels[i] if legend_labels else col
        if plot_type == "line":
            plt.plot(data.index, data[col], alpha=alpha_num, label=label)
        elif plot_type == "bar":
            plt.bar(data.index, data[col], alpha=alpha_num, label=label)
        elif plot_type == "hist" and bins:
            plt.hist(data[col], bins=bins, alpha=alpha_num, label=label, edgecolor='black')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend_labels and show_legend:
        plt.legend(loc='upper right', ncol=2)
    plt.grid(True)
    plt.gcf().canvas.manager.set_window_title(window_title)
    os.makedirs('./picture128', exist_ok=True)
    plt.savefig(f'./picture128/{save_as}')
    plt.show()

# Функция для построения парных графиков
def plot_group_pair(
    data,
    columns_top,
    columns_bottom,
    title_top,
    title_bottom,
    xlabel,
    ylabel_top,
    ylabel_bottom,
    window_title,
    save_as,
    legend_labels_top=None,
    legend_labels_bottom=None,
    show_legend_top=True,
    show_legend_bottom=True,
    alpha_num=0.5,
):
    """
    Построение двух графиков в одном окне (вертикально), с возможностью задавать названия линий и отключать легенды.
    """
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Верхний график
    for i, col in enumerate(columns_top):
        label = legend_labels_top[i] if legend_labels_top else col
        axs[0].plot(data.index, data[col], alpha=alpha_num, label=label if show_legend_top else None)
    axs[0].set_title(title_top)
    axs[0].set_ylabel(ylabel_top)
    if show_legend_top and legend_labels_top:  # Легенда только если она нужна
        axs[0].legend(loc='upper right', ncol=2)
    axs[0].grid(True)

    # Нижний график
    for i, col in enumerate(columns_bottom):
        label = legend_labels_bottom[i] if legend_labels_bottom else col
        axs[1].plot(data.index, data[col], alpha=alpha_num, label=label if show_legend_bottom else None)
    axs[1].set_title(title_bottom)
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel(ylabel_bottom)
    if show_legend_bottom and legend_labels_bottom:  # Легенда только если она нужна
        axs[1].legend(loc='upper right', ncol=2)
    axs[1].grid(True)

    # Общий заголовок и сохранение графика
    fig.suptitle(window_title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs('./picture128', exist_ok=True)
    plt.savefig(f'./picture128/{save_as}')
    plt.show()

def plot_differences_with_intervals(data, columns, title, save_as):
    """
    Построение комбинированного графика:
    1. Среднее распределение с доверительными интервалами.
    2. Тепловая карта для всех строк.
    """
    # Подготовка данных
    mean_values = data[columns].mean(axis=0)  # Средние значения по столбцам
    std_values = data[columns].std(axis=0)   # Стандартное отклонение по столбцам
    data_array = data[columns].to_numpy()    # Преобразование данных в массив

    # Создание графиков
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 2]})

    # Среднее распределение с доверительными интервалами
    axs[0].plot(range(len(columns)), mean_values, color='red', label='Средние значения')
    axs[0].fill_between(
        range(len(columns)),
        mean_values - std_values,
        mean_values + std_values,
        color='red',
        alpha=0.2,
        label='Доверительный интервал (±1σ)'
    )
    axs[0].set_title('Среднее распределение с доверительными интервалами')
    axs[0].set_xlabel('Индекс столбца')
    axs[0].set_ylabel('Значение')
    axs[0].legend()
    axs[0].grid(True)

    # Тепловая карта (Heatmap)
    sns.heatmap(data_array, cmap='coolwarm', ax=axs[1], cbar_kws={'label': 'Значение'})
    axs[1].set_title('Тепловая карта')
    axs[1].set_xlabel('Индекс столбца')
    axs[1].set_ylabel('Индекс строки')

    # Общий заголовок и сохранение
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs('./picture128', exist_ok=True)
    plt.savefig(f'./picture128/{save_as}', bbox_inches='tight')
    plt.show()


def plot_clusters_global(data, mean_columns, distance_columns, count_columns, title, xlabel, ylabel_mean, ylabel_distance, ylabel_count, window_title, save_as):
    """
    Построение графика для кластеров: средние значения центроидов, расстояния между ними и количество элементов в каждом кластере.
    """
    num_clusters = len(mean_columns)
    x = range(num_clusters)  # Индексы кластеров
    x_distances = [i + 0.5 for i in range(num_clusters)]  # Сдвиг для расстояний

    means = [data[col].mean() for col in mean_columns]
    distances = [data[col].mean() for col in distance_columns]
    counts = [data[col].mean() for col in count_columns]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Столбчатый график для средних центроидов
    ax1.bar(x, means, alpha=0.7, color='blue', label='Средние значения центроидов')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel_mean, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Линия для расстояний между кластерами (сдвиг точек)
    ax2 = ax1.twinx()
    ax2.plot(x_distances, distances, marker='o', linestyle='--', color='red', label='Расстояния между кластерами')
    ax2.set_ylabel(ylabel_distance, color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Дополнительный график для количества элементов в кластерах
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Сдвиг третьей оси вправо
    ax3.bar(x, counts, alpha=0.5, color='green', label='Количество элементов в кластере', width=0.4)
    ax3.set_ylabel(ylabel_count, color='green')
    ax3.tick_params(axis='y', labelcolor='green')

    # Легенда под графиком
    fig.suptitle(title)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=3)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.gcf().canvas.manager.set_window_title(window_title)
    os.makedirs('./picture128', exist_ok=True)
    plt.savefig(f'./picture128/{save_as}', bbox_inches='tight')  # Учитываем легенду
    plt.show()

def plot_detailed_clusters(data, mean_columns, distance_columns, count_columns, title, save_as):
    """
    Построение подробных графиков для каждого из столбцов (Clusters_mean_, Clusters_distance_, ClusterCount_mean_),
    без легенд и с прозрачностью для линий.
    """
    num_clusters = len(mean_columns)
    x = range(num_clusters)  # Индексы кластеров
    x_distances = [i + 0.5 for i in range(num_clusters-1)]  # Сдвиг для расстояний

    fig, axs = plt.subplots(3, 1, figsize=(12, 18))

    # График для Clusters_mean_
    for _, row in data.iterrows():
        axs[0].plot(x, row[mean_columns], marker='o', linestyle='-', alpha=0.3)
    axs[0].set_title('Подробные данные: Средние значения центроидов (Clusters_mean_)')
    axs[0].set_xlabel('Индекс кластера')
    axs[0].set_ylabel('Значение')
    axs[0].grid(True)

    # График для Clusters_distance_
    for _, row in data.iterrows():
        axs[1].plot(x_distances, row[distance_columns], marker='o', linestyle='-', alpha=0.3)
    axs[1].set_title('Подробные данные: Расстояния между кластерами (Clusters_distance_)')
    axs[1].set_xlabel('Индекс расстояния')
    axs[1].set_ylabel('Расстояние')
    axs[1].grid(True)

    # График для ClusterCount_mean_
    for _, row in data.iterrows():
        axs[2].plot(x, row[count_columns], marker='o', linestyle='-', alpha=0.3)
    axs[2].set_title('Подробные данные: Количество элементов в кластере (ClusterCount_mean_)')
    axs[2].set_xlabel('Индекс кластера')
    axs[2].set_ylabel('Количество элементов')
    axs[2].grid(True)

    # Общий заголовок и сохранение
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs('./picture128', exist_ok=True)
    plt.savefig(f'./picture128/{save_as}', bbox_inches='tight')
    plt.show()


# Вызовы функций для визуализации

# Окно 1: Корреляции (Средние и Дисперсии)
plot_group_pair(
    data,
    columns_top=['Correlation_mean_0', 'Correlation_mean_1'],
    columns_bottom=['Correlation_dispersion_0', 'Correlation_dispersion_1'],
    title_top='Средние значения корреляций',
    title_bottom='Дисперсии корреляций',
    xlabel='Индекс',
    ylabel_top='Средние значения',
    ylabel_bottom='Дисперсии',
    window_title='Линейные коэффициенты корреляции',
    legend_labels_top=['r (s0,res)', 'r (s1,res)'],
    legend_labels_bottom=['r (s0,res)', 'r (s1,res)'],
    alpha_num=0.5,
    save_as='correlations_pair.png'
)

# Окно 2: Равномерность
plot_group(
    data,
    columns=['Uniformity_mean_0'],
    title='Усреднённая равномерность: Средние значения',
    xlabel='Индекс',
    ylabel='Значение',
    window_title='Равномерность',
    show_legend=False,
    alpha_num=0.6,
    save_as='uniformity.png'
)

# Окно 3: Частота бит
plot_group_pair(
    data,
    columns_top=[f"BitFrequency_mean_{i}" for i in range(64)],
    columns_bottom=[f"BitFrequency_entropy_{i}" for i in range(64)],
    title_top="Частота бит в result: Средние значения",
    title_bottom="Частота бит в result: Энтропии",
    xlabel="Индекс",
    ylabel_top="Средние значения",
    ylabel_bottom="Энтропии",
    window_title="Частота бит в result",
    show_legend_top=False,
    show_legend_bottom=False,
    alpha_num=0.3,
    save_as="bit_frequency_pair.png"
)

# Окно 4: Частота пар
plot_group_pair(
    data,
    columns_top=[f'Pairs_mean_{i}' for i in range(4)],
    columns_bottom=[f'Pairs_dispersion_{i}' for i in range(4)],
    title_top='Частота пар бит: Средние значения',
    title_bottom='Частота пар бит: Дисперсии',
    xlabel='Индекс',
    ylabel_top='Средние значения',
    ylabel_bottom='Дисперсии',
    window_title='Частота пар бит в result',
    legend_labels_top=[f'{i:02b}' for i in range(4)],
    legend_labels_bottom=[f'{i:02b}' for i in range(4)],
    alpha_num=0.5,
    save_as='pairs_pair.png'
)

# Окно 5: Частота троек
plot_group_pair(
    data,
    columns_top=[f'Triples_mean_{i}' for i in range(8)],
    columns_bottom=[f'Triples_dispersion_{i}' for i in range(8)],
    title_top='Частота триад бит: Средние значения',
    title_bottom='Частота триад бит: Дисперсии',
    xlabel='Индекс',
    ylabel_top='Средние значения',
    ylabel_bottom='Дисперсии',
    window_title='Частота триад бит в result',
    legend_labels_top=[f'{i:03b}' for i in range(8)],
    legend_labels_bottom=[f'{i:03b}' for i in range(8)],
    alpha_num=0.5,
    save_as='triples_pair.png'
)

# Окно 6: Частота четвёрок
plot_group_pair(
    data,
    columns_top=[f'Quads_mean_{i}' for i in range(16)],
    columns_bottom=[f'Quads_dispersion_{i}' for i in range(16)],
    title_top='Частота тетрад бит: Средние значения',
    title_bottom='Частота тетрад бит: Дисперсии',
    xlabel='Индекс',
    ylabel_top='Средние значения',
    ylabel_bottom='Дисперсии',
    window_title='Частота тетрад бит в result',
    legend_labels_top=[f'{i:04b}' for i in range(16)],
    legend_labels_bottom=[f'{i:04b}' for i in range(16)],
    alpha_num=0.5,
    save_as='quads_pair.png'
)

# Окно 7: Энтропия
plot_group_pair(
    data,
    columns_top=[f"Entropy_mean_{i}" for i in range(64)],
    columns_bottom=[f"Entropy_stDev_{i}" for i in range(64)],
    title_top="Энтропия бит в result: Средние значения",
    title_bottom="Энтропия бит в result: Среднеквадратичные отклонения",
    xlabel="Индекс",
    ylabel_top="Средние значения",
    ylabel_bottom="Среднеквадратичные отклонения",
    window_title="Энтропия бит в result",
    show_legend_top=False,
    show_legend_bottom=False,
    alpha_num=0.3,
    save_as="entropy_pair.png"
)

# Окно 8: Глобальный тест частоты
plot_group_pair(
    data,
    columns_top=['GlobalStats_mean_0'],
    columns_bottom=['GlobalStats_dispersion_0'],
    title_top='Глобальный тест частоты: Средние значения',
    title_bottom='Глобальный тест частоты: Дисперсии',
    xlabel='Индекс',
    ylabel_top='Средние значения',
    ylabel_bottom='Дисперсии',
    window_title='Глобальный тест частоты Frequency Test (глобальный)',
    show_legend_top=False,
    show_legend_bottom=False,
    alpha_num=0.8,
    save_as='global_stats_pair.png'
)

# Окно 9: Тест последовательностей одинаковых бит
plot_group_pair(
    data,
    columns_top=['RunsTest_mean_0'],
    columns_bottom=['RunsTest_dispersion_0'],
    title_top='Тест последовательностей одинаковых бит: Средние значения',
    title_bottom='Тест последовательностей одинаковых бит: Дисперсии',
    xlabel='Индекс',
    ylabel_top='Средние значения',
    ylabel_bottom='Дисперсии',
    window_title='Тест последовательностей одинаковых бит (Runs Test) в result',
    show_legend_top=False,
    show_legend_bottom=False,
    alpha_num=0.8,
    save_as='runs_test.png'
)

# Окно 10: Блочный тест частоты
plot_group_pair(
    data,
    columns_top=[f'BlockFrequency_mean_{i}' for i in range(16)],
    columns_bottom=[f'BlockFrequency_dispersion_{i}' for i in range(16)],
    title_top='Блочный тест частоты: Средние значения для блоков',
    title_bottom='Блочный тест частоты: Дисперсии для блоков',
    xlabel='Индекс',
    ylabel_top='Средние значения',
    ylabel_bottom='Дисперсии',
    window_title='Блочный тест частоты для result - Block Frequency Test',
    legend_labels_top=[f'Block_{i}' for i in range(16)],
    legend_labels_bottom=[f'Block_{i}' for i in range(16)],
    alpha_num=0.3,
    save_as='block_frequency.png'
)

# Окно 11: Взаимная информация
plot_group_pair(
    data,
    columns_top=[f"MutualInformation_mean_{i}" for i in range(3)],
    columns_bottom=[f"MutualInformation_dispersion_{i}" for i in range(3)],
    title_top="Взаимная информация между состояниями s0, s1, и result: Средние значения",
    title_bottom="Взаимная информация между состояниями s0, s1, и result: Дисперсии",
    xlabel="Индекс",
    ylabel_top="Средние значения",
    ylabel_bottom="Дисперсии",
    window_title="Взаимная информация между состояниями s0, s1, и result",
    legend_labels_top=['I (s0;s1)','I (s0;res)','I (s1;res)'],
    legend_labels_bottom=['I (s0;s1)','I (s0;res)','I (s1;res)'],
    alpha_num=0.5,
    save_as="mutual_information_pair.png"
)

# Окно 12: Автокорреляция
plot_group_pair(
    data,
    columns_top=[f'Autocorrelation_mean_{i}' for i in range(1, 10)], 
    columns_bottom=[f'Autocorrelation_stDev_{i}' for i in range(1, 10)],
    title_top='Автокорреляция: Средние значения',
    title_bottom='Автокорреляция: Среднеквадратичные отклонения',
    xlabel='Индекс',
    ylabel_top='Средние значения',
    ylabel_bottom='Среднеквадратичные отклонения',
    window_title='Значения автокорреляции с различными сдвигами для result',
    legend_labels_top=[f'R({i},res)' for i in range(1, 10)], 
    legend_labels_bottom=[f'R({i},res)' for i in range(1, 10)], 
    alpha_num=0.35,
    save_as='autocorrelation.png'
)

# Окно 13: Разности
# Построение графика для Differences_mean_
plot_differences_with_intervals(
    data=data,
    columns=[f'Differences_mean_{i}' for i in range(256)],
    title='Анализ распределения разностей между последовательными значениями result',
    save_as='differences_analysis.png'
)

# Окно 14: Кластеры
plot_clusters_global(
    data=data,
    mean_columns=[f'Clusters_mean_{i}' for i in range(10)],
    distance_columns=[f'Clusters_distance_{i}_{(i + 1) % 10}' for i in range(10)],
    count_columns=[f'ClusterCount_mean_{i}' for i in range(10)],
    title='Кластеры: Средние значения, расстояния и количество элементов',
    xlabel='Кластеры',
    ylabel_mean='Средние значения центроидов',
    ylabel_distance='Расстояния между кластерами',
    ylabel_count='Количество элементов',
    window_title='Кластерный анализ',
    save_as='clusters_visualization_updated.png'
)
plot_detailed_clusters(
    data=data,
    mean_columns=[f'Clusters_mean_{i}' for i in range(10)],
    distance_columns=[f'Clusters_distance_{i}_{(i + 1) % 10}' for i in range(9)],
    count_columns=[f'ClusterCount_mean_{i}' for i in range(10)],
    title='Подробные данные о кластерах',
    save_as='clusters_detailed.png'
)


