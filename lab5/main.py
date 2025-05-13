import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from scipy.signal import butter, filtfilt

# Глобальні змінні
current_noise = None
# Зберігаємо параметри, використані для останньої генерації current_noise
current_noise_params = {"mean": None, "cov": None}
t = np.linspace(0, 10, 1000)

# Початкові параметри
initial_params = {
    "amplitude": 1.0,
    "frequency": 1.0,
    "phase": 0.0,
    "noise_mean": 0.0,
    "noise_cov": 0.1,
    "show_noise": True,
    "show_filtered": True,
    "filter_cutoff": 2.0,
}

# Розраховуємо частоту дискретизації з t один раз
sampling_freq = len(t) / (
    t[-1] - t[0]
)  # кількість точок / тривалість часу (1000/10 = 100 Гц)
nyquist_freq = 0.5 * sampling_freq  # 50 Гц


# Генерація гармоніки з шумом
def harmonic_with_noise(params):
    global current_noise, current_noise_params

    y_clean = params["amplitude"] * np.sin(
        2 * np.pi * params["frequency"] * t + params["phase"]
    )

    # Перевіряємо, чи потрібно згенерувати новий шум:
    noise_params_changed = (
        current_noise_params["mean"] != params["noise_mean"]
        or current_noise_params["cov"] != params["noise_cov"]
    )

    if current_noise is None or noise_params_changed:
        print(
            f"Generating NEW noise mean={params['noise_mean']:.2f}, cov={params['noise_cov']:.2f}"
        )
        np.random.seed(
            42
        )  # Фіксуємо seed для відтворюваності шуму при тих же параметрах
        current_noise = np.random.normal(
            params["noise_mean"], np.sqrt(params["noise_cov"]), len(t)
        )
        # Оновлюємо параметри, які були використані для цієї генерації
        current_noise_params["mean"] = params["noise_mean"]
        current_noise_params["cov"] = params["noise_cov"]
    else:
        # Перевірка на всякий випадок, хоча логіка вище має це покрити
        # Якщо параметри шуму не змінились, використовуємо той самий current_noise
        pass

    # Шумний сигнал
    y_noisy = y_clean + current_noise

    return y_clean, y_noisy


def apply_filter(y, cutoff, order=5):
    nyquist = 0.5 * 100  # Частота дискретизації (припущення)
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y_filtered = filtfilt(b, a, y)
    return y_filtered


# Налаштування графічного інтерфейсу
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  # 1 ряд, 2 стовпці
plt.subplots_adjust(
    left=0.08,
    right=0.95,
    bottom=0.35,
    wspace=0.2,  # Корегуємо bottom, бо немає одного слайдера
)  # wspace - простір між графіками

# Створення віджетів - позиціонуємо їх нижче обох графіків
ax_amplitude = plt.axes([0.1, 0.25, 0.65, 0.03])
ax_frequency = plt.axes([0.1, 0.20, 0.65, 0.03])
ax_phase = plt.axes([0.1, 0.15, 0.65, 0.03])
ax_noise_mean = plt.axes([0.1, 0.10, 0.65, 0.03])
ax_noise_cov = plt.axes([0.1, 0.05, 0.65, 0.03])
ax_filter_cutoff = plt.axes([0.1, 0.00, 0.65, 0.03])

sliders = {
    "amplitude": Slider(
        ax=ax_amplitude,
        label="Amplitude",
        valmin=0.1,
        valmax=5.0,
        valinit=initial_params["amplitude"],
    ),
    "frequency": Slider(
        ax=ax_frequency,
        label="Frequency",
        valmin=0.1,
        valmax=5.0,
        valinit=initial_params["frequency"],
    ),
    "phase": Slider(
        ax=ax_phase,
        label="Phase",
        valmin=0,
        valmax=2 * np.pi,
        valinit=initial_params["phase"],
    ),
    "noise_mean": Slider(  # Цей слайдер впливає на середнє нового шуму при регенерації
        ax=ax_noise_mean,
        label="Noise Mean",
        valmin=-1.0,
        valmax=1.0,
        valinit=initial_params["noise_mean"],
    ),
    "noise_cov": Slider(  # Цей слайдер впливає на коваріацію нового шуму при регенерації
        ax=ax_noise_cov,
        label="Noise Cov",
        valmin=0.0,
        valmax=1.0,
        valinit=initial_params["noise_cov"],
    ),
    "filter_cutoff": Slider(
        ax=ax_filter_cutoff,
        label="Filter Cutoff (Hz)",
        valmin=0.1,
        valmax=nyquist_freq,
        valinit=initial_params["filter_cutoff"],
    ),
}

# Кнопки та чекбокси
reset_ax = plt.axes([0.8, 0.2, 0.1, 0.04])  # Корегуємо позицію
reset_button = Button(reset_ax, "Reset")

check_ax = plt.axes([0.8, 0.25, 0.1, 0.1])  # Корегуємо позицію
check = CheckButtons(
    check_ax,
    ["Show Noise (Plot 1)", "Show Filtered (Plot 2)"],  # Змінено текст для ясності
    [initial_params["show_noise"], initial_params["show_filtered"]],
)


# Початковий графік
y_clean, y_noisy = harmonic_with_noise(initial_params)
# Використовуємо фіксований порядок при початковому розрахунку
y_filtered = apply_filter(y_noisy, initial_params["filter_cutoff"])

# --- ПЕРШИЙ ГРАФІК (Чистий + Шумний) ---
(line_clean,) = ax1.plot(t, y_clean, label="Clean", alpha=0.7)
(line_noisy,) = ax1.plot(
    t, y_noisy, label="Noisy", alpha=0.7, visible=initial_params["show_noise"]
)  # Видимість шумного за початковими параметрами
ax1.set_title("Clean and Noisy Signal")
ax1.set_xlabel("Time")
ax1.set_ylabel("Amplitude")
ax1.legend()
ax1.grid(True)


# --- ДРУГИЙ ГРАФІК (Відфільтрований) ---
(line_filtered,) = ax2.plot(
    t,
    y_filtered,
    label="Filtered",
    alpha=0.7,
    visible=initial_params["show_filtered"],
    color="green",  # Можна змінити колір
)
ax2.set_xlabel("Time")
ax2.set_ylabel("Amplitude")  # Зазвичай амплітуда після фільтрації схожа
ax2.legend()
ax2.grid(True)


# Функція оновлення
def update(val=None):
    # Зчитуємо всі поточні значення параметрів
    params = {
        "amplitude": sliders["amplitude"].val,
        "frequency": sliders["frequency"].val,
        "phase": sliders["phase"].val,
        "noise_mean": sliders["noise_mean"].val,
        "noise_cov": sliders["noise_cov"].val,
        "show_noise": check.get_status()[0],
        "show_filtered": check.get_status()[1],
        "filter_cutoff": sliders["filter_cutoff"].val,
        # "filter_order" тепер фіксований
    }

    # Важливо: Перегенеруємо тільки гармоніку. Шум залишається тим самим, якщо параметри шуму не змінились.
    # Функція harmonic_with_noise вже містить логіку для повторного використання шуму.
    y_clean_new, y_noisy_new = harmonic_with_noise(params)

    # Застосовуємо фільтр до шумного сигналу (з поточним шумом) з фіксованим порядком
    y_filtered_new = apply_filter(y_noisy_new, params["filter_cutoff"])

    # Оновлюємо дані на ГРАФІКАХ
    line_clean.set_ydata(y_clean_new)
    line_noisy.set_ydata(y_noisy_new)
    line_filtered.set_ydata(y_filtered_new)

    # Оновлюємо видимість ліній згідно чекбоксів
    line_noisy.set_visible(params["show_noise"])
    line_filtered.set_visible(params["show_filtered"])

    # Автомасштабування ВІСЕЙ для ОБОХ графіків
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()

    fig.canvas.draw_idle()


def reset(event):
    # Скидаємо значення слайдерів до початкових
    for name, slider in sliders.items():
        # Використовуємо .get() на випадок, якщо ключ відсутній (хоча тут всі ключі є)
        slider.set_val(initial_params.get(name, slider.val))

    # Скидаємо чекбокси
    check.set_active(0, initial_params["show_noise"])
    check.set_active(1, initial_params["show_filtered"])

    # Оновлюємо графік після скидання
    update()


# Підключення подій (для слайдерів, що залишились)
for slider in sliders.values():
    slider.on_changed(update)

reset_button.on_clicked(reset)
check.on_clicked(update)

plt.show()
