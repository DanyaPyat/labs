import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from scipy.signal import butter, filtfilt

# Глобальні змінні для збереження стану шуму
current_noise = None
t = np.linspace(0, 10, 1000)

# Початкові параметри
initial_params = {
    "amplitude": 1.0,
    "frequency": 1.0,
    "phase": 0.0,
    "noise_mean": 0.0,
    "noise_cov": 0.1,
    "show_noise": True,
    "show_filtered": False,
    "filter_cutoff": 2.0,
}


# Генерація гармоніки з шумом
def harmonic_with_noise(params, regenerate_noise=False):
    global current_noise
    y_clean = params["amplitude"] * np.sin(
        2 * np.pi * params["frequency"] * t + params["phase"]
    )

    if regenerate_noise or (current_noise is None):
        np.random.seed(42)
        current_noise = np.random.normal(
            params["noise_mean"], np.sqrt(params["noise_cov"]), len(t)
        )

    y_noisy = y_clean + current_noise if params["show_noise"] else y_clean
    return y_clean, y_noisy


# Фільтрація сигналу
def apply_filter(y, cutoff, order=5):
    nyquist = 0.5 * 100  # Частота дискретизації (припущення)
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y_filtered = filtfilt(b, a, y)
    return y_filtered


# Налаштування графічного інтерфейсу
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.1, right=0.75, bottom=0.4)

# Створення віджетів
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
    "noise_mean": Slider(
        ax=ax_noise_mean,
        label="Noise Mean",
        valmin=-1.0,
        valmax=1.0,
        valinit=initial_params["noise_mean"],
    ),
    "noise_cov": Slider(
        ax=ax_noise_cov,
        label="Noise Cov",
        valmin=0.0,
        valmax=1.0,
        valinit=initial_params["noise_cov"],
    ),
    "filter_cutoff": Slider(
        ax=ax_filter_cutoff,
        label="Filter Cutoff",
        valmin=0.1,
        valmax=5.0,
        valinit=initial_params["filter_cutoff"],
    ),
}
# Кнопки та чекбокси
reset_ax = plt.axes([0.8, 0.2, 0.1, 0.04])
reset_button = Button(reset_ax, "Reset")

check_ax = plt.axes([0.8, 0.3, 0.1, 0.1])
check = CheckButtons(check_ax, ["Show Noise", "Show Filtered"], [True, False])

# Початковий графік
y_clean, y_noisy = harmonic_with_noise(initial_params)
(line_clean,) = ax.plot(t, y_clean, label="Clean", alpha=0.7)
(line_noisy,) = ax.plot(t, y_noisy, label="Noisy", alpha=0.7)
(line_filtered,) = ax.plot(t, y_clean, label="Filtered", alpha=0.7, visible=False)
ax.legend()


# Функції оновлення
def update(val=None):
    params = {
        "amplitude": sliders["amplitude"].val,
        "frequency": sliders["frequency"].val,
        "phase": sliders["phase"].val,
        "noise_mean": sliders["noise_mean"].val,
        "noise_cov": sliders["noise_cov"].val,
        "show_noise": check.get_status()[0],
        "filter_cutoff": sliders["filter_cutoff"].val,
    }

    regenerate_noise = any(
        [
            sliders["noise_mean"].val != initial_params["noise_mean"],
            sliders["noise_cov"].val != initial_params["noise_cov"],
        ]
    )

    y_clean_new, y_noisy_new = harmonic_with_noise(params, regenerate_noise)
    y_filtered_new = apply_filter(y_noisy_new, params["filter_cutoff"])

    line_clean.set_ydata(y_clean_new)
    line_noisy.set_ydata(y_noisy_new)
    line_filtered.set_ydata(y_filtered_new)
    line_filtered.set_visible(check.get_status()[1])

    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()


def reset(event):
    for name, slider in sliders.items():
        slider.set_val(initial_params.get(name, 0))
    check.set_active(0, initial_params["show_noise"])
    check.set_active(1, initial_params["show_filtered"])
    update()


# Підключення подій
for slider in sliders.values():
    slider.on_changed(update)

reset_button.on_clicked(reset)
check.on_clicked(update)

plt.show()
