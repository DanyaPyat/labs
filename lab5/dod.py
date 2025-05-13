import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt  # Використовуємо SciPy для фільтрації

# Ініціалізуємо Dash додаток
app = dash.Dash(__name__)

# Глобальні змінні для збереження стану шуму
current_noise = None
# Зберігаємо параметри, використані для останньої генерації current_noise
current_noise_params = {"mean": None, "cov": None}

# Параметри за замовчуванням для генерації сигналу та шуму
default_params = {
    "amplitude": 1.0,
    "frequency": 1.0,  # Гц
    "phase": 0.0,  # радіани
    "noise_mean": 0.0,
    "noise_cov": 0.1,  # Дисперсія шуму
    "show_noise": True,
    "show_filtered": True,
    "filter_type": "none",  # Початковий тип фільтра
}


# --- Функції фільтрації ---
# ковзне середнє з вікном 10 точок
def custom_moving_average(signal):
    return np.convolve(
        signal, np.ones(10) / 10, mode="same"
    )  # mode="same" вирівнює вихідний розмір до вхідного


# Низькочастотний фільтр Баттерворта
def butterworth_filter(signal):
    b, a = butter(4, 0.1, btype="low")
    # Застосовуємо фільтр двічі (вперед і назад) для отримання нульового фазового зсуву
    return filtfilt(b, a, signal)


# --- Макет інтерфейсу ---

app.layout = html.Div(
    [
        html.Div(  # Основний контейнер
            [
                html.H1(
                    "Гармонічний аналізатор",
                    style={"textAlign": "center", "color": "#2c3e50"},
                ),
                # Графік відображення сигналу
                dcc.Graph(
                    id="signal-plot",
                    style={"height": "60vh", "border": "1px solid #dfe6e9"},
                ),
                # Панель керування: кнопка скидання, перемикачі відображення, вибір фільтра
                html.Div(
                    [
                        html.Button(
                            "Скинути",
                            id="reset-button",
                            style={
                                "width": "120px",
                                "margin": "10px",
                                "background": "#3498db",
                                "color": "white",
                                "border": "none",
                                "borderRadius": "5px",
                                "cursor": "pointer",
                            },
                        ),
                        dcc.Checklist(
                            id="switches",
                            options=[
                                {"label": " Показати шум", "value": "show_noise"},
                                {"label": " Показати фільтр", "value": "show_filtered"},
                            ],
                            value=["show_noise", "show_filtered"],
                            labelStyle={"display": "inline-block", "margin": "0 15px"},
                            style={"display": "flex", "alignItems": "center"},
                        ),
                        dcc.Dropdown(
                            id="filter-type",
                            options=[
                                {
                                    "label": "Без фільтра",
                                    "value": "none",
                                },  # Додано опцію "Без фільтра"
                                {
                                    "label": "Фільтр Баттерворта (НЧ)",
                                    "value": "butterworth",
                                },  # Уточнено назву
                                {
                                    "label": "Ковзне середнє (10 точок)",
                                    "value": "custom",
                                },  # Уточнено назву
                            ],
                            value="none",
                            clearable=False,
                            style={
                                "width": "250px",
                                "margin": "10px",
                            },  # Розширено ширину
                        ),
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "center",
                        "alignItems": "center",
                        "flexWrap": "wrap",
                    },
                ),
                # Контейнер з повзунками для налаштування параметрів
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Амплітуда", style={"fontWeight": "bold"}),
                                dcc.Slider(
                                    id="amplitude",
                                    min=0.1,
                                    max=5.0,
                                    step=0.1,
                                    value=1.0,
                                    marks={i: f"{i}" for i in [0, 1, 2, 3, 4, 5]},
                                ),
                            ],
                            className="slider-container",
                        ),
                        html.Div(
                            [
                                html.Label(
                                    "Частота (Гц)", style={"fontWeight": "bold"}
                                ),
                                dcc.Slider(
                                    id="frequency",
                                    min=0.1,
                                    max=5.0,
                                    step=0.1,
                                    value=1.0,
                                    marks={i: f"{i}" for i in [0, 1, 2, 3, 4, 5]},
                                ),
                            ],
                            className="slider-container",
                        ),
                        html.Div(
                            [
                                html.Label("Фаза (рад)", style={"fontWeight": "bold"}),
                                dcc.Slider(
                                    id="phase",
                                    min=0,
                                    max=6.28,  # Приблизно 2*pi
                                    step=0.1,
                                    value=0.0,
                                    marks={0: "0", 3.14: "π", 6.28: "2π"},
                                ),
                            ],
                            className="slider-container",
                        ),
                        html.Div(
                            [
                                html.Label("μ шуму", style={"fontWeight": "bold"}),
                                dcc.Slider(
                                    id="noise-mean",
                                    min=-1.0,
                                    max=1.0,
                                    step=0.1,
                                    value=0.0,
                                    marks={-1: "-1", 0: "0", 1: "1"},
                                ),
                            ],
                            className="slider-container",
                        ),
                        html.Div(
                            [
                                html.Label("σ² шуму", style={"fontWeight": "bold"}),
                                dcc.Slider(
                                    id="noise-cov",
                                    min=0.0,
                                    max=1.0,
                                    step=0.01,
                                    value=0.1,
                                    marks={0: "0", 0.5: "0.5", 1: "1"},
                                ),
                            ],
                            className="slider-container",
                        ),
                    ],
                    style={
                        "padding": "20px",
                        "backgroundColor": "#f8f9fa",
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(250px, 1fr))",
                        "gap": "20px",
                    },
                ),
            ],
            style={"maxWidth": "1200px", "margin": "0 auto"},
        )
    ]
)

# --- Callbacks ---


# Оновлення графіку при зміні будь-якого контрола параметрів чи відображення
@app.callback(
    Output("signal-plot", "figure"),
    [
        Input("amplitude", "value"),
        Input("frequency", "value"),
        Input("phase", "value"),
        Input("noise-mean", "value"),
        Input("noise-cov", "value"),
        Input("switches", "value"),
        Input("filter-type", "value"),
    ],
)
def update_plot(amp, freq, phase, noise_mean, noise_cov, switches, filter_type):
    global current_noise, current_noise_params
    t = np.linspace(0, 10, 1000)  # Часовий вектор (фіксована довжина і тривалість)

    # Перевіряємо, чи потрібно згенерувати новий шум:
    noise_params_changed = (
        current_noise_params["mean"] != noise_mean
        or current_noise_params["cov"] != noise_cov
    )

    if current_noise is None or noise_params_changed:
        print(f"Generating NEW noise mean={noise_mean:.2f}, cov={noise_cov:.2f}")
        np.random.seed(
            42
        )  # Фіксуємо seed для відтворюваності шуму при тих же параметрах
        current_noise = np.random.normal(noise_mean, np.sqrt(noise_cov), len(t))
        # Оновлюємо параметри, які були використані для цієї генерації
        current_noise_params["mean"] = noise_mean
        current_noise_params["cov"] = noise_cov
    else:
        print("Using EXISTING noise realization.")

    # Генерація чистого сигналу (залежить від амплітуди, частоти, фази)
    y_clean = amp * np.sin(2 * np.pi * freq * t + phase)

    # Зашумлений сигнал (використовуємо поточний шум)
    y_noisy = y_clean + current_noise

    # Застосування вибраного фільтра або використання чистого сигналу
    if filter_type == "butterworth":
        y_filtered = butterworth_filter(y_noisy)
    elif filter_type == "custom":
        y_filtered = custom_moving_average(y_noisy)
    else:
        y_filtered = y_clean  # Режим "Без фільтра"

    fig = go.Figure()

    # Додавання трас до графіку відповідно до налаштувань видимості
    fig.add_trace(
        go.Scatter(x=t, y=y_clean, name="Чистий", line=dict(color="#2ecc71", width=2))
    )
    if "show_noise" in switches:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=y_noisy,
                name="Шумний",
                line=dict(
                    color="#e74c3c", dash="dot"
                ),  # Змінено назву на "Шумний" для ясності
            )
        )
    # Показуємо відфільтрований сигнал, якщо чекбокс "Показати фільтр" увімкнено
    if "show_filtered" in switches:
        # Визначаємо назву траси залежно від вибраного фільтра
        filter_name = {
            "none": "Чистий (Без фільтра)",  # Назва, якщо вибрано "none" але відображаємо
            "butterworth": "Відфільтрований (Баттерворт)",
            "custom": "Відфільтрований (Ковзне середнє)",
        }.get(
            filter_type, "Відфільтрований"
        )  # Запасний варіант назви

        fig.add_trace(
            go.Scatter(
                x=t, y=y_filtered, name=filter_name, line=dict(color="#3498db", width=2)
            )
        )

    # Налаштування макету графіку
    fig.update_layout(
        title="Динаміка сигналу",
        xaxis_title="Час (с)",
        yaxis_title="Амплітуда",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        # Масштабуємо вісь Y автоматично, щоб вмістити всі видимі траси
        yaxis=dict(autorange=True),
    )
    return fig


# Скидання параметрів на UI до значень за замовчуванням
@app.callback(
    [
        Output("amplitude", "value"),
        Output("frequency", "value"),
        Output("phase", "value"),
        Output("noise-mean", "value"),
        Output("noise-cov", "value"),
        Output("switches", "value"),
        Output("filter-type", "value"),
    ],
    [Input("reset-button", "n_clicks")],
)
def reset_params(n_clicks):
    # Виконуємо скидання тільки після першого кліка і наступних
    if n_clicks is not None and n_clicks > 0:  # Перевірка, що кнопка була натиснута
        # Не скидаємо глобальні змінні current_noise, вони оновляться в update_plot при зміні параметрів шуму
        return (
            default_params["amplitude"],
            default_params["frequency"],
            default_params["phase"],
            default_params["noise_mean"],
            default_params["noise_cov"],
            ["show_noise", "show_filtered"],  # Скидаємо перемикачі до початкового стану
            "none",  # Скидаємо тип фільтра
        )
    # При першому завантаженні сторінки або якщо n_clicks None/0, нічого не оновлюємо
    return (
        dash.no_update,
        dash.no_update,
        dash.no_update,
        dash.no_update,
        dash.no_update,
        dash.no_update,
        dash.no_update,
    )


# Запуск додатка
if __name__ == "__main__":
    app.run(debug=True)
