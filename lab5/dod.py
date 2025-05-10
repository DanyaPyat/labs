import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt  # Використовуємо SciPy для фільтрації

# Ініціалізуємо Dash додаток
app = dash.Dash(__name__)

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


# Просте ковзне середнє з вікном 10 точок
def custom_moving_average(signal):
    # Згортка з ядром, що усереднює 10 сусідніх точок
    # mode="same" вирівнює вихідний розмір до вхідного
    return np.convolve(signal, np.ones(10) / 10, mode="same")


# Низькочастотний фільтр Баттерворта
def butterworth_filter(signal):
    # Розробляємо фільтр: 4-го порядку, нормалізована частота зрізу 0.1 (від Nyquist)
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
                            "🔄 Скинути",
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
                                {"label": "Фільтр Баттерворта", "value": "butterworth"},
                                {"label": "Ковзне середнє", "value": "custom"},
                                {"label": "Без фільтра", "value": "none"},
                            ],
                            value="none",
                            clearable=False,
                            style={"width": "200px", "margin": "10px"},
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
                                    max=6.28,
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
    t = np.linspace(0, 10, 1000)  # Часовий вектор

    # Генерація чистого та зашумленого сигналів
    y_clean = amp * np.sin(2 * np.pi * freq * t + phase)
    noise = np.random.normal(
        noise_mean, np.sqrt(noise_cov), len(t)
    )  # Використовуємо std dev (sqrt of variance)
    y_noisy = y_clean + noise

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
                x=t, y=y_noisy, name="Шум", line=dict(color="#e74c3c", dash="dot")
            )
        )
    if "show_filtered" in switches:
        # Показуємо відфільтрований сигнал лише якщо вибрано фільтр (або "none", де y_filtered=y_clean)
        # Або можна було б зробити y_filtered = None якщо filter_type == "none", і перевіряти тут.
        # Поточна логіка покаже чистий сигнал як "Фільтр", якщо вибрано "Без фільтра" та увімкнено "Показати фільтр".
        fig.add_trace(
            go.Scatter(
                x=t, y=y_filtered, name="Фільтр", line=dict(color="#3498db", width=2)
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
    if n_clicks:
        return (
            default_params["amplitude"],
            default_params["frequency"],
            default_params["phase"],
            default_params["noise_mean"],
            default_params["noise_cov"],
            ["show_noise", "show_filtered"],  # Скидаємо перемикачі до початкового стану
            "none",  # Скидаємо тип фільтра
        )
    # При першому завантаженні сторінки (n_clicks є None або 0) нічого не оновлюємо
    return dash.no_update


# Запуск додатка
if __name__ == "__main__":
    app.run(debug=True)
