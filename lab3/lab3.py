import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Агро-аналітика",
    page_icon="🌻",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("E:\\uni\\anal\\lab3\\out.csv")
        df["year_week"] = (
            df["year"].astype(str) + "-W" + df["week"].astype(int).astype(str)
        )
        df["date"] = pd.to_datetime(df["year_week"] + "-1", format="%Y-W%W-%w")
        return df
    except Exception as e:
        st.error(f"Помилка завантаження даних: {e}")
        return pd.DataFrame()


data = load_data()

# Initialize parameters
default_params = {
    "ts": "VCI",
    "region": "Вінницька",
    "weeks": (1, 52),
    "years": (data["year"].min(), data["year"].max()),
}

if "params" not in st.session_state:
    st.session_state.params = default_params.copy()

# Sidebar
with st.sidebar:
    st.header("Панель управління")

    if st.button("Скинути до стандартних", use_container_width=True):
        st.session_state.params = default_params.copy()
        st.rerun()

    st.divider()

    st.session_state.params["ts"] = st.selectbox(
        "Вибір метрики",
        options=["VCI", "TCI", "VHI"],
        index=["VCI", "TCI", "VHI"].index(st.session_state.params["ts"]),
        help="Оберіть індекс для аналізу.",
    )

    st.session_state.params["region"] = st.selectbox(
        "Оберіть область",
        options=data["region_name"].unique(),
        index=list(data["region_name"].unique()).index(
            st.session_state.params["region"]
        ),
        help="Фільтр за регіоном.",
    )

    st.divider()

    st.session_state.params["weeks"] = st.slider(
        "Діапазон тижнів",
        min_value=1,
        max_value=52,
        value=st.session_state.params["weeks"],
        format="Тиждень %d",
        help="Виберіть період для аналізу.",
    )

    st.session_state.params["years"] = st.slider(
        "Діапазон років",
        min_value=int(data["year"].min()),
        max_value=int(data["year"].max()),
        value=st.session_state.params["years"],
        format="Рік %d",
        help="Виберіть роки для аналізу.",
    )

    st.divider()

    # Sorting options
    with st.expander("Параметри сортування"):
        asc = st.checkbox(
            "За зростанням", value=False, help="Від найменшого до найбільшого."
        )
        desc = st.checkbox(
            "За спаданням", value=False, help="Від найбільшого до найменшого."
        )

# Main panel
try:
    filtered = data[
        (data["region_name"] == st.session_state.params["region"])
        & (data["week"].between(*st.session_state.params["weeks"]))
        & (data["year"].between(*st.session_state.params["years"]))
    ]

    if asc and desc:
        st.warning("Будь ласка, оберіть лише один тип сортування.")
    elif asc:
        filtered = filtered.sort_values(st.session_state.params["ts"], ascending=True)
    elif desc:
        filtered = filtered.sort_values(st.session_state.params["ts"], ascending=False)

except Exception as e:
    st.error(f"Помилка обробки даних: {str(e)}")

tab1, tab2, tab3 = st.tabs(
    ["📋 Таблиця даних", "📈 Динаміка змін", "📊 Порівняння регіонів"]
)

with tab1:
    st.subheader("Детальні дані")
    st.dataframe(
        filtered[
            ["date", "year", "week", st.session_state.params["ts"], "region_name"]
        ],
        column_config={
            "date": st.column_config.DateColumn("Дата"),
            "year": "Рік",
            "week": "Тиждень",
            st.session_state.params["ts"]: f"Значення {st.session_state.params['ts']}",
            "region_name": "Регіон",
        },
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    st.subheader("Динаміка змін")
    if not filtered.empty:
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=filtered,
            x="date",
            y=st.session_state.params["ts"],
            marker="o",
            color="green",
        )
        plt.title(
            f"Динаміка {st.session_state.params['ts']} для {st.session_state.params['region']}"
        )
        plt.xlabel("Дата")
        plt.ylabel("Значення індексу")
        plt.grid()
        st.pyplot(plt)
    else:
        st.info("Немає даних для відображення.")

with tab3:
    st.subheader("Порівняння регіонів")
    try:
        # Filter data for selected weeks and years
        compare_data = data[
            (data["week"].between(*st.session_state.params["weeks"]))
            & (data["year"].between(*st.session_state.params["years"]))
        ]

        # Prepare data for bar chart
        bar_data = (
            compare_data.groupby("region_name")[st.session_state.params["ts"]]
            .mean()
            .reset_index()
        )

        # Create bar chart
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=bar_data,
            x="region_name",
            y=st.session_state.params["ts"],
            palette="viridis",
        )
        plt.title(f"Середнє значення {st.session_state.params['ts']} по областях")
        plt.xlabel("Область")
        plt.ylabel("Середнє значення індексу")
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y")
        st.pyplot(plt)

    except Exception as e:
        st.error(f"Помилка побудови графіка: {str(e)}")
