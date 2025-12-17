import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler

st.title("ИИ ДЗ1 Баранов")

def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

def remove_um(val):
    if val is None:
        return val
    if type(val) != str:
        return val
    parts = val.split()
    if len(parts) == 2:
        return float(parts[0])
    try:
        res = float(val)
        return res
    except ValueError:
        return None

page = st.sidebar.radio("Перейти к", ["Визуализации EDA", "Прогнозирование", "Веса модели"])

if page == "Визуализации EDA":
    st.header("EDA")

    def load_data():
        df_train = pd.read_csv(
            'https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
        return df_train

    df = load_data()

    df_viz = df.copy()
    for col in ['mileage', 'engine', 'max_power']:
        if df_viz[col].dtype == 'object':
            df_viz[col] = df_viz[col].apply(remove_um)

    df_viz['engine'] = pd.to_numeric(df_viz['engine'], errors='coerce').fillna(0).astype(int)
    df_viz['seats'] = pd.to_numeric(df_viz['seats'], errors='coerce').fillna(0).astype(int)

    viz_type = st.selectbox(
        "Выберите тип визуализации",
        ["Парные графики числовых признаков", "Тепловая карта корреляций", "Распределение цен"]
    )

    if viz_type == "Парные графики числовых признаков":
        numerical_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats', 'selling_price']
        fig = sns.pairplot(df_viz[numerical_cols].dropna(), diag_kind='kde')
        st.pyplot(fig)

    elif viz_type == "Тепловая карта корреляций":
        numerical_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats', 'selling_price']
        corr_matrix = df_viz[numerical_cols].dropna().corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, ax=ax)
        st.pyplot(fig)

    elif viz_type == "Распределение цен":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df_viz['selling_price'].dropna(), bins=50, kde=True, ax=ax)
        ax.set_xlabel('Цена')
        ax.set_ylabel('Количество')
        st.pyplot(fig)

elif page == "Прогнозирование":

    try:
        model_data = load_model()
        model = model_data['model']
        scaler = model_data['scaler_X']
        y_scaler = model_data['scaler_y']
        feature_names = model_data['feature_names']
        st.write("Модель успешно загружена.")
    except:
        st.write("Файл модели не найден.")
        st.stop()

    st.header("Прогноз")

    input_method = st.radio("Способ ввода:", ["Загрузить CSV", "Ручной ввод"])

    if input_method == "Загрузить CSV":
        uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            st.write(input_df.head())

            for col in ['mileage', 'engine', 'max_power']:
                if col in input_df.columns and input_df[col].dtype == 'object':
                    input_df[col] = input_df[col].apply(remove_um)

            if 'engine' in input_df.columns:
                input_df['engine'] = pd.to_numeric(input_df['engine'], errors='coerce').fillna(0).astype(int)
            if 'seats' in input_df.columns:
                input_df['seats'] = pd.to_numeric(input_df['seats'], errors='coerce').fillna(0).astype(int)

            input_features = input_df[feature_names]
            input_scaled = scaler.transform(input_features)
            predictions = model.predict(input_scaled)

            predictions = y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

            input_df['predicted_price'] = predictions
            st.write("Прогнозы:")
            st.write(input_df[feature_names + ['predicted_price']])

            csv = input_df.to_csv(index=False)
            st.download_button("Скачать прогнозы", csv, "predictions.csv", "text/csv")

    else:

        year = st.number_input("Год выпуска", value=2015)
        km_driven = st.number_input("Пробег", value=50000)
        mileage = st.number_input("Расход топлива", value=20.0)
        engine = st.number_input("Объем двигателя", value=1200)
        max_power = st.number_input("Мощность", value=80.0)
        seats = st.number_input("Количество мест", value=5)

        if st.button("Спрогнозировать цену"):
            input_data = pd.DataFrame({
                'year': [year],
                'km_driven': [km_driven],
                'mileage': [mileage],
                'engine': [engine],
                'max_power': [max_power],
                'seats': [seats]
            })

            input_scaled_X = scaler.transform(input_data[feature_names])
            predictions_scaled = model.predict(input_scaled_X)

            prediction = y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()[0]

            st.write(f"Прогноз: {prediction:.2f}")

elif page == "Веса модели":
    st.header("Коэффиценты модели")

    try:
        model_data = load_model()
        model = model_data['model']
        scaler = model_data['scaler_X']
        y_scaler = model_data['scaler_y']
        feature_names = model_data['feature_names']
        st.write("Модель успешно загружена.")
    except:
        st.write("Файл модели не найден.")
        st.stop()

    coefficients = model.coef_
    intercept = model.intercept_ if hasattr(model, 'intercept_') else 0

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(coefficients)), coefficients)
    ax.set_xlabel('Признаки')
    ax.set_ylabel('Значение коэффициента')
    ax.set_title('Коэффициенты признаков модели')
    ax.set_xticks(range(len(coefficients)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    st.pyplot(fig)

    coef_df = pd.DataFrame({
        'Признак': feature_names,
        'Коэффициент': coefficients,
        'Абс_коэффициент': np.abs(coefficients)
    }).sort_values('Абс_коэффициент', ascending=False)
    st.write(coef_df[['Признак', 'Коэффициент']])