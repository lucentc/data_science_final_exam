import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Prediksi Harga Mobil", layout="wide")

st.title("PREDIKSI HARGA MOBIL")
st.markdown("---")

@st.cache_data
def latih_model_otomatis():
    df = pd.read_csv('Car_sales.csv')
    df.dropna(subset=['Price_in_thousands'], inplace=True)
    
    kolom_numerik = ['Engine_size', 'Horsepower', 'Fuel_efficiency']
    for col in kolom_numerik:
        df[col] = df[col].fillna(df[col].mean())
        
    X = df[kolom_numerik]
    y = df['Price_in_thousands']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_internal = LinearRegression()
    model_internal.fit(X_train, y_train)
    return model_internal

try:
    model = latih_model_otomatis()
    
    kolom_kiri, kolom_kanan = st.columns(2)
    
    with kolom_kiri:
        st.markdown("### **INPUT SPESIFIKASI MOBIL:**")
        engine_size = st.number_input("VARIABLE 1: Engine Size (Liter)", min_value=1.0, max_value=8.0, value=2.5, step=0.1)
        horsepower = st.number_input("VARIABLE 2: Horsepower (HP)", min_value=50, max_value=500, value=180, step=5)
        fuel_efficiency = st.number_input("VARIABLE 3: Fuel Efficiency (MPG)", min_value=10, max_value=50, value=28, step=1)
        
        st.write("")
        tombol_hitung = st.button("Hitung Harga Mobil", type="primary")
        
    with kolom_kanan:
        st.markdown("### **PERKIRAAN HARGA MOBIL:**")
        
        if tombol_hitung:
            input_data = pd.DataFrame({
                'Engine_size': [engine_size],
                'Horsepower': [horsepower],
                'Fuel_efficiency': [fuel_efficiency]
            })
            
            prediksi_harga = model.predict(input_data)[0]
            harga_usd_asli = prediksi_harga * 1000
            
            harga_formatted = f"${harga_usd_asli:,.0f}"
            
            st.markdown(
                f"""
                <div style="background-color: #ffffff; padding: 20px; border-radius: 6px; border: 1px solid #ccd1d9; box-shadow: 1px 1px 4px rgba(0,0,0,0.05); min-height: 180px;">
                    <h1 style="color: #333333; margin-top: 0; font-size: 40px; font-weight: bold; font-family: sans-serif;">{harga_formatted}</h1>
                    <br>
                    <p style="font-size: 15px; margin: 4px 0; font-family: sans-serif; color: #333333;"><b>VARIABLE 1 (Engine Size) :</b> {engine_size} Liter</p>
                    <p style="font-size: 15px; margin: 4px 0; font-family: sans-serif; color: #333333;"><b>VARIABLE 2 (Horsepower) :</b> {horsepower} HP</p>
                    <p style="font-size: 15px; margin: 4px 0; font-family: sans-serif; color: #333333;"><b>VARIABLE 3 (Fuel Efficiency) :</b> {fuel_efficiency} MPG</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="background-color: #f8f9fa; padding: 20px; border-radius: 6px; border: 1px dashed #ccd1d9; min-height: 180px; display: flex; align-items: center; justify-content: center;">
                    <p style="color: #6c757d; margin: 0; font-family: sans-serif;">Silakan klik tombol "Hitung Harga Mobil" untuk memproses.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

except Exception as e:
    st.error(f"Terjadi kesalahan sistem: {e}")

st.markdown("<br><br><hr>", unsafe_allow_html=True)
st.caption("Bajsan Arsyurrohman / 237006088")
