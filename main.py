import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Superstore - Klasifikasi & Regresi", layout="wide")

# =========================
# LOAD MODEL & DATASET
# =========================
@st.cache_resource
def load_artifact(path: str):
    return joblib.load(path)

@st.cache_data
def load_dataset(csv_path: str):
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"File dataset tidak ditemukan: {csv_path}")
    try:
        return pd.read_csv(p, encoding="latin1")
    except Exception:
        return pd.read_csv(p)

# =========================
# FEATURE ENGINEERING
# =========================
def add_feature_engineering(df: pd.DataFrame):
    df = df.copy()

    needed_cols = {"Order Date", "Ship Date"}
    if not needed_cols.issubset(df.columns):
        missing = needed_cols - set(df.columns)
        raise ValueError(f"Kolom wajib tidak ada: {', '.join(missing)}")

    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"]  = pd.to_datetime(df["Ship Date"], errors="coerce")

    df["OrderYear"]  = df["Order Date"].dt.year
    df["OrderMonth"] = df["Order Date"].dt.month
    df["ShipDays"]   = (df["Ship Date"] - df["Order Date"]).dt.days

    if df["ShipDays"].isna().any():
        df["ShipDays"] = df["ShipDays"].fillna(df["ShipDays"].median())

    return df

def prepare_features(df_enriched: pd.DataFrame, drop_cols, target_col):
    df = df_enriched.copy()

    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # drop target jika ada (Segment untuk klasifikasi, Sales untuk regresi)
    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    return df

# =========================
# UI HEADER
# =========================
st.title("📌 Aplikasi Superstore — Klasifikasi Segment & Regresi Sales")
st.markdown(
    """
Aplikasi ini memiliki 2 mode:
- **Klasifikasi Segment**: memprediksi **Consumer / Corporate / Home Office**
- **Regresi Sales**: memprediksi **nilai Sales (angka)**

Model sudah disimpan sebagai **PKL**, jadi aplikasi hanya **load + predict** (tanpa training ulang).
"""
)

# =========================
# SIDEBAR PATH & MODE
# =========================
st.sidebar.header("⚙️ Sumber File (tanpa upload)")

csv_path = st.sidebar.text_input("File dataset (.csv)", "Sample - Superstore.csv")

st.sidebar.markdown("---")
st.sidebar.subheader("📦 File Model (.pkl)")
pkl_segment = st.sidebar.text_input("PKL Klasifikasi Segment", "superstore_segment_voting.pkl")
pkl_sales   = st.sidebar.text_input("PKL Regresi Sales (RF)", "superstore_reg_sales_rf.pkl")

st.sidebar.markdown("---")
mode = st.sidebar.radio("Pilih Mode", ["Klasifikasi Segment", "Regresi Sales"])

# =========================
# LOAD DATASET
# =========================
try:
    df_raw = load_dataset(csv_path)
except Exception as e:
    st.error(f"❌ Gagal load dataset: {e}")
    st.info("Pastikan CSV ada di folder yang sama dengan main.py, atau ubah path di sidebar.")
    st.stop()

# =========================
# DATASET OVERVIEW
# =========================
c1, c2, c3 = st.columns(3)
c1.metric("Jumlah Baris", f"{df_raw.shape[0]:,}")
c2.metric("Jumlah Kolom", f"{df_raw.shape[1]}")
c3.metric("Mode Aktif", mode)

st.subheader("🔎 Preview Dataset Asli (sebelum Feature Engineering)")
st.dataframe(df_raw.head(15), use_container_width=True)

# =========================
# FEATURE ENGINEERING DISPLAY
# =========================
st.markdown("---")
st.header("🛠️ Feature Engineering")
st.markdown(
    """
Kolom tanggal diubah jadi fitur:
- **OrderYear** (tahun order)
- **OrderMonth** (bulan order)
- **ShipDays** (selisih hari pengiriman - pemesanan)
"""
)

try:
    df_enriched = add_feature_engineering(df_raw)

    show_cols = [c for c in ["Order Date", "Ship Date", "OrderYear", "OrderMonth", "ShipDays"] if c in df_enriched.columns]
    st.subheader("Hasil Feature Engineering (contoh 20 baris)")
    st.dataframe(df_enriched[show_cols].head(20), use_container_width=True)

    st.subheader("Ringkasan ShipDays")
    st.write(df_enriched["ShipDays"].describe())
except Exception as e:
    st.error(f"❌ Feature engineering gagal: {e}")
    st.stop()

# =========================
# MODE: KLASIFIKASI SEGMENT
# =========================
if mode == "Klasifikasi Segment":
    st.markdown("---")
    st.header("🚀 Prediksi Segment (Klasifikasi)")

    try:
        artifact_cls = load_artifact(pkl_segment)
        model_cls = artifact_cls["model"]
        drop_cols_cls = artifact_cls["drop_cols"]
        target_cls = artifact_cls["target_col"]  # "Segment"
        st.sidebar.success("PKL Klasifikasi loaded ✅")
    except Exception as e:
        st.sidebar.error(f"Gagal load PKL Klasifikasi: {e}")
        st.stop()

    with st.expander("Lihat kolom input yang dipakai model klasifikasi"):
        X_for_model = prepare_features(df_enriched, drop_cols_cls, target_cls)
        st.write("Jumlah kolom input:", X_for_model.shape[1])
        st.write(list(X_for_model.columns))

    if st.button("Prediksi Segment", type="primary"):
        try:
            X_pred = prepare_features(df_enriched, drop_cols_cls, target_cls)
            preds = model_cls.predict(X_pred)

            out = df_raw.copy()
            out["Predicted Segment"] = preds

            tab1, tab2, tab3 = st.tabs(["📌 Ringkasan", "📄 Tabel Hasil", "⬇️ Download"])

            with tab1:
                st.subheader("Distribusi Hasil Prediksi Segment")
                st.write(out["Predicted Segment"].value_counts())

            with tab2:
                st.subheader("Preview Hasil Prediksi (50 baris)")
                st.dataframe(out.head(50), use_container_width=True)

            with tab3:
                st.subheader("Download Hasil Prediksi")
                st.download_button(
                    "Download hasil prediksi segment (.csv)",
                    out.to_csv(index=False).encode("utf-8"),
                    file_name="superstore_predicted_segment.csv",
                    mime="text/csv"
                )

            st.success("✅ Prediksi Segment selesai!")
        except Exception as e:
            st.error(f"❌ Error saat prediksi segment: {e}")

# =========================
# MODE: REGRESI SALES (RF)
# =========================
else:
    st.markdown("---")
    st.header("📈 Prediksi Sales (Regresi - Random Forest)")

    try:
        artifact_reg = load_artifact(pkl_sales)
        model_reg = artifact_reg["model"]
        drop_cols_reg = artifact_reg["drop_cols"]
        target_reg = artifact_reg["target_col"]  # "Sales"
        st.sidebar.success("PKL Regresi loaded ✅")
    except Exception as e:
        st.sidebar.error(f"Gagal load PKL Regresi: {e}")
        st.stop()

    st.info(f"Target regresi dari PKL: **{target_reg}**")

    with st.expander("Lihat kolom input yang dipakai model regresi"):
        X_for_model = prepare_features(df_enriched, drop_cols_reg, target_reg)
        st.write("Jumlah kolom input:", X_for_model.shape[1])
        st.write(list(X_for_model.columns))

    if st.button("Prediksi Sales", type="primary"):
        try:
            X_pred = prepare_features(df_enriched, drop_cols_reg, target_reg)
            preds = model_reg.predict(X_pred)

            out = df_raw.copy()
            out[f"Predicted {target_reg}"] = preds

            tab1, tab2, tab3 = st.tabs(["📌 Ringkasan", "📄 Tabel Hasil", "⬇️ Download"])

            with tab1:
                st.subheader(f"Ringkasan Prediksi {target_reg}")
                st.write(out[f"Predicted {target_reg}"].describe())

            with tab2:
                st.subheader("Preview Hasil Prediksi (50 baris)")
                st.dataframe(out.head(50), use_container_width=True)

            with tab3:
                st.subheader("Download Hasil Prediksi")
                st.download_button(
                    f"Download hasil prediksi {target_reg} (.csv)",
                    out.to_csv(index=False).encode("utf-8"),
                    file_name=f"superstore_predicted_{target_reg.lower()}.csv",
                    mime="text/csv"
                )

            st.success(f"✅ Prediksi {target_reg} selesai!")
        except Exception as e:
            st.error(f"❌ Error saat prediksi regresi: {e}")
