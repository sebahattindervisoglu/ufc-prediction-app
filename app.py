# --- GEREKLİ KÜTÜPHANELER ---
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier

# --- MODEL VE VERİ SETLERİ YÜKLE ---
model = joblib.load("best_catboost_model.pkl")
scaler = joblib.load("robust_scaler.pkl")
model_columns = joblib.load("model_columns.pkl")
full_df = pd.read_csv("Datasets/large_dataset.csv")
athlete_df = pd.read_csv("Datasets/athlete_data.csv")

# --- GÖRSEL GETİRME ---
fighter_image_map = dict(zip(athlete_df["Athlete Name"], athlete_df["Image URL"]))

def get_fighter_image(fighter_name):
    url = fighter_image_map.get(fighter_name)
    if isinstance(url, str) and url.startswith("http"):
        return url
    else:
        return "https://cdn.pixabay.com/photo/2024/03/22/15/32/ai-generated-8649918_1280.png"

# --- KART GÖSTERME ---
def show_fighter_card(fighter_name, border_color, is_winner=False):
    img_url = get_fighter_image(fighter_name)
    trophy = " 🏆" if is_winner else ""
    st.markdown(
        f"""
        <div style="text-align: center; margin-bottom: 10px;">
            <div style="display:inline-block; border: 4px solid {border_color}; border-radius: 10px; padding:5px; background-color: #f9f9f9;">
                <img src="{img_url}" style="border-radius:10px; width:130px; height:130px; object-fit:cover;"/>
            </div>
            <div style="margin-top:5px; font-weight:bold; font-size:17px;">{fighter_name}{trophy}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- DÖVÜŞÇÜ BİLGİ GETİRME ---
def get_fighter_info(df, fighter, prefix):
    row = df[(df["r_fighter"] == fighter) | (df["b_fighter"] == fighter)]
    if row.empty:
        return {}
    info = {}
    wins = int(row[f"{prefix}_wins_total"].mean())
    losses = int(row[f"{prefix}_losses_total"].mean())
    total_matches = wins + losses
    win_rate = (wins / total_matches * 100) if total_matches > 0 else 0
    info["Galibiyet"] = wins
    info["Mağlubiyet"] = losses
    info["Toplam Maç"] = total_matches
    info["Galibiyet Oranı"] = f"%{win_rate:.2f}"
    info["Yaş"] = int(row[f"{prefix}_age"].mean())
    info["Boy"] = f"{row[f'{prefix}_height'].mean():.0f} cm"
    info["Kilo"] = f"{row[f'{prefix}_weight'].mean():.0f} kg"
    info["Stil"] = row[f"{prefix}_stance"].mode().iloc[0] if not row[f"{prefix}_stance"].mode().empty else "-"
    return info

# --- UYGULAMA BAŞLANGICI ---
st.set_page_config(page_title="🏆 UFC Dövüş Tahmini", layout="centered")
st.title("🏆 UFC Dövüş Tahmin Uygulaması")

fighter_gender_map = full_df[['r_fighter', 'gender']].drop_duplicates().set_index('r_fighter').to_dict()['gender']
fighter_gender_map.update(full_df[['b_fighter', 'gender']].drop_duplicates().set_index('b_fighter').to_dict()['gender'])
fighters = sorted(set(full_df['r_fighter'].unique()).union(set(full_df['b_fighter'].unique())))

# Cinsiyet seçimi
genders = sorted(full_df["gender"].dropna().unique())
selected_gender = st.selectbox("👤 Cinsiyet Seçin", genders)
filtered_fighters = sorted([f for f in fighters if fighter_gender_map.get(f) == selected_gender])

col1, col2 = st.columns(2)
with col1:
    red_fighter = st.selectbox("🔴 Red Köşe", filtered_fighters)
with col2:
    valid_blue_fighters = [f for f in filtered_fighters if f != red_fighter]
    blue_fighter = st.selectbox("🔵 Blue Köşe", valid_blue_fighters)

# Tahmin butonu ortalanmış
colA, colB, colC = st.columns([1,2,1])
with colB:
    tahmin = st.button("⚔️ Tahmini Yap", use_container_width=True)

if tahmin:
    # --- Hızlı Çözüm: Model inputunda alfabetik sıralama zorunluluğu ---
    fighter_a, fighter_b = sorted([red_fighter, blue_fighter])

    red_stats = full_df[(full_df["r_fighter"] == fighter_a) | (full_df["b_fighter"] == fighter_a)]
    blue_stats = full_df[(full_df["r_fighter"] == fighter_b) | (full_df["b_fighter"] == fighter_b)]

    numeric_cols = red_stats.select_dtypes(include=[np.number]).columns

    red_row = {col: red_stats[red_stats["r_fighter"] == fighter_a][col].mean() if col.startswith("r_") else red_stats[red_stats["b_fighter"] == fighter_a][col].mean() for col in numeric_cols}
    blue_row = {col: blue_stats[blue_stats["r_fighter"] == fighter_b][col].mean() if col.startswith("r_") else blue_stats[blue_stats["b_fighter"] == fighter_b][col].mean() for col in numeric_cols}

    red_row = pd.Series(red_row)
    blue_row = pd.Series(blue_row)

    sample = pd.DataFrame()
    for col in red_row.index:
        if col.startswith("r_"):
            sample[col] = [red_row[col]]
        elif col.startswith("b_"):
            sample[col] = [blue_row[col]]

    shared_cols = ["weight_class", "gender", "is_title_bout", "total_rounds"]
    for col in shared_cols:
        if col in full_df.columns:
            try:
                filtered = full_df[((full_df['r_fighter'] == fighter_a) & (full_df['b_fighter'] == fighter_b)) | ((full_df['r_fighter'] == fighter_b) & (full_df['b_fighter'] == fighter_a))]
                mode_val = filtered[col].mode()
                sample[col] = [mode_val.iloc[0] if not mode_val.empty else np.nan]
            except:
                sample[col] = [np.nan]
        else:
            sample[col] = [np.nan]

    sample["wins_total_diff"] = red_row.get("r_wins_total", 0) - blue_row.get("b_wins_total", 0)
    sample["losses_total_diff"] = red_row.get("r_losses_total", 0) - blue_row.get("b_losses_total", 0)
    sample["age_diff"] = red_row.get("r_age", 0) - blue_row.get("b_age", 0)
    sample["td_avg_diff"] = red_row.get("r_td_avg", 0) - blue_row.get("b_td_avg", 0)
    sample["kd_diff"] = red_row.get("r_kd", 0) - blue_row.get("b_kd", 0)

    for col in model_columns:
        if col not in sample.columns:
            sample[col] = 0
    sample = sample[model_columns]

    sample_scaled = scaler.transform(sample)
    proba = model.predict_proba(sample_scaled)[0][1]

    # Modelde fighter_a vs fighter_b input verildiği için, red/blue gösterimi kullanıcı tarafında tutuluyor
    winner_name = red_fighter if ((red_fighter == fighter_a and proba >= 0.5) or (red_fighter == fighter_b and proba < 0.5)) else blue_fighter
    winning_proba = proba if (winner_name == fighter_a) else 1 - proba

    # --- Kazananı göster ---
    st.success(f"🏆 Kazanan: {winner_name}")

    # Kazanma Olasılığı yazısında kazananın rengine göre ayarla
    kazanma_color = "#FF0000" if winner_name == red_fighter else "#0000FF"

    st.markdown(f"""
    <div style="text-align: center; margin-top: 30px; font-size: 24px; color: {kazanma_color};">
        <b>Kazanma Olasılığı</b>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align: center; margin-top: 10px; font-size: 40px; font-weight: bold; color: {kazanma_color};">
        %{winning_proba*100:.2f}
    </div>
    """, unsafe_allow_html=True)

    # --- Seçilen Dövüşçüler ve 🆚 İkonu ---
    st.markdown("---")
    col7, colM, col8 = st.columns([4,1,4])
    with col7:
        show_fighter_card(red_fighter, "#FF0000", is_winner=(red_fighter == winner_name))
    with colM:
        st.markdown("<div style='text-align:center; font-size:50px;'>🆚</div>", unsafe_allow_html=True)
    with col8:
        show_fighter_card(blue_fighter, "#0000FF", is_winner=(blue_fighter == winner_name))

    # --- Dövüşçü Özeti ---
    col9, col10 = st.columns(2)
    with col9:
        red_info = get_fighter_info(full_df, red_fighter, "r")
        for key, val in red_info.items():
            st.markdown(f"<span style='color:white;'>🔴 <b>{key}:</b> {val}</span>", unsafe_allow_html=True)
    with col10:
        blue_info = get_fighter_info(full_df, blue_fighter, "b")
        for key, val in blue_info.items():
            st.markdown(f"<span style='color:white;'>🔵 <b>{key}:</b> {val}</span>", unsafe_allow_html=True)
