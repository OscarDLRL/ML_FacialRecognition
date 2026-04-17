import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from main import generar_dataset

# Cargar dataset
X, y = generar_dataset()

# Nombres de columnas
columnas = [
    "l_eye_x","l_eye_y","l_eye_w","l_eye_h","l_ear",
    "r_eye_x","r_eye_y","r_eye_w","r_eye_h","r_ear",
    "l_brow_x","l_brow_y","l_brow_curv",
    "r_brow_x","r_brow_y","r_brow_curv",
    "nose_x","nose_y","nose_length",
    "mouth_x","mouth_y","mouth_width","mouth_open","mouth_ratio","mouth_curv",
    "iod_norm","eye_open_mean","eye_open_diff",
    "left_brow_eye_dist","right_brow_eye_dist","brow_height_diff",
    "eye_size_diff","brow_inner_dist","brow_mean_height",
    "mouth_width_face","mouth_corner_balance"
]

# Crear DataFrame
df = pd.DataFrame(X, columns=columnas)

# Calcular VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = df.columns
vif_data["VIF"] = [
    variance_inflation_factor(df.values, i)
    for i in range(df.shape[1])
]

# Ordenar de mayor a menor
vif_data = vif_data.sort_values(by="VIF", ascending=False)

print("\n===== Variables con mayor VIF =====")
print(vif_data)

print("\n===== Variables problemáticas (VIF > 5) =====")
print(vif_data[vif_data["VIF"] > 5])

print("\n===== Variables muy redundantes (VIF > 10) =====")
print(vif_data[vif_data["VIF"] > 10])