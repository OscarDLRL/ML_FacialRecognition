import pandas as pd
from sklearn.linear_model import LinearRegression
from main import generar_dataset

X, y = generar_dataset()

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

df = pd.DataFrame(X, columns=columnas)

vif_results = []

for i in range(df.shape[1]):
    y_var = df.iloc[:, i]
    X_vars = df.drop(df.columns[i], axis=1)

    model = LinearRegression()
    model.fit(X_vars, y_var)

    r2 = model.score(X_vars, y_var)

    if r2 >= 0.9999:
        vif = float('inf')
    else:
        vif = 1 / (1 - r2)

    vif_results.append((df.columns[i], vif))

vif_results = sorted(vif_results, key=lambda x: x[1], reverse=True)

print("\n===== Variables con mayor VIF =====")
for variable, vif in vif_results:
    print(f"{variable}: {vif:.2f}")