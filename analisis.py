import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = {
    "Tanggal": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03", 
                "2024-01-04", "2024-01-04", "2024-01-05", "2024-01-05", "2024-01-06", "2024-01-06",
                "2024-01-07", "2024-01-07", "2024-01-08", "2024-01-08", "2024-01-09", "2024-01-09",
                "2024-01-10", "2024-01-10", "2024-01-11", "2024-01-11", "2024-01-12", "2024-01-12",
                "2024-01-13", "2024-01-13", "2024-01-14", "2024-01-14", "2024-01-15", "2024-01-15"],
    "Produk": ["Mochi", "Egg Tart", "Mochi", "Egg Tart", "Mochi", "Egg Tart",
               "Mochi", "Egg Tart", "Mochi", "Egg Tart", "Mochi", "Egg Tart",
               "Mochi", "Egg Tart", "Mochi", "Egg Tart", "Mochi", "Egg Tart",
               "Mochi", "Egg Tart", "Mochi", "Egg Tart", "Mochi", "Egg Tart",
               "Mochi", "Egg Tart", "Mochi", "Egg Tart", "Mochi", "Egg Tart"],
    "Jumlah Terjual": [50, 30, 45, 35, 60, 40, 55, 32, 70, 45, 65, 50,
                       80, 60, 75, 55, 85, 65, 90, 70, 95, 75, 100, 80,
                       110, 85, 105, 90, 120, 95],
    "Harga per Unit": [5.00, 3.00, 5.00, 3.00, 5.00, 3.00, 5.00, 3.00, 5.00, 3.00, 5.00, 3.00,
                       5.00, 3.00, 5.00, 3.00, 5.00, 3.00, 5.00, 3.00, 5.00, 3.00, 5.00, 3.00,
                       5.00, 3.00, 5.00, 3.00, 5.00, 3.00],
    "Total Pendapatan": [250.00, 90.00, 225.00, 105.00, 300.00, 120.00, 275.00, 96.00, 350.00, 135.00, 
                         325.00, 150.00, 400.00, 180.00, 375.00, 165.00, 425.00, 195.00, 450.00, 210.00, 
                         475.00, 225.00, 500.00, 240.00, 550.00, 255.00, 525.00, 270.00, 600.00, 285.00]
}

df = pd.DataFrame(data)

plt.figure(figsize=(14, 7))
sns.lineplot(data=df, x="Tanggal", y="Jumlah Terjual", hue="Produk", marker="o")
plt.title('Jumlah Penjualan Harian Mochi dan Egg Tart')
plt.xticks(rotation=45)
plt.ylabel('Jumlah Terjual')
plt.xlabel('Tanggal')
plt.legend(title='Produk')
plt.show()

plt.figure(figsize=(14, 7))
sns.lineplot(data=df, x="Tanggal", y="Total Pendapatan", hue="Produk", marker="o")
plt.title('Total Pendapatan Harian Mochi dan Egg Tart')
plt.xticks(rotation=45)
plt.ylabel('Total Pendapatan')
plt.xlabel('Tanggal')
plt.legend(title='Produk')
plt.show()

# Memisahkan data untuk Mochi dan Egg Tart
df_mochi = df[df['Produk'] == 'Mochi']
df_egg_tart = df[df['Produk'] == 'Egg Tart']

# Fungsi untuk melakukan regresi linear dan visualisasinya
def regresi_linear(df, produk):
    X = df[['Jumlah Terjual']]
    y = df['Total Pendapatan']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"Model Regresi Linear untuk {produk}:")
    print("Koefisien:", model.coef_[0])
    print("Intercept:", model.intercept_)
    print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
    print("R^2 Score:", r2_score(y_test, y_pred))

    # Plot hasil regresi
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_test['Jumlah Terjual'], y=y_test, color='blue', label='Data Aktual')
    sns.lineplot(x=X_test['Jumlah Terjual'], y=y_pred, color='red', label='Prediksi Regresi')
    plt.title(f'Regresi Linear untuk {produk}')
    plt.xlabel('Jumlah Terjual')
    plt.ylabel('Total Pendapatan')
    plt.legend()
    plt.show()

# Regresi Linear untuk Mochi
regresi_linear(df_mochi, 'Mochi')

# Regresi Linear untuk Egg Tart
regresi_linear(df_egg_tart, 'Egg Tart')
