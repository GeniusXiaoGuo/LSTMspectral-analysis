import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False


# 从 Excel 文件读取光谱数据
data = pd.read_excel('ndfndf.xlsx')

# 提取输入特征和目标特征
# 假设数据的第一列是标签，后面的列是光谱数据
labels = data.iloc[:, 0]
spectra = data.iloc[:, 1:]

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_spectra = scaler.fit_transform(spectra)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(scaled_spectra, labels, test_size=0.2, random_state=42)

# 将数据调整为 LSTM 所需的 3D 张量形状 (samples, timesteps, features)
# 这里假设每个样本代表一个时间序列，特征数即为光谱波段数
# 可根据数据的具体形式进行调整
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 构建 LSTM 模型
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))  # 假设是一个回归问题，输出一个值
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=32)

# 在测试集上评估模型
loss = model.evaluate(X_test, y_test)
print('Test Loss:', loss)

# 使用模型进行预测
predictions = model.predict(X_test)

# 可视化预测结果
plt.figure(figsize=(10, 6))
plt.plot(predictions, label='Predicted')
plt.plot(y_test.values, label='True')
plt.xlabel('样本序号', fontsize=14)
plt.ylabel('光谱特征值', fontsize=14)
plt.title('光谱特征值预测结果', fontsize=16)
plt.legend()
plt.show()
