import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Passo 1: Ler os dados do arquivo CSV
df = pd.read_csv('cultivares_soja.csv')

# Passo 2: Normalizar os dados
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.iloc[:, 1:])  # Normalizando a partir da segunda coluna

# Passo 3: Aplicar K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(scaled_features)

# Passo 4: Aplicar Hierarchical Clustering (Aglomerativo)
agglo = AgglomerativeClustering(n_clusters=3)
df['Agglo_Cluster'] = agglo.fit_predict(scaled_features)

# Passo 5: Visualizar os resultados
plt.figure(figsize=(14, 7))

# Plot KMeans
plt.subplot(1, 2, 1)
sns.scatterplot(x=df['Teor Médio de Proteína (%)'], y=df['Teor Médio de Óleo (%)'], hue=df['KMeans_Cluster'], palette='viridis')
for i in range(df.shape[0]):
    plt.text(df['Teor Médio de Proteína (%)'][i], df['Teor Médio de Óleo (%)'][i], df['Cultivar'][i], fontsize=9)
plt.title('K-Means Clustering')
plt.xlabel('Teor Médio de Proteína (%)')
plt.ylabel('Teor Médio de Óleo (%)')

# Plot Hierarchical Clustering
plt.subplot(1, 2, 2)
sns.scatterplot(x=df['Teor Médio de Proteína (%)'], y=df['Teor Médio de Óleo (%)'], hue=df['Agglo_Cluster'], palette='viridis')
for i in range(df.shape[0]):
    plt.text(df['Teor Médio de Proteína (%)'][i], df['Teor Médio de Óleo (%)'][i], df['Cultivar'][i], fontsize=9)
plt.title('Hierarchical Clustering')
plt.xlabel('Teor Médio de Proteína (%)')
plt.ylabel('Teor Médio de Óleo (%)')

# Caso queira plotar a tabela
#df_table = df[['Cultivar', 'KMeans_Cluster', 'Agglo_Cluster']]

# Passo 6: Plotar a tabela com os clusters
#plt.figure(figsize=(8, 4))

# Criar uma tabela usando matplotlib
#plt.axis('off')  # Remover os eixos
#tbl = plt.table(cellText=df_table.values, colLabels=df_table.columns, cellLoc='center', loc='center')

df[['Cultivar', 'KMeans_Cluster', 'Agglo_Cluster']].to_csv('cultivares_clusters.csv', index=False)


plt.tight_layout()
plt.show()
