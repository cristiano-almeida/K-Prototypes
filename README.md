
# Protótipo de Clusterização de Dados Mistos com K-Prototypes

## 📌 Descrição
Este projeto apresenta um **protótipo funcional de clusterização para dados mistos** (numéricos e categóricos) usando o algoritmo **K-Prototypes**.  
O protótipo permite:

- Upload de datasets em formato CSV (`clientes.csv` e `academico.csv` já fornecidos como exemplo)  
- Pré-processamento automático (remoção de outliers, padronização, codificação de variáveis categóricas)  
- Seleção automática do número ideal de clusters (`K`) e do parâmetro gamma  
- Clusterização e análise detalhada (médias, distribuição categórica, variância explicada)  
- Visualização dos resultados via PCA para interpretação intuitiva  

## 📂 Estrutura do Repositório
```
.
├── notebook.py (copiar para arquivo .ipynb no Colab)
├── data/
│   ├── clientes.csv
│   └── academico.csv
├── README.md
```

## ⚙️ Pré-requisitos
- Google Colab ou Jupyter Notebook  
- Python 3.10+  
- Bibliotecas necessárias (listadas no `requirements.txt`):  
```
pandas
numpy
matplotlib
scikit-learn
kmodes
tqdm
```
- Conexão com a internet para instalar pacotes no Colab

## 🚀 Como Rodar
1. Abra o [Google Colab](https://colab.research.google.com/)  
2. Faça upload do notebook `clusterizacao_prototipo.ipynb`  
3. Faça upload dos datasets na pasta `data/` ou altere os caminhos no notebook  
4. Execute todas as células do notebook  

O protótipo realizará:  
- Pré-processamento automático  
- Detecção e remoção de outliers  
- Padronização de variáveis numéricas  
- Clusterização K-Prototypes  
- Análise detalhada e visualização via PCA  

## 📊 Datasets de Exemplo
1. **Clientes (`clientes.csv`)**  
   - Variáveis: `idade`, `compras_ultimo_ano`, `valor_total_gasto`, `pontuacao_fidelidade`, `canal_preferido`  
   - Objetivo: segmentar clientes para estratégias de marketing  

2. **Acadêmico (`academico.csv`)**  
   - Variáveis: `nota_final`, `faltas`, `perfil_social`, `ativ_extra`  
   - Objetivo: identificar perfis de desempenho e engajamento de alunos  

## 📈 Resultados Esperados
- Número ideal de clusters sugerido automaticamente  
- Análise detalhada por cluster:  
  - Quantidade de registros  
  - Médias das variáveis numéricas  
  - Distribuição de variáveis categóricas (Top 3 categorias)  
  - Indicadores de desempenho (gasto médio, nota média)  
- Visualização PCA mostrando separação dos clusters  
- Relatórios de outliers detectados e removidos  

## 📝 Observações Técnicas
- **Gamma (`γ`)** é calculado automaticamente para balancear a importância de variáveis categóricas e numéricas  
- **Outliers** são detectados usando Z-score (univariado) e LOF (multivariado)  
- **Padronização** é aplicada para melhorar a performance do PCA  
- **One-Hot Encoding** transforma variáveis categóricas para permitir cálculo de distâncias  

## 📌 Referências
- Huang, Z. (1997). *Clustering large data sets with mixed numeric and categorical values*.  
- [kmodes GitHub](https://github.com/nicodv/kmodes)  

## 👨‍💻 Autor
**Cristiano Almeida** – Protótipo desenvolvido para fins acadêmicos e de demonstração de clusterização em dados mistos.
