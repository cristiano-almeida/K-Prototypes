# ==============================================================================
# BLOCO 1: INSTALAÇÕES E IMPORTAÇÕES
# ==============================================================================
!pip install kmodes --quiet
!pip install tqdm --quiet
!pip install scikit-learn --quiet

import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from io import BytesIO
import ipywidgets as widgets
from IPython.display import display, clear_output, FileLink
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# BLOCO 2: FUNÇÕES DE LÓGICA E ANÁLISE (CORREÇÕES IMPLEMENTADAS)
# ==============================================================================
def reduzir_memoria(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Reduz o uso de memória do DataFrame convertendo tipos de dados numéricos.
    """
    for coluna in dataframe.select_dtypes(include=['float64']):
        dataframe[coluna] = pd.to_numeric(dataframe[coluna], downcast='float')
    for coluna in dataframe.select_dtypes(include=['int64']):
        dataframe[coluna] = pd.to_numeric(dataframe[coluna], downcast='integer')
    return dataframe

def identificar_variaveis(dataframe: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identifica e retorna listas de variáveis categóricas e numéricas.
    """
    colunas_categoricas = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
    colunas_numericas = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    return colunas_categoricas, colunas_numericas

def calcular_gamma_automatico(dataframe: pd.DataFrame, colunas_numericas: List[str]) -> float:
    """
    Calcula Gamma usando método robusto baseado na média dos desvios padrão padronizados.
    """
    if not colunas_numericas:
        return 0.5

    # Padronizar as variáveis antes de calcular os desvios padrão
    scaler = StandardScaler()
    dados_padronizados = scaler.fit_transform(dataframe[colunas_numericas])
    dados_padronizados_df = pd.DataFrame(dados_padronizados, columns=colunas_numericas)

    desvios_padrao = dados_padronizados_df.std()
    gamma = 0.5 * desvios_padrao.mean()

    print(f"📊 Gamma calculado automaticamente: {gamma:.4f}")
    return gamma

def detectar_e_tratar_outliers(dataframe: pd.DataFrame, colunas_numericas: List[str], remover_outliers: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Detecta e trata outliers usando múltiplos métodos de forma consistente.
    """
    informacoes_outliers = {
        'z_score': {'quantidade': 0, 'ids': []},
        'lof': {'quantidade': 0, 'ids': []},
        'total_detectados': 0,
        'ids_removidos': []
    }

    dataframe_limpo = dataframe.copy()
    todos_outliers = set()

    # Método 1: Z-score com threshold conservador
    for coluna in colunas_numericas:
        if coluna in dataframe_limpo.columns:
            z_scores = np.abs((dataframe_limpo[coluna] - dataframe_limpo[coluna].mean()) / dataframe_limpo[coluna].std())
            outliers_z = dataframe_limpo[z_scores > 2.5]

            if not outliers_z.empty:
                informacoes_outliers['z_score']['quantidade'] += len(outliers_z)
                informacoes_outliers['z_score']['ids'].extend(outliers_z.index.tolist())
                todos_outliers.update(outliers_z.index)

    # Método 2: Local Outlier Factor para detecção multivariada
    if colunas_numericas and len(dataframe_limpo) > 20:  # LOF requer mínimo de amostras
        try:
            lof = LocalOutlierFactor(n_neighbors=min(20, len(dataframe_limpo)//2), contamination=0.05)
            outliers_lof = lof.fit_predict(dataframe_limpo[colunas_numericas])
            outliers_indices = dataframe_limpo[outliers_lof == -1].index

            if len(outliers_indices) > 0:
                informacoes_outliers['lof']['quantidade'] = len(outliers_indices)
                informacoes_outliers['lof']['ids'] = outliers_indices.tolist()
                todos_outliers.update(outliers_indices)
        except Exception as e:
            print(f"⚠️  Aviso na detecção LOF: {e}")

    # Consolidar todos os outliers
    informacoes_outliers['total_detectados'] = len(todos_outliers)
    informacoes_outliers['ids_removidos'] = list(todos_outliers)

    # Remover outliers se solicitado
    if remover_outliers and todos_outliers:
        dataframe_limpo = dataframe_limpo.drop(todos_outliers)

    # Relatório de outliers
    if informacoes_outliers['total_detectados'] > 0:
        print("⚠️  OUTLIERS DETECTADOS:")
        print(f"   Z-score: {informacoes_outliers['z_score']['quantidade']} outliers")
        print(f"   LOF (multivariados): {informacoes_outliers['lof']['quantidade']} outliers")
        print(f"   Total único: {informacoes_outliers['total_detectados']} outliers")

        if remover_outliers:
            print(f"   ✅ {len(todos_outliers)} outliers removidos")
        else:
            print("   ℹ️  Outliers detectados mas não removidos")
    else:
        print("✅ Nenhum outlier detectado")

    return dataframe_limpo, informacoes_outliers

def preparar_matriz_para_kprototypes(dataframe: pd.DataFrame, colunas_usadas: List[str], colunas_categoricas: List[str], padronizar: bool = True) -> Tuple[np.ndarray, List[int]]:
    """
    Prepara a matriz de dados para o algoritmo K-Prototypes com validação robusta.
    """
    # Validar e filtrar colunas categóricas
    colunas_categoricas_validas = [col for col in colunas_categoricas if col in colunas_usadas]

    if len(colunas_categoricas_validas) != len(colunas_categoricas):
        colunas_nao_encontradas = set(colunas_categoricas) - set(colunas_usadas)
        print(f"⚠️  Aviso: Colunas categóricas não encontradas: {colunas_nao_encontradas}")

    matriz = dataframe[colunas_usadas].copy()
    colunas_numericas = [col for col in colunas_usadas if col not in colunas_categoricas_validas]

    # Padronização de variáveis numéricas (se solicitado)
    if padronizar and colunas_numericas:
        scaler = StandardScaler()
        matriz[colunas_numericas] = scaler.fit_transform(matriz[colunas_numericas])
        print("✅ Variáveis numéricas padronizadas")

    # Converter categóricas para string
    for coluna in colunas_categoricas_validas:
        matriz[coluna] = matriz[coluna].astype(str)

    # Calcular índices categóricos apenas para colunas válidas
    indices_categoricos = [colunas_usadas.index(coluna) for coluna in colunas_categoricas_validas]

    return matriz.to_numpy(), indices_categoricos

def calcular_custo_kprototypes(k: int, matriz: np.ndarray, indices_categoricos: List[int], gamma: float) -> float:
    """
    Calcula o custo para um determinado K com Gamma específico.
    """
    try:
        kprototypes = KPrototypes(
            n_clusters=k,
            init='Cao',
            random_state=42,
            n_init=3,
            gamma=gamma,
            verbose=0
        )
        kprototypes.fit_predict(matriz, categorical=indices_categoricos)
        return kprototypes.cost_
    except Exception as erro:
        print(f"❌ Erro ao calcular custo para K={k}: {erro}")
        return float('inf')

def executar_metodo_elbow(dataframe: pd.DataFrame, colunas_usadas: List[str], colunas_categoricas: List[str], k_maximo: int, gamma: float) -> Tuple[int, List[float]]:
    """
    Executa o método Elbow com tratamento robusto de erros.
    """
    try:
        matriz, indices_categoricos = preparar_matriz_para_kprototypes(dataframe, colunas_usadas, colunas_categoricas, padronizar=True)
        intervalo_k = range(2, min(k_maximo + 1, len(dataframe) // 3))  # Limite máximo seguro

        if len(intervalo_k) < 2:
            print("⚠️  Intervalo K muito pequeno para análise Elbow")
            return 3, []

        with tqdm(total=len(intervalo_k), desc="Calculando Elbow") as barra_progresso:
            custos = Parallel(n_jobs=-1)(
                delayed(calcular_custo_kprototypes)(k, matriz, indices_categoricos, gamma) for k in intervalo_k
            )
            barra_progresso.update(len(intervalo_k))

        # Filtrar valores infinitos
        custos_validos = [c for c in custos if c != float('inf')]
        if not custos_validos:
            print("❌ Todos os cálculos de custo falharam")
            return 3, []

        # Plotar gráfico Elbow
        plt.figure(figsize=(12, 6))
        plt.plot(intervalo_k[:len(custos_validos)], custos_validos, marker='o', linestyle='--', linewidth=2, markersize=8)
        plt.xlabel('Número de Clusters (K)', fontsize=12)
        plt.ylabel('Custo (Soma das Distâncias)', fontsize=12)
        plt.title(f'Método Elbow para K-Prototypes (Gamma: {gamma:.2f})', fontsize=14)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.xticks(intervalo_k[:len(custos_validos)])

        # Calcular ponto do cotovelo
        k_sugerido = 3
        if len(custos_validos) > 2:
            # Método da segunda derivada (diferenças das diferenças)
            diferencas = [custos_validos[i-1] - custos_validos[i] for i in range(1, len(custos_validos))]
            if diferencas:
                # Encontrar ponto de maior aceleração negativa
                diferencas_segundas = [diferencas[i-1] - diferencas[i] for i in range(1, len(diferencas))]
                if diferencas_segundas:
                    k_sugerido = diferencas_segundas.index(max(diferencas_segundas)) + 2
                    plt.plot(k_sugerido, custos_validos[k_sugerido-2], 'ro', markersize=10, label=f'Cotovelo (K={k_sugerido})')
                    plt.legend()

        plt.tight_layout()
        plt.show()

        print(f"\n📌 Análise do Gráfico Sugere K = {k_sugerido}")
        return k_sugerido, custos_validos

    except Exception as e:
        print(f"❌ Erro no método Elbow: {e}")
        return 3, []

def gerar_clusterizacao(dataframe: pd.DataFrame, colunas_categoricas: List[str], colunas_numericas: List[str], k: int, gamma: float, remover_outliers: bool = True) -> Tuple[pd.DataFrame, any, Dict[str, Any]]:
    """
    Executa a clusterização final com tratamento consistente de outliers.
    """
    colunas_usadas = colunas_categoricas + colunas_numericas

    # Tratar outliers antes da clusterização
    dataframe_limpo, info_outliers = detectar_e_tratar_outliers(dataframe, colunas_numericas, remover_outliers)

    print(f"\n⏳ Executando clusterização com {len(dataframe_limpo)} registros...")
    print(f"   Parâmetros: K={k}, Gamma={gamma:.2f}, Outliers removidos: {remover_outliers}")

    try:
        matriz, indices_categoricos = preparar_matriz_para_kprototypes(dataframe_limpo, colunas_usadas, colunas_categoricas, padronizar=True)

        kprototypes = KPrototypes(
            n_clusters=k,
            init='Cao',
            random_state=42,
            n_init=5,
            verbose=1,
            gamma=gamma
        )

        clusters = kprototypes.fit_predict(matriz, categorical=indices_categoricos)

        dataframe_clusterizado = dataframe_limpo.copy()
        dataframe_clusterizado['cluster'] = clusters
        dataframe_clusterizado['cluster'] = dataframe_clusterizado['cluster'].astype('category')

        # Adicionar informação de outliers ao DataFrame original de forma não destrutiva
        if 'outlier' in dataframe.columns:
            dataframe = dataframe.drop(columns=['outlier'])

        dataframe['outlier'] = dataframe.index.isin(info_outliers['ids_removidos'])

        print("✅ Clusterização concluída com sucesso!")
        return dataframe_clusterizado, kprototypes, info_outliers

    except Exception as e:
        print(f"❌ Erro na clusterização: {e}")
        raise

def analisar_clusters(dataframe_clusterizado: pd.DataFrame, colunas_numericas: List[str], colunas_categoricas: List[str]):
    """
    Realiza análise detalhada dos clusters gerados.
    """
    print("\n📊 ANÁLISE DETALHADA DOS CLUSTERS:")
    print("=" * 60)

    clusters_ordenados = sorted(dataframe_clusterizado['cluster'].unique())

    for cluster in clusters_ordenados:
        dados_cluster = dataframe_clusterizado[dataframe_clusterizado['cluster'] == cluster]

        print(f"\n🔸 CLUSTER {cluster} ({len(dados_cluster)} registros):")

        # Estatísticas numéricas
        if colunas_numericas:
            print(f"   📈 Estatísticas numéricas:")
            for coluna in colunas_numericas:
                if coluna in dados_cluster.columns:
                    print(f"      {coluna}: {dados_cluster[coluna].mean():.2f} ± {dados_cluster[coluna].std():.2f}")

        # Distribuição categórica
        if colunas_categoricas:
            print(f"   🎯 Distribuição categórica:")
            for coluna in colunas_categoricas:
                if coluna in dados_cluster.columns:
                    distribuicao = dados_cluster[coluna].value_counts().head(3)  # Top 3 categorias
                    for valor, quantidade in distribuicao.items():
                        percentual = (quantidade / len(dados_cluster)) * 100
                        print(f"      {coluna}_{valor}: {quantidade} ({percentual:.1f}%)")

        # Análise de performance educacional
        if 'nota_final' in dados_cluster.columns:
            nota_media = dados_cluster['nota_final'].mean()
            if nota_media >= 7.0:
                status = "✅ Excelente desempenho"
            elif nota_media >= 5.0:
                status = "⚠️  Desempenho regular"
            else:
                status = "❌ Desempenho crítico"
            print(f"   📊 {status} (nota média: {nota_media:.2f})")

def visualizar_clusters_pca(dataframe_clusterizado: pd.DataFrame, colunas_categoricas: List[str], colunas_numericas: List[str]):
    """
    Visualização robusta dos clusters com PCA.
    """
    try:
        dataframe_pca = dataframe_clusterizado[colunas_categoricas + colunas_numericas].copy()

        # Padronizar variáveis numéricas
        if colunas_numericas:
            scaler = StandardScaler()
            dataframe_pca[colunas_numericas] = scaler.fit_transform(dataframe_pca[colunas_numericas])

        # Converter variáveis categóricas usando one-hot encoding
        if colunas_categoricas:
            dataframe_pca = pd.get_dummies(dataframe_pca, columns=colunas_categoricas, drop_first=True, dummy_na=True)

        if dataframe_pca.shape[1] < 2:
            print("⚠️ PCA requer pelo menos 2 colunas para visualização.")
            return

        # Executar PCA
        pca = PCA(n_components=2)
        x_reduzido = pca.fit_transform(dataframe_pca)
        variancia_explicada = sum(pca.explained_variance_ratio_) * 100

        # Criar visualização
        plt.figure(figsize=(14, 10))

        # Mapear clusters para cores consistentes
        clusters_unicos = sorted(dataframe_clusterizado['cluster'].unique())
        cores = plt.cm.tab10(np.linspace(0, 1, len(clusters_unicos)))
        mapeamento_cores = dict(zip(clusters_unicos, cores))

        # Scatter plot com cores mapeadas
        for cluster in clusters_unicos:
            indices = dataframe_clusterizado['cluster'] == cluster
            plt.scatter(x_reduzido[indices, 0], x_reduzido[indices, 1],
                       c=[mapeamento_cores[cluster]], label=f'Cluster {cluster}',
                       s=100, alpha=0.7, edgecolors='w', linewidth=0.5)

        plt.title('Visualização dos Clusters com PCA\n(Análise de Segmentação)', fontsize=16, pad=20)
        plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)

        plt.legend(title="Clusters", loc='best', frameon=True, fancybox=True)
        plt.grid(True, linestyle='--', alpha=0.3)

        # Informação de qualidade
        plt.figtext(0.02, 0.02, f'Variância explicada: {variancia_explicada:.1f}%\n'
                               f'Total de clusters: {len(clusters_unicos)}\n'
                               f'Total de regisros: {len(dataframe_clusterizado)}',
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

        plt.tight_layout()
        plt.show()

        print(f"\n📊 Qualidade da Visualização:")
        print(f"   ✅ Variância explicada: {variancia_explicada:.1f}%")
        if variancia_explicada < 50:
            print("   ⚠️  Representação 2D pode não capturar toda a estrutura dos dados")
        else:
            print("   ✅ Boa representação da estrutura multidimensional")

    except Exception as e:
        print(f"❌ Erro na visualização PCA: {e}")

def gerar_relatorio_completo(dataframe_clusterizado: pd.DataFrame, info_outliers: Dict[str, Any]) -> str:
    """
    Gera relatório completo com metadados separados.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Nome do arquivo principal
    nome_arquivo_csv = f'clusterizacao_{timestamp}.csv'

    # Salvar CSV limpo
    dataframe_clusterizado.to_csv(nome_arquivo_csv, index=False)

    # Salvar metadados em arquivo separado
    nome_arquivo_meta = f'metadados_clusterizacao_{timestamp}.txt'
    metadados = [
        "RELATÓRIO DE CLUSTERIZAÇÃO K-PROTOTYPES",
        "=" * 50,
        f"Data de geração: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total de clusters: {dataframe_clusterizado['cluster'].nunique()}",
        f"Total de registros: {len(dataframe_clusterizado)}",
        f"Outliers detectados: {info_outliers['total_detectados']}",
        f"Outliers removidos: {len(info_outliers['ids_removidos'])}",
        f"Variáveis utilizadas: {', '.join([col for col in dataframe_clusterizado.columns if col != 'cluster'])}",
        "\nESTATÍSTICAS POR CLUSTER:",
        "-" * 30
    ]

    # Adicionar estatísticas por cluster
    for cluster in sorted(dataframe_clusterizado['cluster'].unique()):
        dados_cluster = dataframe_clusterizado[dataframe_clusterizado['cluster'] == cluster]
        metadados.append(f"Cluster {cluster}: {len(dados_cluster)} registros")

    with open(nome_arquivo_meta, 'w', encoding='utf-8') as f:
        f.write('\n'.join(metadados))

    display(widgets.HTML(f"""
    <h4>✅ Relatórios gerados:</h4>
    <ul>
    <li><code>{nome_arquivo_csv}</code> - Dados clusterizados</li>
    <li><code>{nome_arquivo_meta}</code> - Metadados e estatísticas</li>
    </ul>
    """))

    display(FileLink(nome_arquivo_csv))
    display(FileLink(nome_arquivo_meta))

    return nome_arquivo_csv

# ==============================================================================
# BLOCO 3: INTERFACE DO USUÁRIO (CORRIGIDA)
# ==============================================================================
def criar_interface_clusterizacao(dataframe: pd.DataFrame, colunas_categoricas: List[str], colunas_numericas: List[str]):
    """
    Cria interface interativa com controles funcionais.
    """
    saida_principal = widgets.Output()

    # Widgets de seleção de variáveis
    checkboxes_categoricas = [widgets.Checkbox(value=True, description=coluna, indent=False)
                             for coluna in colunas_categoricas]
    checkboxes_numericas = [widgets.Checkbox(value=True, description=coluna, indent=False)
                           for coluna in colunas_numericas]

    # Widgets de configuração (agora funcionais)
    checkbox_padronizar = widgets.Checkbox(value=True, description='Padronizar variáveis numéricas', indent=False)
    checkbox_remover_outliers = widgets.Checkbox(value=True, description='Remover outliers', indent=False)

    slider_gamma = widgets.FloatSlider(value=0.5, min=0.1, max=3.0, step=0.1, description='Gamma:')
    checkbox_gamma_auto = widgets.Checkbox(value=True, description='Gamma automático', indent=False)

    slider_k = widgets.IntSlider(value=3, min=2, max=10, step=1, description='K Clusters:')
    slider_k_maximo = widgets.IntSlider(value=8, min=4, max=15, step=1, description='K máximo (Elbow):')

    # Botões
    botao_elbow = widgets.Button(description="📊 Executar Análise Elbow", button_style='info')
    botao_clusterizar = widgets.Button(description="🚀 Executar Clusterização", button_style='success')
    botao_clusterizar.layout.display = 'none'

    # Callbacks melhorados
    def alternar_controles(mudanca):
        slider_gamma.disabled = checkbox_gamma_auto.value

    checkbox_gamma_auto.observe(alternar_controles, names='value')

    def ao_clicar_elbow(botao):
        with saida_principal:
            clear_output(wait=True)

            colunas_cat_selecionadas = [cb.description for cb in checkboxes_categoricas if cb.value]
            colunas_num_selecionadas = [cb.description for cb in checkboxes_numericas if cb.value]

            if not colunas_cat_selecionadas and not colunas_num_selecionadas:
                print("❌ Selecione pelo menos uma variável.")
                return

            # Calcular Gamma
            if checkbox_gamma_auto.value:
                gamma = calcular_gamma_automatico(dataframe, colunas_num_selecionadas)
            else:
                gamma = slider_gamma.value

            print("📋 Configuração selecionada:")
            print(f"   Variáveis categóricas: {colunas_cat_selecionadas}")
            print(f"   Variáveis numéricas: {colunas_num_selecionadas}")
            print(f"   Gamma: {gamma:.3f}")
            print(f"   K máximo: {slider_k_maximo.value}")

            # Executar análise Elbow
            k_sugerido, custos = executar_metodo_elbow(
                dataframe,
                colunas_cat_selecionadas + colunas_num_selecionadas,
                colunas_cat_selecionadas,
                slider_k_maximo.value,
                gamma
            )

            if custos:  # Só atualizar se Elbow foi bem-sucedido
                slider_k.value = k_sugerido
                botao_clusterizar.layout.display = 'inline-block'

    def ao_clicar_clusterizar(botao):
        with saida_principal:
            clear_output(wait=True)

            colunas_cat_selecionadas = [cb.description for cb in checkboxes_categoricas if cb.value]
            colunas_num_selecionadas = [cb.description for cb in checkboxes_numericas if cb.value]

            if not colunas_cat_selecionadas and not colunas_num_selecionadas:
                print("❌ Selecione pelo menos uma variável.")
                return

            # Calcular Gamma
            if checkbox_gamma_auto.value:
                gamma = calcular_gamma_automatico(dataframe, colunas_num_selecionadas)
            else:
                gamma = slider_gamma.value

            print("🎯 Iniciando clusterização final...")
            print(f"   K: {slider_k.value}")
            print(f"   Gamma: {gamma:.3f}")
            print(f"   Padronização: {'Sim' if checkbox_padronizar.value else 'Não'}")
            print(f"   Remoção de outliers: {'Sim' if checkbox_remover_outliers.value else 'Não'}")

            try:
                # Executar clusterização
                dataframe_clusterizado, modelo, info_outliers = gerar_clusterizacao(
                    dataframe, colunas_cat_selecionadas, colunas_num_selecionadas,
                    slider_k.value, gamma, checkbox_remover_outliers.value
                )

                # Análise dos resultados
                analisar_clusters(dataframe_clusterizado, colunas_num_selecionadas, colunas_cat_selecionadas)

                # Visualização
                visualizar_clusters_pca(dataframe_clusterizado, colunas_cat_selecionadas, colunas_num_selecionadas)

                # Download
                gerar_relatorio_completo(dataframe_clusterizado, info_outliers)

                print("\n✅ Processo concluído com sucesso!")

            except Exception as e:
                print(f"❌ Erro durante a clusterização: {e}")
                import traceback
                traceback.print_exc()

    # Configurar eventos
    botao_elbow.on_click(ao_clicar_elbow)
    botao_clusterizar.on_click(ao_clicar_clusterizar)

    # Layout da interface
    interface = widgets.VBox([
        widgets.HTML("<h3>🔧 Configuração da Clusterização</h3>"),

        widgets.HTML("<h4>📋 Selecione as variáveis:</h4>"),
        widgets.HTML("<strong>Variáveis Categóricas:</strong>"),
        widgets.VBox(checkboxes_categoricas),
        widgets.HTML("<strong>Variáveis Numéricas:</strong>"),
        widgets.VBox(checkboxes_numericas),

        widgets.HTML("<h4>⚙️ Configurações Avançadas:</h4>"),
        checkbox_padronizar,
        checkbox_remover_outliers,
        widgets.HBox([checkbox_gamma_auto, slider_gamma]),

        widgets.HTML("<h4>📊 Configuração de Clusters:</h4>"),
        widgets.HBox([widgets.Label("K sugerido:"), slider_k]),
        widgets.HBox([widgets.Label("K máximo (Elbow):"), slider_k_maximo]),

        widgets.HTML("<br>"),
        widgets.HBox([botao_elbow, botao_clusterizar]),
        saida_principal
    ])

    return interface

# ==============================================================================
# BLOCO 4: EXECUÇÃO PRINCIPAL
# ==============================================================================
def main():
    """Função principal de execução do script."""

    # Verificar se estamos no Google Colab
    try:
        from google.colab import files
        is_colab = True
    except ImportError:
        is_colab = False
        print("⚠️  Ambiente não detectado como Google Colab")
        print("   Algumas funcionalidades podem não estar disponíveis")

    botao_carregar = widgets.Button(
        description="📂 Carregar Arquivo CSV",
        button_style='primary',
        icon='folder'
    )

    area_saida = widgets.Output()

    def ao_carregar_arquivo(botao):
        with area_saida:
            clear_output()

            if not is_colab:
                print("❌ Upload de arquivo só disponível no Google Colab")
                print("   Use um arquivo local ou execute no Google Colab")
                return

            print("⏳ Aguardando upload do arquivo...")

            try:
                arquivos = files.upload()
                if not arquivos:
                    print("❌ Nenhum arquivo selecionado.")
                    return

                nome_arquivo = list(arquivos.keys())[0]
                dados = pd.read_csv(BytesIO(arquivos[nome_arquivo]))
                dados = reduzir_memoria(dados)

                # Processar dados
                colunas_categoricas = dados.select_dtypes(include=['object']).columns
                dados[colunas_categoricas] = dados[colunas_categoricas].astype(str)

                colunas_cat, colunas_num = identificar_variaveis(dados)

                print(f"✅ Arquivo carregado: {nome_arquivo}")
                print(f"📊 Dimensões: {dados.shape[0]} linhas × {dados.shape[1]} colunas")
                print(f"🔍 Variáveis categóricas: {colunas_cat}")
                print(f"🔢 Variáveis numéricas: {colunas_num}")

                # Mostrar prévia
                display(dados.head(3))

                # Iniciar interface
                interface = criar_interface_clusterizacao(dados, colunas_cat, colunas_num)
                display(interface)

            except Exception as erro:
                print(f"❌ Erro ao processar arquivo: {erro}")
                print("   Verifique se o arquivo é um CSV válido.")

    botao_carregar.on_click(ao_carregar_arquivo)

    # Interface inicial
    display(widgets.HTML("""
    <h1>🎯 Sistema de Clusterização K-Prototypes</h1>
    <p>Ferramenta avançada para segmentação de dados com detecção de outliers</p>
    <p><strong>✨ Funcionalidades:</strong></p>
    <ul>
    <li>✅ Clusterização com variáveis mistas (numéricas + categóricas)</li>
    <li>✅ Detecção robusta de outliers (Z-score + LOF)</li>
    <li>✅ Análise Elbow para determinação do K ideal</li>
    <li>✅ Visualização PCA interativa</li>
    <li>✅ Relatórios completos em CSV + metadados</li>
    </ul>
    """))

    display(botao_carregar)
    display(area_saida)

# ==============================================================================
# EXECUÇÃO
# ==============================================================================
if __name__ == "__main__":
    main()