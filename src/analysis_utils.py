import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import numpy as np

@st.cache_data
def prepare_data(df_pokemon, df_combat):
    """
    Combina os DataFrames e cria features de diferença de atributos.
    A feature 'Vantagem Lendária' foi removida.
    """
    # Removendo 'legendary' dos atributos que serão usados para calcular a diferença
    attributes_and_status = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed', 'legendary']
    attributes = attributes_and_status[:-1]  # Atributos numéricos

    # 1. Preparar dados do Primeiro Pokémon (P1)
    p1_data = df_combat.merge(df_pokemon[['id'] + attributes_and_status],
                              left_on='first_pokemon',
                              right_on='id',
                              suffixes=('', '_p1'))

    # 2. Preparar dados do Segundo Pokémon (P2)
    final_df = p1_data.merge(df_pokemon[['id'] + attributes_and_status],
                             left_on='second_pokemon',
                             right_on='id',
                             suffixes=('_p1', '_p2'))

    # --- Passo de Conversão ---
    # Conversão de bool/string para 0/1, usada no passado, mas mantida por segurança caso seja necessária:
    bool_map = {True: 1, False: 0, 'True': 1, 'False': 0, 'true': 1, 'false': 0}
    final_df['legendary_p1'] = final_df['legendary_p1'].astype(str).map(bool_map).fillna(0)
    final_df['legendary_p2'] = final_df['legendary_p2'].astype(str).map(bool_map).fillna(0)

    # --- Criação das Features de Diferença ---
    feature_cols = []
    for attr in attributes:
        diff_col = f'diff_{attr}'
        final_df[diff_col] = final_df[f'{attr}_p1'] - final_df[f'{attr}_p2']
        feature_cols.append(diff_col)

    # --- REMOÇÃO DA VANTAGEM LENDÁRIA ---
    # A feature 'legendary_advantage' não será criada/adicionada a feature_cols.

    # 4. Criar a variável alvo: p1_won (1 se p1 ganhou, 0 se p2 ganhou)
    final_df['p1_won'] = np.where(final_df['first_pokemon'] == final_df['winner'], 1, 0)

    return final_df, feature_cols


@st.cache_data
def train_model(df_model, feature_cols):
    """
    Treina um modelo RandomForest e calcula a importância dos atributos/status,
    expressando o resultado em porcentagem.
    """
    # Verifica se há features para treinar.
    if not feature_cols:
        return pd.DataFrame({'Atributo': [], 'Importância': []})

    X = df_model[feature_cols]
    y = df_model['p1_won']

    # Padronizar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Modelo: Random Forest para Feature Importance
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    # Obter a importância dos atributos
    feature_importances = pd.Series(model.feature_importances_, index=feature_cols)
    feature_importances = feature_importances.sort_values(ascending=False)

    # Preparar para o gráfico
    importance_df = pd.DataFrame({
        'Atributo': feature_importances.index,
        'Importância': feature_importances.values
    })

    # Multiplica por 100 para transformar em porcentagem
    importance_df['Importância'] = importance_df['Importância'] * 100

    # Limpar nomes dos atributos para exibição
    importance_df['Atributo'] = (
        importance_df['Atributo']
        .str.replace('diff_', '')
        .str.replace('hp', 'HP')
        .str.replace('sp_', 'Sp. ')
        # REMOVENDO A LIMPEZA DE 'Vantagem Lendária'
        .str.capitalize()
    )

    return importance_df


@st.cache_data
def analyze_top_winners(df_pokemon, df_combat):
    """
    Calcula o total de vitórias e a taxa de vitória para cada Pokémon individual.
    """

    # Dicionário para armazenar o total de vitórias e combates por Pokémon ID
    pokemon_stats = {}

    for _, row in df_combat.iterrows():
        p1_id = row['first_pokemon']
        p2_id = row['second_pokemon']
        winner_id = row['winner']

        # Função auxiliar para registrar a batalha
        def update_stats(pokemon_id, is_winner):
            if pokemon_id not in pokemon_stats:
                pokemon_stats[pokemon_id] = {'wins': 0, 'total': 0}

            pokemon_stats[pokemon_id]['total'] += 1
            if is_winner:
                pokemon_stats[pokemon_id]['wins'] += 1

        # Atualiza estatísticas de P1
        is_p1_winner = (winner_id == p1_id)
        update_stats(p1_id, is_p1_winner)

        # Atualiza estatísticas de P2
        update_stats(p2_id, not is_p1_winner)

    # Converte o dicionário para DataFrame
    df_stats = pd.DataFrame.from_dict(pokemon_stats, orient='index').reset_index()
    df_stats.columns = ['id', 'Total de Vitórias', 'Total de Combates']

    # Calcula a taxa de vitória
    df_stats['Taxa de Vitória (%)'] = (df_stats['Total de Vitórias'] / df_stats['Total de Combates']) * 100

    # Adiciona o nome do Pokémon para exibição
    # Usamos o df_pokemon que contém os nomes
    df_stats = df_stats.merge(df_pokemon[['id', 'name']], on='id', how='left')

    # Ordena pelo número de vitórias (o que o usuário pediu) e pega o Top 10
    # Opcionalmente, pode-se ordenar pela Taxa de Vitória, mas o Total é mais direto.
    df_top_winners = df_stats.sort_values(by='Total de Vitórias', ascending=False)

    return df_top_winners


@st.cache_data
def analyze_average_attributes(df_pokemon):
    """
    Calcula os atributos médios (HP, Attack, Defense, etc.)
    para Pokémon Lendários e Não Lendários.
    """

    # Lista dos atributos a serem analisados
    attributes = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']

    # 1. Agrupar por status lendário e calcular a média
    # É necessário primeiro garantir que a coluna 'legendary' seja numérica (0/1) ou booleana.
    # Como já corrigimos para lidar com strings ('true'/'false') em 'analysis_utils.py',
    # faremos uma conversão segura aqui também, assumindo que a coluna original possa ser string.

    # Mapeamento seguro para booleano
    legendary_map = {True: 'Lendário', False: 'Não Lendário', 'True': 'Lendário', 'False': 'Não Lendário',
                     'true': 'Lendário', 'false': 'Não Lendário'}

    df_analysis = df_pokemon.copy()

    # Cria uma coluna de status legível
    df_analysis['Status'] = df_analysis['legendary'].astype(str).str.lower().map(legendary_map)

    # Agrupa e calcula a média para os atributos
    df_avg_stats = df_analysis.groupby('Status')[attributes].mean().reset_index()

    # 2. Transpor os dados para o formato longo (ideal para Plotly)
    # Transforma as colunas de atributos em uma única coluna 'Atributo'
    df_melted = df_avg_stats.melt(
        id_vars='Status',
        value_vars=attributes,
        var_name='Atributo',
        value_name='Média do Atributo'
    )

    # Formatação dos nomes dos atributos
    df_melted['Atributo'] = (
        df_melted['Atributo']
        .str.replace('hp', 'HP')
        .str.replace('attack', 'Attack')
        .str.replace('defense', 'Defense')
        .str.replace('sp_attack', 'Sp. Attack')
        .str.replace('sp_defense', 'Sp. Defense')
        .str.replace('speed', 'Speed')
    )

    return df_melted

@st.cache_data
def analyze_win_distribution(df_combat):
    """
    Calcula o número total de vitórias para cada Pokémon e retorna
    o DataFrame pronto para análise de distribuição.
    """

    # 1. Contar as vitórias para cada Pokémon ID
    # A coluna 'winner' do df_combat já tem o ID do Pokémon vencedor em cada batalha.
    # Contar as ocorrências é o mesmo que contar as vitórias.

    df_wins = df_combat['winner'].value_counts().reset_index()
    df_wins.columns = ['id', 'Total de Vitórias']

    # NOTA: Esta função considera apenas as vitórias, pois para a distribuição (histograma),
    # o total de vitórias é a variável que queremos medir a frequência.

    return df_wins
