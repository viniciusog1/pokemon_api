import streamlit as st
import pandas as pd

@st.cache_data
def analyze_type_win_rate(df_pokemon, df_combat):
    """
    Calcula a taxa de vitória de cada tipo de Pokémon em todos os combates.
    """

    # 1. Desmembrar Tipos (Primary e Secondary)
    type_splits = df_pokemon[['id', 'types']].copy()
    type_splits['type1'] = type_splits['types'].apply(lambda x: x.split('/')[0].strip())
    type_splits['type2'] = type_splits['types'].apply(
        lambda x: x.split('/')[1].strip() if len(x.split('/')) > 1 else None)

    df_types_melted = type_splits.melt(
        id_vars=['id'],
        value_vars=['type1', 'type2'],
        value_name='Type'
    ).dropna(subset=['Type'])
    df_types_melted = df_types_melted[['id', 'Type']].drop_duplicates()

    # 2. Contar Vitórias e Derrotas
    type_stats = {}

    for _, row in df_combat.iterrows():
        p1_id = row['first_pokemon']
        p2_id = row['second_pokemon']
        winner_id = row['winner']

        p1_types = df_types_melted[df_types_melted['id'] == p1_id]['Type'].tolist()
        p2_types = df_types_melted[df_types_melted['id'] == p2_id]['Type'].tolist()

        # Função auxiliar para atualizar as estatísticas
        def update_stats(type_list, is_winner):
            for p_type in type_list:
                if p_type not in type_stats:
                    type_stats[p_type] = {'wins': 0, 'total': 0}

                type_stats[p_type]['total'] += 1
                if is_winner:
                    type_stats[p_type]['wins'] += 1

        # Atualizar estatísticas
        is_p1_winner = (winner_id == p1_id)
        update_stats(p1_types, is_p1_winner)
        update_stats(p2_types, not is_p1_winner)

    # 3. Calcular a Taxa de Vitória
    type_win_rates = []
    for p_type, stats in type_stats.items():
        win_rate = (stats['wins'] / stats['total']) * 100 if stats['total'] > 0 else 0
        type_win_rates.append({
            'Tipo': p_type,
            'Taxa de Vitória (%)': win_rate,
            'Total de Combates': stats['total'],
            'Total de Vitórias': stats['wins']
        })

    df_win_rate = pd.DataFrame(type_win_rates)
    df_win_rate = df_win_rate.sort_values(by='Taxa de Vitória (%)', ascending=False)

    return df_win_rate
