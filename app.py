import streamlit as st
import plotly.express as px
import pandas as pd

# Importando as funções dos módulos
from src.data_loader import load_data
from src.analysis_utils import prepare_data, train_model, analyze_top_winners, analyze_average_attributes, \
    analyze_win_distribution
from src.analysis_types import analyze_type_win_rate

# --- Constantes de Arquivo ---
FILE_POKEMON_REF = "data/pokemon.xlsx"
FILE_COMBAT_REF = "data/combat_pokemon.xlsx"

# --- Configuração ---
st.set_page_config(layout="wide", page_title="Análise de Combate Pokémon")

st.title("📊 Análise de Combate Pokémon")

try:
    # 1. Carregamento dos Dados
    df_pokemon, df_combat = load_data(FILE_POKEMON_REF, FILE_COMBAT_REF)

    # 2. Análise de Atributos
    st.header("1. Importância de Atributos e Status Lendário na Vitória")

    # Prepara e treina o modelo
    df_final, feature_cols = prepare_data(df_pokemon, df_combat)
    importance_df = train_model(df_final, feature_cols)

    # Gráfico de Importância
    fig_attr = px.bar(
        importance_df,
        x='Atributo',
        y='Importância',
        color='Importância',
        color_continuous_scale=px.colors.sequential.Teal,
        title='Importância da Diferença de Atributos e Status Lendário (Random Forest)',
        labels={'Importância': 'Importância Relativa (%)', 'Atributo': 'Atributo Pokémon'}
        # =================================================
    )
    fig_attr.update_layout(xaxis={'categoryorder': 'total descending'}, height=500)
    st.plotly_chart(fig_attr, use_container_width=True)

    # Nova Interpretação
    st.markdown("### Interpretação da Importância")
    st.markdown("""
        A altura da barra indica o quanto o atributo ou status contribui para prever o vencedor da batalha.
        * **Velocidade (Speed)**: Proporciona uma vantagem absurda se comparada com os demais atributos.
        * **Atributos Ofensivos**: **Attack** e **Sp.attack** oferecem uma vantagem maior se comparados com os **atributos defensivos**, como **Defense** e **Sp.defense**. 
        * **Prioridade**: Para montar uma equipe vencedora é ideal priorizar por Pokémon's com altos níveis em velocidade e status ofensivos.  
        """)
    st.markdown("---")

    # 3. Análise de Tipos
    st.header("2. Taxa de Vitória por Tipo de Pokémon")
    st.markdown("""
    Esta análise calcula a porcentagem de vitórias para cada Tipo (considerando Tipos primário e secundário) em todos os combates.
    """)

    df_win_rate = analyze_type_win_rate(df_pokemon, df_combat)

    # Gráfico de Taxa de Vitória
    fig_type = px.bar(
        df_win_rate,
        x='Tipo',
        y='Taxa de Vitória (%)',
        color='Taxa de Vitória (%)',
        color_continuous_scale=px.colors.sequential.Viridis,
        title='Taxa de Vitória Média por Tipo de Pokémon',
        hover_data=['Total de Combates', 'Total de Vitórias']
    )
    fig_type.update_layout(xaxis={'categoryorder': 'total descending'},
                           height=550, yaxis_range=[30, df_win_rate['Taxa de Vitória (%)'].max() * 1.05])

    st.plotly_chart(fig_type, use_container_width=True)

    # Nova Interpretação
    st.markdown("### Interpretação da Importância")
    st.markdown("""
        A altura da barra indica o quanto o tipo do pokemon contribui para prever o vencedor da batalha.
        * Podemos perceber claramente os tipos de pokemon's mais vencedores, sendo o principal o tipo Flying (Voador).
        * Demais pokémon's dos tipos Dragon, Dark, Eletric são uma ótima escolha para montar uma equipe vencedora 
        """)

    with st.expander("Ver Tabela de Taxas de Vitória"):
        st.dataframe(df_win_rate)

    st.markdown("---")

    # --- Seção 3: Top N Vencedores ---
    st.header("3. 🥇 Top N Pokémon com Mais Vitórias (Barra Horizontal)")

    df_sorted_winners = analyze_top_winners(df_pokemon, df_combat)

    # Adiciona o Slider
    max_pokemons = len(df_sorted_winners)
    top_n = st.slider(
        'Selecione o número de Pokémon a serem exibidos (Top N):',
        min_value=5,
        max_value=min(25, max_pokemons),
        value=10,
        step=1
    )

    st.markdown(f"**Exibindo os Top {top_n} Pokémon** com o maior número de vitórias.")

    df_top_winners = df_sorted_winners.head(top_n)

    # CÓDIGO: GRÁFICO DE BARRA HORIZONTAL
    fig_winners = px.bar(
        df_top_winners,
        x='Total de Vitórias',  # X agora é o valor (vitórias)
        y='name',  # Y agora são os nomes dos Pokémon
        orientation='h',  # Define como horizontal
        color='Total de Vitórias',
        color_continuous_scale=px.colors.sequential.Sunset,
        title=f'Top {top_n} Pokémon pelo Número Absoluto de Vitórias',
        labels={'name': 'Pokémon', 'Total de Vitórias': 'Total de Vitórias'},
        hover_data=['Taxa de Vitória (%)', 'Total de Combates']
    )

    # Inverte o eixo Y para ter o Pokémon #1 no topo
    fig_winners.update_layout(yaxis={'categoryorder': 'total ascending'},
                              height=550, xaxis_range=[110, df_top_winners['Total de Vitórias'].max() * 1.01])

    st.plotly_chart(fig_winners, use_container_width=True)

    # Nova Interpretação
    st.markdown("### Interpretação da Importância")
    st.markdown("""
            O comprimento da barra indica quantas vitórias tem um determinado Pokémon em combate.
            * Podemos perceber claramente que o Pokémon Mewtwo é disparado um Pokémon com uma taxa de vitória altíssima comparado com os demais.
            * Demais pokémon's como Aerodactyl, Infernape, Jirachi também são Pokémon's com bastante vítorias em combate. 
            """)

    with st.expander(f"Ver Tabela dos Top {top_n} Vencedores"):
        st.dataframe(df_top_winners.set_index('id'))

    st.markdown("---")

    # --- Seção 4: Comparativo de Atributos Médios ---
    st.header("4. 🕸️ Comparativo: Perfil Médio de Atributos (Radar Chart)")
    st.markdown("""
    Este gráfico de teia (Radar Chart) mostra o *perfil* de atributos médios para Pokémon Lendários, Não Lendários e a Média Geral.
    O formato do polígono visualiza a superioridade relativa em cada atributo.
    """)

    df_avg_attr = analyze_average_attributes(df_pokemon)

    # Define a ordem desejada e as cores (reutilizamos as definições anteriores)
    status_order = ["Não Lendário", "Média Geral", "Lendário"]
    color_map = {
        "Não Lendário": "#3336DE",
        "Média Geral": "#33a02c",
        "Lendário": "#e31a1c"
    }

    # Ordena o DataFrame (IMPORTANTE para o Plotly)
    df_avg_attr['Status'] = pd.Categorical(df_avg_attr['Status'], categories=status_order, ordered=True)
    df_avg_attr = df_avg_attr.sort_values(['Atributo', 'Status'])

    # === GRÁFICO DE TEIA (RADAR CHART) ===
    fig_avg = px.line_polar(
        df_avg_attr,
        r='Média do Atributo',  # O raio (distância do centro)
        theta='Atributo',  # O ângulo (os atributos: HP, Attack, etc.)
        color='Status',  # As diferentes linhas/polígonos
        line_close=True,  # Fecha o polígono
        color_discrete_map=color_map,
        title='Perfil dos Atributos de Pokémon: Lendários, Não Lendários e Média Geral'
    )

    fig_avg.update_traces(fill='toself')  # Preenche a área do polígono
    fig_avg.update_layout(height=650)  # Altura maior é melhor para Radar Chart

    st.plotly_chart(fig_avg, use_container_width=True)

    st.subheader("Interpretação:")
    st.markdown(f"""
        O gráfico demonstra claramente que Pokémon Lendários possuem, em média, atributos significativamente superiores em todas as categorias (HP, Attack, Defense, etc.) quando comparados aos Não Lendários.
        * **Pokémon lendário**: Apresenta no geral atributos 30% superior do que um Pokémon não lendário. 
        """)

    st.markdown("---")

    # --- NOVA SEÇÃO 5: Distribuição de Vitórias ---
    st.header("5. 📈 Distribuição de Vitórias por Pokémon (Concentração de Poder)")
    st.markdown("""
        Este histograma mostra a frequência (o número de Pokémon) que se enquadra em diferentes faixas de vitórias. 
        Uma distribuição "alta" em vitórias baixas e "rabo longo" em vitórias altas indica que pouquíssimos Pokémon concentram a maioria dos sucessos.
        """)

    df_win_dist = analyze_win_distribution(df_combat)

    # Calcula o número de bins (faixas) ideal. 30 é um bom padrão.
    n_bins = 30

    fig_dist = px.histogram(
        df_win_dist,
        x='Total de Vitórias',
        nbins=n_bins,
        title='Distribuição da Contagem de Vitórias entre todos os Pokémon',
        labels={'Total de Vitórias': 'Número de Vitórias por Pokémon', 'count': 'Número de Pokémon nessa Faixa'},
        color_discrete_sequence=['#4c78a8']  # Cor simples
    )

    fig_dist.update_layout(bargap=0.05, height=550)  # Espaçamento entre as barras

    st.plotly_chart(fig_dist, use_container_width=True)

    st.subheader("Interpretação da Distribuição:")
    st.markdown("""
        * **Pico à Esquerda:** O pico mais alto (à esquerda) representa o grande número de Pokémon que possuem poucas vitórias, ou até nenhuma vitória (se o modelo permitisse).
        * **Cauda Longa (Long Tail):** A cauda da distribuição, estendendo-se para a direita, indica os Pokémon de elite (os Lendários e Top Atributos) que conseguem um número desproporcionalmente alto de vitórias.
        """)

    st.markdown("---")

    # --- Seção 6: Conclusão e Montagem da Equipe Ideal ---
    st.header("6. 🏆 Conclusão: Montagem da Equipe Ideal")
    st.markdown("""
        Uma equipe padrão em Pokémon é composta por 6 Pokémons's, 
        com base nas análises de importância de atributos, taxa de vitória por tipo e desempenho individual, propomos uma equipe de 6 Pokémon com a maior probabilidade de sucesso.
        """)

    st.subheader("🧩 Passo 1 – Fatores que Mais Influenciam a Vitória")
    data_fatores = {
        'Fator de Influência': [
            'Speed (Velocidade)',
            'Attack / Sp. Attack',
            'Ser Lendário',
            'Tipos Dominantes'
        ],
        'Descrição/Impacto': [
            '🥇 Mais determinante (cerca de 74% da importância total no modelo Random Forest)',
            '🥈 Alta influência ofensiva',
            '🏆 Aumenta as chances de vitória em +30%',
            'Flying, Dragon, Dark, Psychic, Fire, Steel'
        ]
    }
    df_fatores = pd.DataFrame(data_fatores).set_index('Fator de Influência')
    st.dataframe(df_fatores)

    st.markdown("""
        ➡️ **Conclusão:** A melhor equipe deve priorizar a **Velocidade** combinada com **Alto Ataque**, cobrindo os tipos **Flying / Dragon / Psychic / Dark / Fire / Steel**, de preferência com Pokémon Lendários ou Mega Evoluídos.
        """)
    st.markdown("""
        ➡️ **Observação:** Pokémons com altos atributos em caraterstícas **Defensivas** normalmente não possuem um bom desempenho quando comparado com Pokémon's com altos atributos **Ofensivos**, por isso devemos priorizar Pokémons ofensivos.
        """)

    st.markdown("---")

    st.subheader("🧠 Passo 2 – Critérios para Escolher a Equipe")
    st.markdown("""
        Para uma equipe equilibrada e dominante, a seleção seguiu as seguintes regras:
        * **2 Ofensivos Rápidos:** Para garantir a iniciativa e nocautear oponentes frágeis.
        * **2 Versáteis / Híbridos:** Com bom equilíbrio entre ataque, defesa e velocidade.
        * **2 Tanques Inteligentes:** Defensivos ou com grande "bulk" para absorver dano e cobrir fraquezas.
        * **Cobertura de Tipos:** Máxima variação de tipos para evitar vulnerabilidades repetidas.
        """)

    st.markdown("---")

    st.subheader("⚔️ Passo 3 – Equipe Ideal Sugerida")

    data_equipe = {
        'Função': ['🐉 Ofensivo 1', '🔥 Ofensivo 2', '🧠 Versátil 1', '⚙️ Versátil 2', '🛡️ Tanque 1', '🌑 Tanque 2'],
        'Pokémon': ['Mega Rayquaza', 'Infernape', 'Mewtwo', 'Jirachi', 'Mega Salamence', 'Mega Houndoom'],
        'Tipo(s)': ['Dragon / Flying', 'Fire / Fighting', 'Psychic', 'Steel / Psychic', 'Dragon / Flying',
                    'Dark / Fire'],
        'Motivo Principal': [
            'Altíssimo ataque e velocidade, ótimo contra quase tudo.',
            'Rápido e versátil, excelente contra Steel, Ice, e Normal.',
            'Domina em Sp. Attack e velocidade, alto winrate individual.',
            'Boa defesa e velocidade, cobertura contra Fairy e Ice.',
            'Excelente bulk e dano, cobre vulnerabilidade de Psychic.',
            'Contra-ataca Psychic e Grass, cobre fraqueza de Steel.'
        ]
    }
    df_equipe = pd.DataFrame(data_equipe)
    st.dataframe(df_equipe, hide_index=True)

    st.markdown("### 🧬 Resultado Esperado:")
    st.markdown("""
        * **Cobertura de Tipos:** Flying, Dragon, Fire, Psychic, Fighting, Steel, Dark.
        * **Taxa Estimada de Vitória da Equipe (Modelo Agregado):** **~86–88%**, com base na média ponderada das vitórias individuais e sinergias de tipo.
        """)
except FileNotFoundError as e:
    st.error(
        f"Erro: Arquivos Excel não encontrados. Verifique se '{FILE_POKEMON_REF}' e '{FILE_COMBAT_REF}' estão na mesma pasta do 'app.py' ou se o caminho está correto.")
except Exception as e:
    st.error(
        f"Ocorreu um erro na execução do script. Verifique se todas as bibliotecas estão instaladas (`pandas`, `openpyxl`, `scikit-learn`, `plotly`).")
    st.exception(e)