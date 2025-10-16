import streamlit as st
import pandas as pd

@st.cache_data
def load_data(file_pokemon_excel, file_combat_excel):
    """
    Carrega os dados dos arquivos Excel (.xlsx) usando pd.read_excel().
    """

    def safe_read_excel(filepath, filename_ref):
        try:
            # Tenta ler o arquivo Excel
            # A planilha 'Sheet1' é o padrão se não for especificado
            df = pd.read_excel(filepath)
            st.success(f"Arquivo '{filename_ref}' carregado com sucesso.")

            # Renomear colunas do DataFrame Pokemon por segurança (cabeçalho da primeira linha)
            if filename_ref == "pokemon.xlsx":
                df.columns = ['id', 'name', 'hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed', 'generation',
                              'legendary', 'types']

            return df

        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}. Verifique o caminho.")
        except ImportError:
            raise ImportError(
                "A biblioteca 'openpyxl' é necessária para ler arquivos .xlsx. Por favor, execute: pip install openpyxl")
        except Exception as e:
            st.error(f"Erro ao tentar ler o arquivo Excel '{filename_ref}'.")
            raise Exception(f"Erro de leitura de Excel: {e}")

    df_pokemon = safe_read_excel(file_pokemon_excel, "pokemon.xlsx")
    df_combat = safe_read_excel(file_combat_excel, "combat_pokemon.xlsx")

    return df_pokemon, df_combat