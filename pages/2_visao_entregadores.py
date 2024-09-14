# importa as bibliotecas
import pandas as pd
from datetime import datetime
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title='Visão Entregadores',
    page_icon='🍲',
    layout='wide'
)


def clean_dataframe(df):
    """
    Esta função tem a responsabilidade
    de limpar os dados do dataframe

    Tipos de limpeza:
    1. Remoção dos registros com NaN
    2. Remoção de espaços extra nas strings
    3. Alteração dos tipos de dados
    4. Formatação de datas
    5. Remoção de caracteres extra

    Input: Dataframe
    Output: Dataframe
    """
    # faz a limpeza dos dados
    df = df.loc[df['Delivery_person_Age'] != 'NaN ', :].copy()
    df = df.loc[df['Festival'] != 'NaN ', :].copy()
    df = df.loc[df['City'] != 'NaN ', :].copy()
    df = df.loc[df['Road_traffic_density'] != 'NaN ', :].copy()
    df = df.loc[df['multiple_deliveries'] != 'NaN ', :].copy()
    df['ID'] = df.loc[:, 'ID'].str.strip()
    df['Delivery_person_ID'] = df.loc[:, 'Delivery_person_ID'].str.strip()
    df['Road_traffic_density'] = df.loc[:, 'Road_traffic_density'].str.strip()
    df['Type_of_order'] = df.loc[:, 'Type_of_order'].str.strip()
    df['Type_of_vehicle'] = df.loc[:, 'Type_of_vehicle'].str.strip()
    df['Festival'] = df.loc[:, 'Festival'].str.strip()
    df['City'] = df.loc[:, 'City'].str.strip()
    df['Time_taken(min)'] = df['Time_taken(min)'].apply(lambda x: x.split('(min) ')[1])
    df['Weatherconditions'] = df['Weatherconditions'].apply(lambda x: x.split('conditions ')[1])
    df['Delivery_person_Age'] = df['Delivery_person_Age'].astype(int)
    df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].astype(float)
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y')
    df['multiple_deliveries'] = df['multiple_deliveries'].astype(int)
    df['Time_taken(min)'] = df['Time_taken(min)'].astype(int)
    df = df.reset_index(drop=True)
    return df


def avg_ratings(df):
    return ( df.loc[:, ['Delivery_person_ID', 'Delivery_person_Ratings']]
               .groupby('Delivery_person_ID')
               .mean()
               .reset_index() )


def avg_ratings_by(df, groupby):
    df_avg_std_ratings_by = ( df.loc[:, ['Delivery_person_Ratings', groupby]]
                                .groupby(groupby)
                                .agg({ 'Delivery_person_Ratings': ['mean', 'std'] }) )
    df_avg_std_ratings_by.columns = ['delivery_ratings_mean', 'delivery_ratings_std']
    df_avg_std_ratings_by.reset_index()
    return df_avg_std_ratings_by


def top_delivers(df, ascending=True):
    df_top_10 = ( df.loc[:, ['Delivery_person_ID', 'City', 'Time_taken(min)']]
                    .groupby(['City', 'Delivery_person_ID'])
                    .mean()
                    .sort_values(['City', 'Time_taken(min)'], ascending=ascending)
                    .reset_index() )
    df_aux1 = df_top_10.loc[df_top_10['City'] == 'Metropolitian', :].head(10)
    df_aux3 = df_top_10.loc[df_top_10['City'] == 'Urban', :].head(10)
    df_aux2 = df_top_10.loc[df_top_10['City'] == 'Semi-Urban', :].head(10)
    df_top_10_by_city = pd.concat([df_aux1, df_aux2, df_aux3]).reset_index(drop=True)
    return df_top_10_by_city


# ==============================
# Início da estrutura lógica
# ==============================

# input dataset e limpeza dos dados
df_raw = pd.read_csv('./dataset/food_delivery_india_dataset_train.csv')
df = clean_dataframe(df_raw)

# ==============================
# Sidebar
# ==============================

st.header('Marketplace - Visão Entregadores')

image = Image.open('./assets/logo.png')
st.sidebar.image(image, width=120)

st.sidebar.markdown('# Curry Company')
st.sidebar.markdown('## Fastest Delivery in Town')

st.sidebar.markdown("""---""")
st.sidebar.markdown('## Selecione uma data limite')

date_slider = st.sidebar.slider(
    'Até qual data?',
    value=datetime(2022, 4, 13),
    min_value=datetime(2022, 2, 11),
    max_value=datetime(2022, 6, 4),
    format='DD-MM-YYYY'
)

st.sidebar.markdown("""---""")
st.sidebar.markdown('## Selecione a condição de trânsito')
traffic_options = st.sidebar.multiselect(
    'Quais as condições do trânsito?',
    ['Low', 'Medium', 'High', 'Jam'],
    default=['Low', 'Medium', 'High', 'Jam']
)

st.sidebar.markdown("""---""")
st.sidebar.markdown('## Selecione a condição de clima')
weather_options = st.sidebar.multiselect(
    'Quais as condições do clima?',
    ['Cloudy', 'Fog', 'Sandstorm', 'Stormy', 'Sunny', 'Windy'],
    default=['Cloudy', 'Fog', 'Sandstorm', 'Stormy', 'Sunny', 'Windy']
)

st.sidebar.markdown("""---""")
st.sidebar.markdown( '#### Powered by Comunidade DS')

# link filter
df = df.loc[df['Order_Date'] < date_slider, :]

df = df.loc[df['Road_traffic_density'].isin(traffic_options), :]

df = df.loc[df['Weatherconditions'].isin(weather_options), :]

# ==============================
# Layout
# ==============================

tab1, tab2, tab3 = st.tabs(['Visão Gerencial', '_', '_'])

with tab1:
    with st.container():
        st.title('Overall Metrics')
        col1, col2, col3, col4 = st.columns(4, gap='large')
        with col1:
            maior_idade = df.loc[:, 'Delivery_person_Age'].max()
            col1.metric('Maior de idade', maior_idade)
        with col2:
            menor_idade = df.loc[:, 'Delivery_person_Age'].min()
            col2.metric('Menor idade', menor_idade)
        with col3:
            melhor_condicao = df.loc[:, 'Vehicle_condition'].max()
            col3.metric('Melhor condição', melhor_condicao)
        with col4:
            pior_condicao = df.loc[:, 'Vehicle_condition'].min()
            col4.metric('Pior condição', pior_condicao)
    with st.container():
        st.markdown("""---""")
        st.title('Avaliações')
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('##### Médias por entregador')
            st.dataframe(avg_ratings(df))
        with col2:
            st.markdown('##### Médias por trânsito')
            df_avg_ratings = avg_ratings_by(df, groupby='Road_traffic_density')
            st.dataframe(df_avg_ratings)

            st.markdown('##### Médias por clima')
            df_avg_ratings = avg_ratings_by(df, groupby='Weatherconditions')
            st.dataframe(df_avg_ratings)
            
    with st.container():
        st.markdown("""---""")
        st.title('Velocidade de Entrega')
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('##### Top entregadores mais rápidos')
            df_top_delivers = top_delivers(df, ascending=True)
            st.dataframe(df_top_delivers)
        with col2:
            st.markdown('##### Top entregadores mais lentos')
            df_top_delivers = top_delivers(df, ascending=False)
            st.dataframe(df_top_delivers)
