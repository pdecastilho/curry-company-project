# importa as bibliotecas
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from haversine import haversine
from datetime import datetime
import streamlit as st
from PIL import Image

# seleciona o arquivo do dataset
df_raw = pd.read_csv('./dataset/food_delivery_india_dataset_train.csv')

# faz uma cópia do dataframe de backup
df = df_raw.copy()

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

# ==============================
# Sidebar
# ==============================

st.header('Marketplace - Visão Restaurantes')

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
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            num_deliver = len(df.loc[:, 'Delivery_person_ID'].unique())
            col1.metric('Nº de entregadores', num_deliver)
        with col2:
            avg_distance = df['distance(km)'] = ( df.loc[:, ['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude']]
                                                    .apply(lambda x: haversine(
                                                         (x['Restaurant_latitude'], x['Restaurant_longitude']),
                                                         (x['Delivery_location_latitude'], x['Delivery_location_longitude'])
                                                     ), axis=1)
                                                )
            col2.metric('Distância média das entregas (km)', np.round(float(df.loc[:, 'distance(km)'].mean()), 1))
        with col3:
            df_aux = ( df.loc[:, ['Time_taken(min)', 'Festival']]
                         .groupby('Festival')
                         .agg({ 'Time_taken(min)': ['mean', 'std'] }) )
            df_aux.columns = ['Avg_time', 'Std_time']
            df_aux = df_aux.reset_index()
            df_aux = np.round(df_aux.loc[df_aux['Festival'] == 'Yes', 'Avg_time'], 1)
            col3.metric('Tempo médio de entrega no Festival (min)', df_aux)
        with col4:
            df_aux = ( df.loc[:, ['Time_taken(min)', 'Festival']]
                         .groupby('Festival')
                         .agg({ 'Time_taken(min)': ['mean', 'std'] }) )
            df_aux.columns = ['Avg_time', 'Std_time']
            df_aux = df_aux.reset_index()
            df_aux = np.round(df_aux.loc[df_aux['Festival'] == 'Yes', 'Std_time'], 1)
            col4.metric('Desvio padrão médio de entrega no Festival (min)', df_aux)
        with col5:
            df_aux = ( df.loc[:, ['Time_taken(min)', 'Festival']]
                         .groupby('Festival')
                         .agg({ 'Time_taken(min)': ['mean', 'std'] }) )
            df_aux.columns = ['Avg_time', 'Std_time']
            df_aux = df_aux.reset_index()
            df_aux = np.round(df_aux.loc[df_aux['Festival'] == 'No', 'Avg_time'], 1)
            col5.metric('Tempo médio de entrega fora do Festival (min)', df_aux)
        with col6:
            df_aux = ( df.loc[:, ['Time_taken(min)', 'Festival']]
                         .groupby('Festival')
                         .agg({ 'Time_taken(min)': ['mean', 'std'] }) )
            df_aux.columns = ['Avg_time', 'Std_time']
            df_aux = df_aux.reset_index()
            df_aux = np.round(df_aux.loc[df_aux['Festival'] == 'No', 'Std_time'], 1)
            col6.metric('Desvio padrão médio de entrega fora do Festival (min)', df_aux)
    
    with st.container():
        st.title('Tempo médio de entrega')
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('#### Por cidade')
            df['distance'] = ( df.loc[:, ['Delivery_location_latitude', 'Delivery_location_longitude', 'Restaurant_latitude', 'Restaurant_longitude']]
                                .apply(lambda x: haversine(
                                    (x['Restaurant_latitude'], x['Restaurant_longitude']),
                                    (x['Delivery_location_latitude'], x['Delivery_location_longitude'])
                                ), axis=1))
            avg_distance = ( df.loc[:, ['City', 'distance']]
                            .groupby('City')
                            .mean()
                            .reset_index()
            )
            fig = go.Figure(data=go.Pie(labels=avg_distance['City'], values=avg_distance['distance'], pull=[0, 0.1, 0]))
            st.plotly_chart(fig)
        with col2:
            st.markdown('#### Distribuição da distância')
            df_aux = df.loc[:, ['Time_taken(min)', 'City', 'Type_of_order']].groupby(['City', 'Type_of_order']).agg({ 'Time_taken(min)': ['mean', 'std'] })
            df_aux.columns = ['Avg_time', 'Std_time']
            df_aux = df_aux.reset_index()
            st.dataframe(df_aux)
    
    with st.container():
        st.title('Distribuição do tempo')
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('##### Tempo médio por cidade')
            df_aux = ( df.loc[:, ['Time_taken(min)', 'City']]
                         .groupby('City')
                         .agg({ 'Time_taken(min)': ['mean', 'std'] })
            )
            df_aux.columns = ['Avg_time', 'Std_time']
            df_aux = df_aux.reset_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Control', x=df_aux['City'], y=df_aux['Avg_time'], error_y=dict(type='data', array=df_aux['Std_time'])))
            fig.update_layout(barmode='group')
            st.plotly_chart(fig)
        with col2:
            st.markdown('##### Tempo médio por tipo de tráfego')
            df_aux = ( df.loc[:, ['Time_taken(min)', 'City', 'Road_traffic_density']]
                         .groupby(['City', 'Road_traffic_density'])
                         .agg({ 'Time_taken(min)': ['mean', 'std'] })
            )
            df_aux.columns = ['Avg_time', 'Std_time']
            df_aux = df_aux.reset_index()
            fig = px.sunburst(df_aux, path=['City', 'Road_traffic_density'], values='Avg_time', color='Std_time', color_continuous_scale='RdBu', color_continuous_midpoint=np.average(df_aux['Std_time']))
            st.plotly_chart(fig)
