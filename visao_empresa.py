# importa as bibliotecas
import pandas as pd
import plotly.express as px
import folium
from datetime import datetime
import streamlit as st
from PIL import Image
from streamlit_folium import folium_static

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

st.header('Marketplace - Visão Empresa')

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
st.sidebar.markdown( '#### Powered by Comunidade DS')

# link filter
df = df.loc[df['Order_Date'] < date_slider, :]

df = df.loc[df['Road_traffic_density'].isin(traffic_options), :]

# ==============================
# Layout
# ==============================

tab1, tab2, tab3 = st.tabs(['Visão Gerencial', 'Visão Tática', 'Visão Geográfica'])

with tab1:
    with st.container():
        # Order Metric
        st.header('Orders by Date')
        df_aux = df.loc[:, ['ID', 'Order_Date']].groupby('Order_Date').count().reset_index()
        st.plotly_chart(px.bar(df_aux, x='Order_Date', y='ID'), use_container_width=True)
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.header('Traffic Order Share')
            df_aux = df.loc[:, ['ID', 'Road_traffic_density']].groupby('Road_traffic_density').count().reset_index()
            df_aux['Traffic_density_percent'] = df_aux['ID'] / df_aux['ID'].sum()
            st.plotly_chart(px.pie(df_aux, values='Traffic_density_percent', names='Road_traffic_density'), use_container_width=True)
        with col2:
            st.header('Traffic Order City')
            df_aux = df.loc[:, ['ID', 'City', 'Road_traffic_density']].groupby(['City', 'Road_traffic_density']).count().reset_index()
            st.plotly_chart(px.scatter(df_aux, x='City', y='Road_traffic_density', size='ID', color='City'), use_container_width=True)

with tab2:
    with st.container():
        st.header('Orders by Week')
        df['Week_of_Year'] = df['Order_Date'].dt.strftime('%U')
        df_aux = df.loc[:, ['ID', 'Week_of_Year']].groupby('Week_of_Year').count().reset_index()
        st.plotly_chart(px.line(df_aux, x='Week_of_Year', y='ID'), use_container_width=True)

    with st.container():
        st.header('Order Share by Week')
        df_aux1 = df.loc[:, ['ID', 'Week_of_Year']].groupby('Week_of_Year').count().reset_index()
        df_aux2 = df.loc[:, ['Delivery_person_ID', 'Week_of_Year']].groupby('Week_of_Year').nunique().reset_index()
        df_aux = pd.merge(df_aux1, df_aux2, how='inner')
        df_aux['Order_by_deliver'] = df_aux['ID'] / df_aux['Delivery_person_ID']
        st.plotly_chart(px.line(df_aux, x='Week_of_Year', y='Order_by_deliver'), use_container_width=True)

with tab3:
    st.header('Country Map')
    df_aux = df.loc[:, ['City', 'Road_traffic_density', 'Delivery_location_latitude', 'Delivery_location_longitude']].groupby(['City', 'Road_traffic_density']).median().reset_index()
    map = folium.Map()
    for idx, loc_info in df_aux.iterrows():
        folium.Marker([
                loc_info['Delivery_location_latitude'],
                loc_info['Delivery_location_longitude']],
                popup=loc_info[['City', 'Road_traffic_density']]
            ).add_to(map)
    folium_static(map, width=700, height=500)
