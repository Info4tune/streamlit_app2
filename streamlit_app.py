from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

#loading model
model = load_model('salario_ds_best')
st.set_page_config(page_title="Salário DS", page_icon=":bar_chart:")
st.title('Descubra o seu salário como Cientista de Dados :dollar:')

st.subheader("Adicione os seus dados abaixo para utilizar o modelo de previsão de salário desenvolvido pela [4tune](https://4tune.ai/):")

st.markdown("")

col1, col2, col3 = st.columns(3)

with col1:
    idade = st.radio("Idade", ["[18-24]", "[25-30]", "[31-40]", "[41-50]"])
    if idade == "[18-24]":
        idade = '(18, 24]'
    elif idade == "[25-30]":
        idade = '(24, 30]'
    elif idade == "[31-40]":
        idade = '(30, 40]'
    else:
        idade = '(40, 50]'   
    prof = st.selectbox('Profissão', ['Cientista de Dados', 'Analista de Dados', 'Engenheiro de Dados', "Dev/Engenheiro de Software",
    'Analista de BI', 'Business Analyst/Analista de Negócios', 'Analista de Inteligência de Mercado', 'Engenheiro de ML', 'Estatístico',
    'Engenheiro', 'ADM de Banco de dados', 'Economista', 'Analista de Marketing', 'Outras'])
    if prof == 'Cientista de Dados':
        prof = 'Data Scientist/Cientista de Dados'
    if prof == 'Analista de dados':
        prof = 'Data Analyst/Analista de Dados'
    if prof == 'Engenheiro de Dados':
        prof = 'Data Engineer/Engenheiro de Dados'
    if prof == 'Dev/Engenheiro de Software':
        prof = 'Desenvolvedor ou Engenheiro de Software'
    if prof == "Analista de BI":
        prof = 'Business Intelligence/Analista de BI'
    if prof == 'Engenheiro de ML':
        prof = 'Engenheiro de Machine Learning'
    if prof == 'ADM de Banco de dados':
        prof = 'DBA/Administrador de Banco de Dados'
    porte = st.radio('Porte da empresa onde trabalha', ['Micro', 'Pequeno', 'Médio', 'Grande'])
    setor = st.selectbox('Setor de Mercado', ['TI', 'Finanças ou Bancos', 'Varejo', 'Setor Público', 'Educação', 'Área da Saúde', 'Marketing', 'Internet/Ecommerce',
    'Indústria (Manufatura)', 'Telecomunicação', 'Entretenimento/Esportes', 'Agronegócios', 'Seguros ou Previdência', 'Setor Alimentício', 'Setor Automotivo', 'Setor Farmaceutico', 'Outras'])
    if setor == 'TI':
        setor = 'Tecnologia/Fábrica de Software'
    if setor == 'Entretenimento/Esportes':
        setor = 'Entretenimento ou Esportes'
    manager = st.radio('Cargo de Gestão', ['Sim', 'Não'])
    if manager == 'sim':
        manager = 0.0
    else:
        manager = 1.0
with col2:
    degree_level = st.radio('Escolaridade', ['Graduação/Bacharelado', 'Pós-Graduação', 'Mestrado', 'Doutorado ou Phd', 'Estudante de Graduação',
    'Não tenho graduação formal'])
    degree_area = st.selectbox('Área de Formação', ['TI/Engenharia de Software', 'Outras Engenharias', 'Economia/ADM/Contabilidade', 'Estatística/Matemática', 'Química / Física', 'Comunicação', 'Ciências Sociais', 'Outras'])
    if degree_area == 'TI/Engenharia de Software':
        degree_area = 'Computação / Engenharia de Software / Sistemas de Informação'
    if degree_area == 'Economia/ADM/Contabilidade':
        degree_area = 'Economia/ Administração / Contabilidade / Finanças'
    if degree_area == 'Estatística/Matemática':
        degree_area = 'Estatística/ Matemática / Matemática Computacional'
    if degree_area == 'Comunicação':
        degree_area = 'Marketing / Publicidade / Comunicação / Jornalismo'
    python = st.radio('Linguagem Python', ['Sim', 'Não'])
    if python == 'Sim':
        python = 1
    else:
        python = 0
    r = st.radio('Linguagem R', ['Sim', 'Não'])
    if r == 'Sim':
        r = 1
    else:
        r = 0
    sql = st.radio('Linguagem SQL', ['Sim', 'Não'])
    if sql == 'Sim':
        sql = 1
    else:
        sql = 0

with col3:
    job_type = st.selectbox('Tipo de trabalho', ['Empregado (CLT)', 'Empreendedor/CNPJ', 'Estagiário', 'Desempregado',
    'Pesquisador (área acadêmica)', 'Servidor público', 'Somente estudante (graduação)', 'Somente estudante (pós-graduação)', 'Freelancer', 'Prefiro não dizer', 'Não busco recolocação'])
    if job_type == 'Empreendedor/CNPJ':
        job_type = 'Empreendedor ou Empregado (CNPJ)'
    if job_type == 'Pesquisador (área acadêmica)':
        job_type = 'Trabalho na área Acadêmica/Pesquisador'
    if job_type == 'Não busco recolocação':
        job_type = 'Desempregado e não estou buscando recolocação'
    if job_type == 'Desempregado':
        job_type = 'Desempregado e não estou buscando recolocação'
    estado= st.radio('Estado', ['São Paulo (SP)', 'Minas Gerais (MG)', 'Rio de Janeiro (RJ)', 'Paraná (PR)',
    'Rio Grande do Sul (RS)', 'Santa Catarina (SC)', 'Espírito Santo (ES)', 'Outro'])
    experiencia_ds = st.radio('Experiência em DS', ['de 1 a 2 anos', 'Menos de 1 ano', 'de 2 a 3 anos', 'de 4 a 5 anos', 'de 6 a 10 anos',
    'Mais de 10 anos', 'Não tenho experiência em DS'])
    if experiencia_ds == 'Não tenho experiência em DS':
        experiencia_ds = 'Não tenho experiência na área de dados'
  
output = ''

input_dict = {'age_bins': idade, 'role': prof, 'degree_level': degree_level, 'time_experience_ds': experiencia_ds,
'job_situation' : job_type, 'degree_area': degree_area, 'state': estado, 'manager': manager,
'market_sector': setor, 'sql': sql, 'r': r, 'python': python, 'porte': porte}
input_df = pd.DataFrame([input_dict])

if st.button('Executar modelo'):
    output = predict_model(model, data = input_df)
    output_value =  output['Label'][0]
    inflação_acumulada = 0.25
    dif_PJ_CLT = 0.4
    if job_type == 'Empreendedor ou Empregado (CNPJ)':
        output_value2 = output_value*inflação_acumulada + output_value + output_value*dif_PJ_CLT
    else:
         output_value2 = output_value*inflação_acumulada + output_value
    int_conf = 351.94
    output_value3 = round(output_value2 - int_conf, 2)
    output_value4 = round(output_value2 + int_conf, 2)
    st.success('O seu salário estaria entre R$ {} - {}'.format(output_value3, output_value4))
   


