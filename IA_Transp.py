import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

train_file = 'dados_entrega.csv'
result_column = 'Prazo_Entrega'
res_file = 'dados_novos.csv'

def SaveModel():
    # lê o arquivo CSV
    dataframe = pd.read_csv(train_file, sep=';', encoding='ANSI')

    # Transforma as variáveis categóricas em numéricas
    dataframe = pd.get_dummies(dataframe)

    # Extrai os nomes das colunas usadas para treinar o modelo
    feature_names = list(dataframe.drop(result_column, axis=1).columns)

    # Salva a lista em um arquivo com extensão pkl usando a biblioteca pickle
    import pickle

    with open('model_columns.pkl', 'wb') as file:
        pickle.dump(feature_names, file)



    # separa as colunas de entrada (X) e a coluna de saída (y)
    X = dataframe.drop(result_column, axis=1)
    y = dataframe[result_column]

    # divide os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # cria o modelo de regressão
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # treina o modelo com os dados de treinamento
    model.fit(X_train, y_train)

    # faz previsões usando os dados de teste
    y_pred = model.predict(X_test)

    # avalia o desempenho do modelo usando o MSE (erro quadrático médio)
    # mse = mean_squared_error(y_test, y_pred)
    # print(f'MSE: {mse}')

    import joblib

    # salvar o modelo em um arquivo .pkl
    joblib.dump(model, 'model.pkl')

def Run():
    import pandas as pd
    import joblib

    # Carrega o modelo treinado
    model = joblib.load('model.pkl')

    # Carrega os dados de entrada
    dataframe = pd.read_csv(res_file, sep=';', encoding='ANSI')

    # Lê as colunas necessárias do arquivo CSV
    cols = list(dataframe.columns)
    dataframe = pd.read_csv(res_file, usecols=cols, sep=';', encoding='ANSI')

    # Transforma as variáveis categóricas em numéricas
    dataframe = pd.get_dummies(dataframe)

    # Obtém as colunas que o modelo espera
    expected_cols = joblib.load('model_columns.pkl')

    # Adiciona quaisquer colunas faltantes com valor 0
    for col in expected_cols:
        if col not in dataframe.columns:
            dataframe[col] = 0

    # Reordena as colunas para ficarem na mesma ordem que o modelo espera
    dataframe = dataframe[expected_cols]

    # Faz a previsão do tempo de entrega
    res = model.predict(dataframe)
    AddToTrain(res)
    dataframe[result_column] = res
    print("O tempo de entrega previsto é: ", res)

def AddToTrain(prev):
    # lê o arquivo CSV com os novos dados
    new_data = pd.read_csv(res_file, sep=';', encoding='ANSI')
    new_data[result_column] = prev
    # adiciona os novos dados ao arquivo CSV de treinamento
    with open(train_file, 'a') as file:
        new_data.to_csv(file, header=False, index=False, lineterminator='\n', sep=';')
SaveModel()
Run()
