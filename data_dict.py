def get_data_dictionary(data):
    data_dict = pd.DataFrame(columns = ['Column','Count','Unique Values','Range',
                                        'Null values','Possible Values'])
    for i in data.columns:
        col = i
        count = data[col].shape[0]
        unique_values = data[col].nunique()
        range = f"{data[col].min()} - {data[col].max()}"
        nulls = data[col].isna().sum()
        values = list(data[col].sample(25))
        values = list(set(values))[:5]
        print(unique_values)
        data_dict.loc[len(data_dict)] = [col,count,unique_values,range,nulls,values]
    return data_dict
