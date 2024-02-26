def get_data_dictionary(data):
    data_dict = pd.DataFrame(columns = ['Column','Count','Unique Values','Range',
                                        'Null values','Possible Values'])
    for col in data.columns:
        count = data[col].shape[0]
        unique_values = data[col].nunique()
        if unique_values>1:
            range = f"{data[col].min()} - {data[col].max()}"
        else:
            range = np.nan
        nulls = data[col].isna().sum()
        values = list(data[col].sample(frac = 0.25, replace=False, random_state=42))
        values = list(set(values))[:5]
        data_dict.loc[len(data_dict)] = [col,count,unique_values,range,nulls,values]
    return data_dict
