import torch
from SurvSet.data import SurvLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
def numpy_to_tensor(np_tensors):
    torch_tensors = []
    for tensor in np_tensors:
        torch_tensors.append(torch.Tensor(tensor))

    return torch_tensors


def load_datasets(ds_name="ova",drop_first=False,normalize=True,test_size=0.2):
    # SurvLoader to load in time to event datasets
    loader = SurvLoader()
    data_df, _ = loader.load_dataset(ds_name=ds_name).values()

    # Event vector where 1 denotes event occurred, 0 denotes no event occured
    event = data_df.pop("event").values.reshape(-1,1)
    # If event=1, time is the time of event, if event=0, time is the right censor time
    time = data_df.pop("time").values.reshape(-1,1)

    # one hot encode all the "fac" categorical variables
    data_df_ohe = pd.get_dummies(data_df,columns=[col for col in data_df.columns if "fac" in col],drop_first=drop_first,dtype=float)

    # need to improve logic of dropping  pid
    data_df_ohe.drop("pid",inplace=True,axis=1)

    # get numpy array for features
    X = data_df_ohe.values

    # convert numpy array to tensors
    X,time,event = numpy_to_tensor((X,time,event))


    # Train-Test split for loading the data
    X_train,X_test, time_train,time_test, event_train,event_test = train_test_split(X,time,event,stratify=event,test_size=test_size)


    if normalize:
        std = StandardScaler()
        X_train = std.fit_transform(X_train)
        X_test = std.transform(X_test)
        X_train,X_test = numpy_to_tensor((X_train,X_test))

    dataset_train = TensorDataset(X_train,time_train,event_train)
    dataset_test = TensorDataset(X_test,time_test,event_test)

    if normalize:
        dataset_train.mean = dataset_test.mean = torch.Tensor(std.mean_).reshape(1,-1)
        dataset_train.std = dataset_test.std = torch.Tensor(std.scale_).reshape(1,-1)

    return dataset_train,dataset_test

def load_dataframe(ds_name="ova",drop_first=False,normalize=True,test_size=0.2):
    # SurvLoader to load in time to event datasets
    loader = SurvLoader()
    data_df, _ = loader.load_dataset(ds_name=ds_name).values()

    # one hot encode all the "fac" categorical variables
    data_df_ohe = pd.get_dummies(data_df,columns=[col for col in data_df.columns if "fac" in col],drop_first=drop_first,dtype=float)

    # need to improve logic of dropping  pid
    data_df_ohe.drop("pid",inplace=True,axis=1)

    # Train-Test split for loading the data
    data_train,data_test = train_test_split(data_df_ohe,stratify=data_df_ohe.event,test_size=test_size)

    features = [col for col in data_train.columns if col not in ["event","time"]]

    if normalize:
        std = StandardScaler()
        data_train.loc[:,features] = std.fit_transform(data_train.loc[:,features])
        data_test.loc[:, features] = std.transform(data_test.loc[:, features])

    return data_train,data_test
if __name__ == "__main__":
    dataset_train,dataset_test = load_datasets()

    print("Train len: ",len(dataset_train))
    print("Test len: ",len(dataset_test))

    load_dataframe(ds_name="ova",normalize=True,test_size=0.2)