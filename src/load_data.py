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


def load_datasets(ds_name="ova",normalize=True,test_size=0.2):
    # SurvLoader to load in time to event datasets
    loader = SurvLoader()
    data_df, _ = loader.load_dataset(ds_name=ds_name).values()

    # Event vector where 1 denotes event occurred, 0 denotes no event occured
    event = data_df.pop("event").values.reshape(-1,1)
    # If event=1, time is the time of event, if event=0, time is the right censor time
    time = data_df.pop("time").values.reshape(-1,1)

    # one hot encode all the "fac" categorical variables
    data_df_ohe = pd.get_dummies(data_df,columns=[col for col in data_df.columns if "fac" in col],dtype=float)

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

    return dataset_train,dataset_test

if __name__ == "__main__":
    dataset_train,dataset_test = load_datasets()

    print("Train len: ",len(dataset_train))
    print("Test len: ",len(dataset_test))

    print()