import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from recurrent import BidirectionalLstmAutoEncoder

DO_TRAINING = False

def main():
    data_dir_path = './data'
    model_dir_path = './models'
    ecg_data = pd.read_csv(data_dir_path + '/ecg_discord_test.csv', header=None)
    print(ecg_data.head())
    ecg_np_data = ecg_data.as_matrix()
    scaler = MinMaxScaler()
    ecg_np_data = scaler.fit_transform(ecg_np_data)
    print(ecg_np_data.shape)

    ae = BidirectionalLstmAutoEncoder()

    if DO_TRAINING:
        ae.fit(ecg_np_data[:23, :], model_dir_path=model_dir_path, estimated_negative_sample_ratio=0.9)

    ae.load_model(model_dir_path)
    anomaly_information = ae.anomaly(ecg_np_data[:23, :])

    predicted_data = ae.predict_dataset(ecg_np_data[:23, :])
    print(predicted_data[1]*1000)

    reconstruction_error = []

    for idx, (is_anomaly, dist) in enumerate(anomaly_information):

        print('# ' + str(idx) + ' is ' + ('abnormal' if is_anomaly else 'normal') + ' (dist: ' + str(dist) + ')')
        reconstruction_error.append(dist)


if __name__ == '__main__':
    main()
