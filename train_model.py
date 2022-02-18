import uuid
import numpy as np
import tensorflow
import valohai
import pandas as pd
import re

def log_metadata(epoch, logs):
    """Helper function to log training metrics"""
    with valohai.logger() as logger:
        logger.log('epoch', epoch)
        logger.log('accuracy', logs['accuracy'])
        logger.log('loss', logs['loss'])
        
def mbtr_ds_generator(directory_MBTR, directory_SASA):
    with open(directory_MBTR, 'rb') as f:
        x = np.load(f, encoding='bytes')
    with open(directory_SASA, 'rb') as f:
        y = np.load(f, encoding='bytes')
    return x, y



def main():
    valohai.prepare(
        step='train-model',
        image='tensorflow/tensorflow:2.6.0',
        default_parameters={
            'learning_rate': 0.001,
            'epochs': 500,
        },
        default_inputs={
            'dataset_train': 'https://valohaidataprod.blob.core.windows.net/valohaidataprod/data/01FW3/01FW3TEFNGBB0W2SYPRHDHF655/upload/mbtr_train.zip?se=2022-02-18T07%3A58%3A16Z&sp=rt&sv=2020-10-02&sr=b&sig=5xk%2FYuRuktiNSxtL90QpezbuQx1Au5JGiEAQXQMEprQ%3D',
            'dataset_train_sasa': 'https://valohaidataprod.blob.core.windows.net/valohaidataprod/data/01FW3/01FW3TEFNGBB0W2SYPRHDHF655/upload/sasa_train.zip?se=2022-02-18T07%3A57%3A54Z&sp=rt&sv=2020-10-02&sr=b&sig=B8%2F0f4m7PzaBbq8nqEK7WW%2Bfa6Xi0Z7vp57vWQ%2BNiO8%3D',
            'dataset_test': 'https://valohaidataprod.blob.core.windows.net/valohaidataprod/data/01FW3/01FW3TEFNGBB0W2SYPRHDHF655/upload/mbtr_test.zip?se=2022-02-18T07%3A58%3A36Z&sp=rt&sv=2020-10-02&sr=b&sig=5gSwhvSpgOU46g23%2BNSA20OFPajhfvqLsydEDYWxHLg%3D',
            'dataset_test_sasa': 'https://valohaidataprod.blob.core.windows.net/valohaidataprod/data/01FW3/01FW3TEFNGBB0W2SYPRHDHF655/upload/sasas_test.zip?se=2022-02-18T07%3A55%3A55Z&sp=rt&sv=2020-10-02&sr=b&sig=z27exhQmt%2BW6dIon40dvXMCbqlTTduINy88rUUsb24g%3D'
        }
    )
    
    
    input_path_train_MBTR = valohai.inputs('dataset_train').path()
    input_path_train_SASA = valohai.inputs('dataset_train_SASA').path()
    input_path_test_MBTR = valohai.inputs('dataset_test').path()
    input_path_test_SASA = valohai.inputs('dataset_test_SASA').path()
    print(input_path_train_MBTR)
    print(type(input_path_train_MBTR))
    x_train, y_train =  mbtr_ds_generator(input_path_train_MBTR, input_path_train_SASA)
    x_test, y_test = mbtr_ds_generator(input_path_test_MBTR, input_path_test_SASA)

    NN_model = tensorflow.keras.models.Sequential()
    shape = 168
    NN_model.add(tensorflow.keras.layers.Dense(shape, kernel_initializer='normal',input_dim = shape, activation='relu'))
    NN_model.add(tensorflow.keras.layers.Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(tensorflow.keras.layers.Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(tensorflow.keras.layers.Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(tensorflow.keras.layers.Dense(1, kernel_initializer='normal'))

    checkpoint_name = 'best_model.hdf5'
    checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
    early_stopping = tensorflow.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=50,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )
    callbacks_list = [checkpoint, early_stopping]
    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=valohai.parameters('learning_rate').value)
    loss_fn = 'mean_absolute_error'
    NN_model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy', 'mae'])
    
    # Print metrics out as JSON
    # This enables Valohai to version your metadata
    # and for you to use it to compare experiments

    callback = tensorflow.keras.callbacks.LambdaCallback(on_epoch_end=log_metadata)
    NN_model.fit(x_train, y_train, epochs=valohai.parameters('epochs').value, callbacks=[callbacks_list, callback])
    wights_file = checkpoint_name 
    NN_model.load_weights(wights_file)
    NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    
    

    test_loss, test_accuracy, test_mae = NN_model.evaluate(x_test, y_test)
    with valohai.logger() as logger:
        logger.log('test_accuracy', test_accuracy)
        logger.log('test_accuracy', test_mae)
        logger.log('test_loss', test_loss)

    # Write output files to Valohai outputs directory
    # This enables Valohai to version your data
    # and upload output it to the default data store

    suffix = uuid.uuid4()
    output_path = valohai.outputs().path(f'model-{suffix}.h5')
    NN_model.save(output_path)


if __name__ == '__main__':
    main()
