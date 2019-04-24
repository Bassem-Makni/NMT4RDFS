import configparser
import json
import logging
import os
from datetime import datetime
from os import path

import numpy as np
import pandas as pd
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Input, Dense, Dropout, Bidirectional, GRU, RepeatVector, TimeDistributed
from keras.models import Model

from nn_utils import CSVLoggerTimed, train_validate_test_split, to_categorical, true_acc
from utils import deserialize

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

config = configparser.ConfigParser()
with open('../config.json', 'r') as f:
    config = json.load(f)

LUBM_ENCODING_DIRECTORY = config['LUBM']['ENCODING_PARAMS']['ENCODING_DIRECTORY']
DATASET_ENCODING_FILE = config['LUBM']['ENCODING_PARAMS']['DATASET_ENCODING_FILE']

folder_name = datetime.now().strftime('%d_%m_%Y_%H_%M_%S') + "/"

LOGGING_FOLDER = config['LUBM']["TRAINING"]["LOGGING_FOLDER"] + folder_name
MODEL_FOLDER = config['LUBM']["TRAINING"]["MODEL_FOLDER"] + folder_name

TRAINING_SET_PERCENT = config['LUBM']["TRAINING"]["TRAINING_SET_PERCENT"]
VALIDATION_SET_PERCENT = config['LUBM']["TRAINING"]["VALIDATION_SET_PERCENT"]
EPOCHS = config['LUBM']["TRAINING"]["EPOCHS"]
EMBEDDING_MATRIX_CATALOGUE_FILE = config['LUBM']['ENCODING_PARAMS']['EMBEDDING_MATRIX_CATALOGUE_FILE']
BATCH_SIZE = config['LUBM']["TRAINING"]["BATCH_SIZE"]
MODEL_ARCHITECTURE = config['LUBM']["TRAINING"]["MODEL_ARCHITECTURE"]


def create_input_target_arrays(rdf_dataframe, embed_matrix, vocab_size):
    x_input = rdf_dataframe['input_graph_words']
    x_input = np.stack(x_input)
    x_input = embed_matrix[x_input]
    y_target = rdf_dataframe['inference_graph_words']
    y_target = np.stack(y_target)
    y_target_categorical = to_categorical(y_target, vocab_size)
    return x_input, y_target_categorical


def create_graph_words_translation_model(x, y):
    graph_input = Input(shape=(x.shape[1], x.shape[2]), name='input_graph_words_sequence')
    graph_input_dense = Dense(256, name='graph_input_dense')(graph_input)
    graph_dropout1 = Dropout(0.2, name='graph_dropout1')(graph_input_dense)
    graph_gru = Bidirectional(GRU(128, name="gru_sequence_encoder"), name='bidirectional')(graph_dropout1)
    graph_dropout2 = Dropout(0.2, name='graph_dropout2')(graph_gru)
    hidden_graph = RepeatVector(y.shape[1], name="repeat_vector")(graph_dropout2)
    inference_output = GRU(128, return_sequences=True, name='sequence_decoder')(hidden_graph)
    inference_output = Dropout(0.2, name="output_dropout")(inference_output)
    inference_output = TimeDistributed(Dense(y.shape[2], name="softmax_layer", activation='softmax'),
                                       name="inference_graph_words_sequence")(inference_output)
    inference_model = Model(graph_input, inference_output)
    inference_model.compile('adam', 'categorical_crossentropy', metrics=['accuracy', true_acc])
    return inference_model


def get_target_vocab_size(df_train):
    y = df_train['inference_graph_words']
    y = np.stack(y)
    return y.max() + 1


def main():
    logging.info("Creating training folders: logging folder = %s, model folder = %s", LOGGING_FOLDER, MODEL_FOLDER)
    if not path.exists(LOGGING_FOLDER):
        os.makedirs(LOGGING_FOLDER)

    if not path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)

    csv_logger = CSVLoggerTimed(LOGGING_FOLDER + '/training.csv', separator=',', append=True)
    tensorboard = TensorBoard(log_dir=LOGGING_FOLDER)
    modelcheckpoint = ModelCheckpoint(MODEL_FOLDER + '/best_model', monitor='val_loss', verbose=0, save_best_only=True,
                                      save_weights_only=True, mode='auto', period=1)

    logging.info("Loading LUBM1 graph words")

    try:
        lubm_graph_words = pd.read_pickle(LUBM_ENCODING_DIRECTORY + DATASET_ENCODING_FILE)
    except:
        logging.error("Loading LUBM1 graph words failed")
        exit()

    logging.info("Loaded LUBM1 graph words dataset of size %s", len(lubm_graph_words))

    rdf_data_train, rdf_data_val, rdf_data_test = train_validate_test_split(lubm_graph_words,
                                                                            train_percent=TRAINING_SET_PERCENT,
                                                                            validate_percent=VALIDATION_SET_PERCENT,
                                                                            stratify="graph_type")

    logging.info("Splitting LUBM1 dataset into %s training, %s validation and %s test ", len(rdf_data_train),
                 len(rdf_data_val), len(rdf_data_test))

    embedding_matrix = deserialize(LUBM_ENCODING_DIRECTORY + EMBEDDING_MATRIX_CATALOGUE_FILE)

    vocab_size = get_target_vocab_size(rdf_data_train)
    x_train, y_train = create_input_target_arrays(rdf_data_train, embedding_matrix, vocab_size)
    x_val, y_val = create_input_target_arrays(rdf_data_val, embedding_matrix, vocab_size)
    x_test, y_test = create_input_target_arrays(rdf_data_test, embedding_matrix, vocab_size)

    logging.info("Creating Neural Network models for LUBM training")

    inference_model = create_graph_words_translation_model(x_train, y_train)
    logging.info("Saving Neural Network model architecture into: %s", MODEL_FOLDER + MODEL_ARCHITECTURE)
    with open(MODEL_FOLDER + MODEL_ARCHITECTURE, "w") as json_file:
        json_file.write(inference_model.to_json())

    logging.info("Created NN model: ")
    inference_model.summary()

    logging.info("Starting training for %s epochs ", EPOCHS)
    inference_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=[csv_logger, tensorboard, modelcheckpoint])
    logging.info("Finished training")
    logging.info("Evaluating on test set")
    test_eval = inference_model.evaluate(x_test, y_test)
    logging.info("Test set accuracy: %s", test_eval[inference_model.metrics_names.index('true_acc')])


if __name__ == "__main__":
    main()
