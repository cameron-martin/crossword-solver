import datetime
import functools
from pathlib import Path
from typing import List, Union

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from unidecode import unidecode

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

SHUFFLE_BUFFER_SIZE = 1000
BATCH_SIZE = 64
STATE_SIZE = 128
CHARACTER_EMBEDDING_SIZE = 64
CHECKPOINT_FILEPATH = Path("tmp/checkpoint/cp-{epoch:04d}.ckpt")
CHECKPOINT_DIR = CHECKPOINT_FILEPATH.parent


class LabelEncoder:
    start = 1
    end = 2
    a_codepoint = ord("a")
    z_codepoint = ord("z")

    vocab_size = z_codepoint - a_codepoint + 4

    @staticmethod
    @tf.function
    def codepoint_to_index(codepoint):
        return codepoint - LabelEncoder.a_codepoint + 3

    @staticmethod
    def index_to_codepoint(index):
        return index + LabelEncoder.a_codepoint - 3


@tf.function
def convert_label(example, label):
    return example, LabelEncoder.codepoint_to_index(label)


def create_dataset(examples_path: Path, labels_path: Path):
    example_chars = tf.data.TextLineDataset(str(examples_path)).map(
        lambda line: tf.strings.unicode_decode(line, "UTF-8")
    )
    label_chars = tf.data.TextLineDataset(str(labels_path)).map(
        lambda line: tf.strings.unicode_decode(tf.strings.lower(line), "UTF-8")
    )

    dataset = (
        tf.data.Dataset.zip((example_chars, label_chars))
        .filter(
            lambda _, label: tf.reduce_all(
                tf.math.logical_and(label >= LabelEncoder.a_codepoint, label <= LabelEncoder.z_codepoint)
            )
        )
        .map(convert_label)
        .map(
            lambda example, label: (
                {
                    "encoder_input": example,
                    "decoder_input": tf.one_hot(tf.concat([[LabelEncoder.start], label], 0), LabelEncoder.vocab_size),
                },
                tf.one_hot(tf.concat([label, [LabelEncoder.end]], 0), LabelEncoder.vocab_size),
            )
        )
    )

    return dataset.cache().shuffle(SHUFFLE_BUFFER_SIZE).padded_batch(BATCH_SIZE).prefetch(2)


def train():
    train_data = create_dataset(Path("tmp", "clues", "examples_train.txt"), Path("tmp", "clues", "labels_train.txt"))
    validation_data = create_dataset(
        Path("tmp", "clues", "examples_validation.txt"), Path("tmp", "clues", "labels_validation.txt")
    )

    # Define an input sequence and process it.
    encoder_inputs = keras.layers.Input(shape=(None,), name="encoder_input")
    x = keras.layers.Embedding(256, CHARACTER_EMBEDDING_SIZE, mask_zero=True)(encoder_inputs)
    x, state_h, state_c = keras.layers.LSTM(STATE_SIZE, return_state=True, name="encoder_lstm")(x)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = keras.layers.Input(shape=(None, LabelEncoder.vocab_size), name="decoder_input")
    x = keras.layers.Masking()(decoder_inputs)
    x, _, _ = keras.layers.LSTM(STATE_SIZE, return_sequences=True, name="decoder_lstm", return_state=True)(
        x, initial_state=encoder_states
    )
    decoder_outputs = keras.layers.Dense(LabelEncoder.vocab_size, activation="softmax", name="decoder_dense")(x)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=[keras.metrics.CategoricalAccuracy()],
    )

    model.summary()

    latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest is not None:
        model.load_weights(latest)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=str(CHECKPOINT_FILEPATH), save_weights_only=True
    )

    log_dir = "tmp/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True
    )

    model.fit(
        train_data,
        validation_data=validation_data,
        epochs=100,
        callbacks=[model_checkpoint_callback, tensorboard_callback, early_stopping_callback],
    )

    model.save("tmp/model.h5")


class Predictor:
    def __init__(self):
        model = keras.models.load_model("tmp/model.h5")

        encoder_inputs = model.input[0]
        encoder_outputs, state_h_enc, state_c_enc = model.get_layer(name="encoder_lstm").output
        encoder_states = [state_h_enc, state_c_enc]
        self.encoder_model = keras.Model(encoder_inputs, encoder_states)

        decoder_inputs = model.input[1]
        decoder_state_input_h = keras.layers.Input(shape=(STATE_SIZE,))
        decoder_state_input_c = keras.layers.Input(shape=(STATE_SIZE,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = model.get_layer(name="decoder_lstm")
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = model.get_layer(name="decoder_dense")
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = keras.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    def predict(self, clue: str):
        ascii_clue = unidecode(clue)
        codepoints = tf.strings.unicode_decode(ascii_clue, "UTF-8")
        states_value = self.encoder_model.predict(tf.reshape(codepoints, (1, *codepoints.shape)))

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, LabelEncoder.vocab_size))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, 27] = 1.0

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        decoded_sentence = ""
        while True:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])

            # Exit condition: either hit max length
            # or find stop character.
            if sampled_token_index == LabelEncoder.end or len(decoded_sentence) > 100:
                return decoded_sentence

            sampled_char = chr(LabelEncoder.index_to_codepoint(sampled_token_index))
            decoded_sentence += sampled_char

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, LabelEncoder.vocab_size))
            target_seq[0, 0, sampled_token_index] = 1.0

            # Update states
            states_value = [h, c]


def predict_repl():
    predictor = Predictor()
    while True:
        clue = input("Clue: ")
        print(predictor.predict(clue))
