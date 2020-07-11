import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from pathlib import Path
from typing import List, Union
import functools

SHUFFLE_BUFFER_SIZE = 1000
BATCH_SIZE = 64
STATE_SIZE = 64
CHARACTER_EMBEDDING_SIZE = 64
CHECKPOINT_FILEPATH = Path("tmp/checkpoint/cp-{epoch:04d}.ckpt")
CHECKPOINT_DIR = CHECKPOINT_FILEPATH.parent

encoder = tfds.features.text.ByteTextEncoder()


class LabelEncoder:
    vocab_size = 29

    @staticmethod
    def encode_label(label: str) -> Union[List[int], None]:
        encoded_label = [27]
        for char in label.lower():
            encoded_char = ord(char) - ord("a") + 1
            if encoded_char < 0 or encoded_char > 26:
                return None
            encoded_label.append(encoded_char)
        encoded_label.append(28)
        return encoded_label


def get_data(examples_path: Path, labels_path: Path):
    with open(examples_path) as fe, open(labels_path) as fl:
        for (example, label) in zip(fe, fl):
            example = example.strip()
            label = label.strip()
            encoded_label = LabelEncoder.encode_label(label)
            if encoded_label is None:
                continue
            decoder_input = encoded_label[:-1]
            decoder_output = encoded_label[1:]
            yield (
                {
                    "encoder_input": encoder.encode(example),
                    "decoder_input": tf.one_hot(decoder_input, LabelEncoder.vocab_size),
                },
                tf.one_hot(decoder_output, LabelEncoder.vocab_size),
            )


def create_dataset(examples_path: Path, labels_path: Path):
    return (
        tf.data.Dataset.from_generator(
            functools.partial(get_data, examples_path, labels_path),
            ({"encoder_input": tf.float32, "decoder_input": tf.float32}, tf.float32),
            (
                {
                    "encoder_input": tf.TensorShape([None]),
                    "decoder_input": tf.TensorShape([None, LabelEncoder.vocab_size]),
                },
                tf.TensorShape([None, LabelEncoder.vocab_size]),
            ),
        )
        .cache()
        .shuffle(SHUFFLE_BUFFER_SIZE)
        .padded_batch(BATCH_SIZE)
    )


def train():
    train_data = create_dataset(Path("crosswords", "examples_train.txt"), Path("crosswords", "labels_train.txt"))
    validation_data = create_dataset(
        Path("crosswords", "examples_validation.txt"), Path("crosswords", "labels_validation.txt")
    )

    # Define an input sequence and process it.
    encoder_inputs = keras.layers.Input(shape=(None,), name="encoder_input")
    x = keras.layers.Embedding(encoder.vocab_size, CHARACTER_EMBEDDING_SIZE, mask_zero=True)(encoder_inputs)
    x, state_h, state_c = keras.layers.LSTM(STATE_SIZE, return_state=True, name="encoder_lstm")(x)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = keras.layers.Input(shape=(None, LabelEncoder.vocab_size), name="decoder_input")
    x = keras.layers.Masking()(decoder_inputs)
    x = keras.layers.LSTM(STATE_SIZE, return_sequences=True, name="decoder_lstm")(x, initial_state=encoder_states)
    decoder_outputs = keras.layers.Dense(LabelEncoder.vocab_size, activation="softmax")(x)

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

    model.fit(
        train_data, validation_data=validation_data, epochs=50, callbacks=[model_checkpoint_callback],
    )
