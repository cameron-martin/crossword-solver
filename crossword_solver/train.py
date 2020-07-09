import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import tensorflow_text as tftext
from pathlib import Path
import numpy as np

BUFFER_SIZE = 50000
BATCH_SIZE = 64
VALIDATION_SIZE = 5000

encoder = tfds.features.text.ByteTextEncoder(additional_tokens=["<SOS>", "<EOS>"])

def encode(text_tensor, label_tensor):
  encoded_text = encoder.encode(text_tensor.numpy())
  encoded_label = encoder.encode(label_tensor.numpy())
  return encoded_text, encoded_label

def encode_map_fn(text, label):
  # py_func doesn't set the shape of the returned tensors.
  encoded_text, encoded_label = tf.py_function(encode, 
                                       inp=[text, label], 
                                       Tout=(tf.int64, tf.int64))

  # `tf.data.Datasets` work best if all components have a shape set
  #  so set the shapes manually: 
  encoded_text.set_shape([None])
  encoded_label.set_shape([None])

  return tf.one_hot(encoded_text, encoder.vocab_size), tf.one_hot(encoded_label, encoder.vocab_size)

class CharacterMap():
  def __init__(self):
    self.map = {}
    self.current_index = 1

  def get(self, character) -> int:
    if character in self.map:
      return self.map[character]
    else:
      self.map[character] = self.current_index
      self.current_index += 1
      return self.map[character]

  @property
  def size(self):
    return self.current_index

def train():
  # # Read data
  # sentences = tf.data.TextLineDataset(str(Path("crosswords", "examples.txt")))
  # labels = tf.data.TextLineDataset(str(Path("crosswords", "labels.txt")))
  # dataset = tf.data.Dataset.zip((sentences, labels)) # Dataset<(str, str)>

  # dataset = dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
  # dataset = dataset.map(encode_map_fn) # Dataset<(Tensor<[None, n]>, Tensor<[None, n]>)>
  # dataset = dataset.map(lambda example, label: ([example, label[0:-1]], label[1:]))

  # train_data = dataset.skip(VALIDATION_SIZE).shuffle(BUFFER_SIZE)
  # train_data = train_data.padded_batch(BATCH_SIZE)

  # validation_data = dataset.take(VALIDATION_SIZE)
  # validation_data = validation_data.padded_batch(BATCH_SIZE)

  # print(list(validation_data.as_numpy_iterator()))

  # Read data
  
  char_map = CharacterMap()
  with open(Path("crosswords", "examples.txt")) as f:
    lines = list(line.strip() for line in f)
    example_count = len(lines)
    max_sentence_length = max(len(line) for line in lines)
    examples = np.empty((example_count, max_sentence_length), dtype='float32')
    for i, line in enumerate(lines):
      for j in range(0, max_sentence_length):
        if j < len(line):
          examples[i][j] = char_map.get(line[j])
        else:
          examples[i][j] = 0
    
  with open(Path("crosswords", "labels.txt")) as f:
    lines = list(bytes(f"\t{line}", 'utf-8') for line in f)
    example_count = len(lines)
    max_sentence_length = max(len(line) for line in lines) - 1
    decoder_input_data = np.empty((example_count, max_sentence_length), dtype='float32')
    decoder_target_data = np.empty((example_count, max_sentence_length), dtype='int')
    for i, line in enumerate(lines):
      for j in range(0, max_sentence_length):
        if (j+1) < len(line):
          decoder_input_data[i][j] = char_map.get(line[j])
          decoder_target_data[i][j] = char_map.get(line[j+1])
        else:
          decoder_input_data[i][j] = 0
          decoder_target_data[i][j] = 0


  decoder_target_data_one_hot = np.eye(char_map.size)[decoder_target_data]

  # Define an input sequence and process it.
  encoder_inputs = keras.layers.Input(shape=(None,), name="encoder_input")
  x = keras.layers.Embedding(char_map.size, 64, mask_zero=True)(encoder_inputs)
  x, state_h, state_c = keras.layers.LSTM(64, return_state=True, name="encoder_lstm")(x)
  # We discard `encoder_outputs` and only keep the states.
  encoder_states = [state_h, state_c]

  # Set up the decoder, using `encoder_states` as initial state.
  decoder_inputs = keras.layers.Input(shape=(None,), name="decoder_input")
  x = keras.layers.Embedding(char_map.size, 64, mask_zero=True)(decoder_inputs)
  x = keras.layers.LSTM(64, return_sequences=True, name="decoder_lstm")(x, initial_state=encoder_states)
  decoder_outputs = keras.layers.Dense(char_map.size, activation='softmax')(x)

  # Define the model that will turn
  # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
  model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[keras.metrics.CategoricalAccuracy()])

  model.summary()

  model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=str(Path("checkpoint")))

  model.fit([examples, decoder_input_data], decoder_target_data_one_hot,
    batch_size=BATCH_SIZE,
    epochs=50,
    validation_split=0.2,
    callbacks=[
      model_checkpoint_callback
    ])

