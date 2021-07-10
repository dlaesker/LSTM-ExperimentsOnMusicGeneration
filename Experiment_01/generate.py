'''
*************************
* AUTHOR: Denis Laesker *
*************************
EXPERIMENT #01 of LSTM MUSIC GENERATION

Description: A rather simplistic approach to using an LSTM to generate a MIDI file.

Procedure
1. Read in notes and velocities from a handful of audio_samples "deemed" to be associated with evoking "sadness."
2. Create two LSTM models: each devised to handle the generation of notes and velocities, respectively.
3. Feed unseen data into the model.
4. Write the predictions to a MIDI file.

Notes: This first experiment turned out to be better than I had expected. Although the generated files are far from being "audible," or pleasant to the ears, the overall theme ("sadness") can be heard. Modifications to the model and the temporal aspect of the generated song ought to be carried out in future experiments.

WARNING: Make sure your volume is low when listening to the generated files. Moreover, if possible, use a piano (or piano-like) timbre when listening to the files.

'''
import mido
from mido import MidiFile, MidiTrack, Message
from numpy import array
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import sys
import os
import random

# Split a univariate sequence based on n_steps
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		end_ix = i + n_steps
		if end_ix > len(sequence) - 1:
			break
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	
	return array(X), array(y)


# Split and reshape the data.
def prepare_data(seq, n_steps, n_features):
	X, y = split_sequence(seq, n_steps)
	X = X.reshape((X.shape[0], X.shape[1], n_features))
	
	return X, y

# For now a simple mode was devised.
def create_model(input_shape):
	model = Sequential()
	model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape))
	model.add(LSTM(50, activation='relu'))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mse')
	
	return model

# A utility to shuffle a sequence of data
def _shuffle(sequence):
	seq = sequence
	random.shuffle(seq)
	
	return seq

# Where are the samples located?
samples_at = '../Samples/'

# Retrieve audio samples
audio_samples = None
for dirpath, dirnames, filenames in os.walk(samples_at):
	audio_samples = filenames
	break;

# Clean up: only retrieve files that end with '.mid'
audio_samples = [samples_at + sample for sample in audio_samples if sample.endswith(".mid")]

notes = list()
velocity = list()

# For every audio sample
for sample in audio_samples:
	mid = MidiFile(sample)									# Open file
	for	i, track in enumerate(mid.tracks):	# for every track in the file
		for msg in track:											# and for every message
			if not msg.is_meta:									# skip if it not sound data
				try:
					if(msg.type == 'note_on'):			# Was the note played?
						data = msg.bytes()
						notes.append(data[1])					# data[1] == note
						velocity.append(data[2])			# data[2] == velocity
				except:
					continue


# a sample will have size 3 and will define a single feature
n_steps = 3
n_features = 1

# Percentage of data used for training. Should check how many data points there are.
train_percent = .8
n_train = int(train_percent * len(notes))


# Create two separate models: one for notes, the other for velocities.
# n_...: notes
# v_...: velocity
n_X_train, n_y_train 	= prepare_data(notes[0:n_train], n_steps, n_features)
n_X_test, n_y_test 		= prepare_data(_shuffle(notes[n_train:]), n_steps, n_features)

v_X_train, v_y_train 	= prepare_data(velocity[0:n_train], n_steps, n_features)
v_X_test, v_y_test 		= prepare_data(_shuffle(velocity[n_train:]), n_steps, n_features)

n_model = create_model((n_steps, n_features))
v_model = create_model((n_steps, n_features))

n_model.fit(n_X_train, n_y_train, epochs=20, verbose=0)
v_model.fit(v_X_train, v_y_train, epochs=20, verbose=0)

# May want to save the model
#n_model.save("notes_model")
#v_model.save("vel_model")

# Predictions
notes_hat = n_model.predict(n_X_test, verbose=0)
vel_hat 	= v_model.predict(v_X_test, verbose=0)

# Preparation to write a MIDI file.
mid = MidiFile()
track = MidiTrack()

note_on = 147

t = 0
for i in range(len(notes_hat)):
	note = np.asarray([note_on, notes_hat[i], vel_hat[i]])
	bytes = note.astype(int)
	msg = Message.from_bytes(bytes[0:3])
	t += 1
	msg.time = t
	track.append(msg)
mid.tracks.append(track)
mid.save('supposed_to_be_a_sad_song.mid')

