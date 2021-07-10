# LSTM Experiments On Music Generation

## This repository serves as a place for exploration with respect to music generation using LSTMs. 

### EXPERIMENT #01: 
Code and generated samples can be found in: Experiment_01

A rather simplistic approach to using an LSTM model to generate a MIDI file. 

Procedure
1. Read in notes and velocities from a handful of audio_samples "deemed" to be associated with evoking "sadness."
2. Create two LSTM models: each devised to handle the generation of notes and velocities, respectively.
3. Feed unseen data into the model.
4. Write the predictions to a MIDI file.

Notes: This first experiment turned out to be better than I had expected. Although the generated files are far from being "audible," or pleasant to the ears, the overall theme ("sadness") can be heard. Modifications to the model and the temporal aspect of the generated song ought to be carried out in future experiments.

WARNING: Make sure your volume is low when listening to the generated files. Moreover, if possible, use a piano (or piano-like) timbre when listening to the files.
