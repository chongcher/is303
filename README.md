## IS303 GPU Programming example
Modified from the [sample code](https://github.com/yoonkim/CNN_sentence) for the paper [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882).

Runs the model on Pang and Lee's [subjectivity dataset v1.0](https://www.cs.cornell.edu/people/pabo/movie-review-data/).

### Requirements
Code is written in Python (2.7) and requires Theano (>0.7).

Using the pre-trained `word2vec` vectors will also require downloading the binary file from
https://code.google.com/p/word2vec/

### Data Preprocessing
To process the raw data, run

```
python process_data.py <vector_file>
```

where `<vector_file>` points to the word2vec binary file (i.e. `GoogleNews-vectors-negative300.bin` file).
This will create a pickle object called `is303.pkl` in the same folder, which contains the dataset
in the right format.


### Running the models (CPU)
Example commands:

```
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python conv_net_sentence.py <pickle_file>
```

where `<pickle_file>` points to the pickle file created in the Data preprocessing step.

This will run the CNN model on the CPU.
To use the GPU, simply change `device=cpu` to `device=gpu` (or whichever gpu you are using).

For example:
```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python conv_net_sentence.py <pickle_file>
```
