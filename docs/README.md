# BasicDeepNets

A collection of notebooks with basic deep neural networks implementation and explanation.


## Motivation

Due to the large number of neural network libraries with ready-to-use models, it is planned to change the introductory course in Machine Learning.
Namely, neural networks are going to be introduced along with the basic ML theory, as a possible first approach to solve applied problems: take a relevant model, train it, then analyze.
For this to work, it is necessary to separate the type of network from the optimized criterion and from the optimization algorithm in the problem statement;
it is also important to present the neural network as a *mapping* which fits the algebraic description of the feature space in which the measurements are made.
The analysis of the statistical nature of the measurements should be performed through the Bayesian derivation, which ends with an error function.


## Description

Notebooks presented in the repository are just laboratory works for third year students, which should help them become acquainted with neural networks and encourage to try to use basic neural networks as a first baseline solution for ML problems where possible.

Ideally, the repository should have the notebooks with all basic modern types of neural networks.

Below is a list of networks to be implemented (those which are already available have an associated link)

* [NN](NN.ipynb), [NN + RBF](NN+RBF.ipynb), [NN + AE](NN+AE.ipynb)
* CNN, ResNet 
* GCN
* [RNN (Text)](Sentiment_Analysis.ipynb), [RNN (Images)](ImageRNN.ipynb)
* LSTM, LSTM + Attention
* [Word2Vec](Word2Vec.ipynb), Emdeddings
* VAE
* [GAN](GAN.ipynb)
* T2T
* PLS
* [RL](RL-DQN.ipynb)
* [Seq2Seq + Attention](Seq2seq%20+%20Attention.ipynb)
* Networks that generate text, music, pictures...


## Notebook Requirements

Each network should be presented in its simplest form so that it is clear how it works.
It is advisable to do everything with PyTorch, avoiding out-of-the-box solutions.
Notebooks should have a heading, sectioning, explanatory comments (if possible, in English).

So, the notebooks must provide

* clear code for constructing and training the nets
* derscriptive text explanations, trying to express ideas in general terms, operating only with such notions as *model*, *sample*, *error function* — so that a beginner in ML can take, read, and understand everything more or less clearly
* Bayesian analysis

### Structure

Notebooks should have the following sections:
* Name
* Brief explanation
* Data loading (preferably more than one sample)
* Initial configuration
* Parameter optimization (indicating the possibility of optimizing the structure)
* Error analysis (plots)
* List of links to more detailed sources, tutorials, and alternative solutions

### Data

It is advisable to illustrate the network with a simple real task, selecting different tasks for different networks.
It is advisable to illustrate the data and the final result with a plot.

### Error Analysis

First of all, analysis of variance of error and parameters, change of error value during optimization ([learning curve](https://en.wikipedia.org/wiki/Learning_curve)), analysis of sample size sufficiency (change of variance during replenishment), analysis of structure (change of error and variance with increasing complexity).

### Notebook Quality Criterion

Criteria for the quality of laptops is just *public benefit*, and *separability from the author*: the code and text should be understandable, so that anyone can open the notebooks, change the code, and understand the core things.

One should provide explanations, explain everything with text, what happens and why.
It is assumed that the notebook will be used by third year students.
The code is possible to be changed: it is clear how to load another sample, change the network structure, change the error function.


In general, it is recommended to find a ready-made code (good external source) and just put it in order, maybe draw some plots.


## References

### Linear Models

What can be demonstrated in the practical part: solve a couple of problems, show that they can be solved analytically, using gradient methods and with the help of PyTorch or TensorFlow.
And in the theoretical part: compare ordinary linear regression and Bayesian one

* [5.06-Linear-Regression.ipynb](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.06-Linear-Regression.ipynb)
* [exploring_endorsement_revenue_nba.ipynb](https://github.com/noahgift/socialpowernba/blob/master/notebooks/exploring_endorsement_revenue_nba.ipynb)
* [ml_regression.ipynb](https://github.com/noahgift/regression-concepts/blob/master/ml_regression.ipynb)
* [deep_learning_basics.ipynb](https://github.com/lexfridman/mit-deep-learning/blob/master/tutorial_deep_learning_basics/deep_learning_basics.ipynb)
* [04_training_linear_models.ipynb](https://github.com/ageron/handson-ml/blob/master/04_training_linear_models.ipynb)
* [linregr_least_squares_fit.ipynb](https://nbviewer.jupyter.org/github/rasbt/algorithms_in_ipython_notebooks/blob/master/ipython_nbs/statistics/linregr_least_squares_fit.ipynb)
* [bayesian-linear-regression-in-python-using-machine-learning-to-predict-student-grades-part-1](https://towardsdatascience.com/bayesian-linear-regression-in-python-using-machine-learning-to-predict-student-grades-part-1-7d0ad817fca5)
* [bayesian-linear-regression-in-python-using-machine-learning-to-predict-student-grades-part-2](https://towardsdatascience.com/bayesian-linear-regression-in-python-using-machine-learning-to-predict-student-grades-part-2-b72059a8ac7e)
* [Bayesian Linear Regression Demonstration.ipynb](https://github.com/WillKoehrsen/Data-Analysis/blob/master/bayesian_lr/Bayesian%20Linear%20Regression%20Demonstration.ipynb)
* [bayes-regression.ipynb](https://github.com/zjost/bayesian-linear-regression/blob/master/src/bayes-regression.ipynb)
* [bayesian-regression-in-pymc3-using-mcmc-variational-inference](https://alexioannides.com/2018/11/07/bayesian-regression-in-pymc3-using-mcmc-variational-inference)


### Logistic regression

What can be demonstrated (theoretical part): consider how the solution changes depending on the values of hyperparameters, visualize.

* [deepreplay](https://github.com/dvgodoy/deepreplay)


### One-Layer Net

What can be demonstrated (theoretical part): [universal function](https://en.wikipedia.org/wiki/UTM_theorem) approximator, overfitting.


### MLP

What can be demonstrated (theoretical part): visualize non-linearity, how the decisive surface changes depending on depth.

* [11_deep_learning.ipynb](https://github.com/ageron/handson-ml/blob/master/11_deep_learning.ipynb)


### Autoencoder

What can be demonstrated (theoretical part): compare with PCA, show compression, denoising, sparsing.

* [AutoEncoders.ipynb](https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/AutoEncoders.ipynb)
* [15_autoencoders.ipynb](https://github.com/ageron/handson-ml/blob/master/15_autoencoders.ipynb)


### CNN

* [Week_04_Convolutional_Neural_Networks.ipynb](https://github.com/DanAnastasyev/DeepNLP-Course/blob/master/Week%2004/Week_04_Convolutional_Neural_Networks.ipynb)
* [lab2](https://github.com/aamini/introtodeeplearning_labs/tree/master/lab2)
* [sst_cnn_classifier.ipynb](https://colab.research.google.com/github/mhagiwara/realworldnlp/blob/master/examples/sentiment/sst_cnn_classifier.ipynb)
* [Sketcher.ipynb](https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/Sketcher.ipynb)
* [13_convolutional_neural_networks.ipynb](https://github.com/ageron/handson-ml/blob/master/13_convolutional_neural_networks.ipynb)


### RNN

What can be demonstrated (theoretical part): vanishing gradients, attention.

* [TextRNN_Torch.ipynb](https://github.com/graykode/nlp-tutorial/blob/master/3-1.TextRNN/TextRNN_Torch.ipynb)
* [TextLSTM_Torch.ipynb](https://github.com/graykode/nlp-tutorial/blob/master/3-2.TextLSTM/TextLSTM_Torch.ipynb)
* [Bi_LSTM_Torch.ipynb](https://github.com/graykode/nlp-tutorial/blob/master/3-3.Bi-LSTM/Bi_LSTM_Torch.ipynb)
* [Seq2Seq_Tensor.ipynb](https://github.com/graykode/nlp-tutorial/blob/master/4-1.Seq2Seq/Seq2Seq_Tensor.ipynb)
* [Seq2Seq(Attention)\_Torch.ipynb](https://github.com/graykode/nlp-tutorial/blob/master/4-2.Seq2Seq(Attention)/Seq2Seq(Attention)\_Torch.ipynb)
* [Bi_LSTM(Attention)\_Torch.ipynb](https://github.com/graykode/nlp-tutorial/blob/master/4-3.Bi-LSTM(Attention)/Bi_LSTM(Attention)\_Torch.ipynb)
* [Week_05_RNNs_Intro.ipynb](https://github.com/DanAnastasyev/DeepNLP-Course/blob/master/Week%2005/Week_05_RNNs_Intro.ipynb)
* [Week_06_RNNs_part_2.ipynb](https://github.com/DanAnastasyev/DeepNLP-Course/blob/master/Week%2006/Week_06_RNNs_part_2.ipynb)
* [Week_09_Seq2seq.ipynb](https://github.com/DanAnastasyev/DeepNLP-Course/blob/master/Week%2009/Week_09_Seq2seq.ipynb)
* [Week_10_Seq2seq_with_Attention.ipynb](https://github.com/DanAnastasyev/DeepNLP-Course/blob/master/Week%2010/Week_10_Seq2seq_with_Attention.ipynb)
* [lab1](https://github.com/aamini/introtodeeplearning_labs/tree/master/lab1)
* [week4-seq2seq.ipynb](https://github.com/hse-aml/natural-language-processing/blob/master/week4/week4-seq2seq.ipynb)
* [sst_classifier.ipynb](https://colab.research.google.com/github/mhagiwara/realworldnlp/blob/master/examples/sentiment/sst_classifier.ipynb)
* [14_recurrent_neural_networks.ipynb](https://github.com/ageron/handson-ml/blob/master/14_recurrent_neural_networks.ipynb)


### ResNets


### Embeddings

* [1-2.Word2Vec](https://github.com/graykode/nlp-tutorial/tree/master/1-2.Word2Vec)
* [1-3.FastText](https://github.com/graykode/nlp-tutorial/blob/master/1-3.FastText)
* [Week_02_Word_Embeddings_(Part_1).ipynb](https://github.com/DanAnastasyev/DeepNLP-Course/blob/master/Week%2002/Week_02_Word_Embeddings_(Part_1).ipynb)
* [Week_03_Word_Embeddings_(Part_2).ipynb](https://github.com/DanAnastasyev/DeepNLP-Course/blob/master/Week%2003/Week_03_Word_Embeddings_(Part_2).ipynb)
* [week3-Embeddings.ipynb](https://github.com/hse-aml/natural-language-processing/blob/master/week3/week3-Embeddings.ipynb)


### Variational Models


### GAN

* [tutorial_gans.ipynb](https://github.com/lexfridman/mit-deep-learning/blob/master/tutorial_gans/tutorial_gans.ipynb)
* [BigGAN.ipynb](https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/BigGAN.ipynb)
* [BigGanEx.ipynb](https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/BigGanEx.ipynb)
* [SC_FEGAN.ipynb](https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/SC_FEGAN.ipynb)


### T2T

* [predicting_movie_reviews_with_bert_on_tf_hub.ipynb](https://colab.research.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb)
* [bertviz](https://github.com/jessevig/bertviz)
* [Transformer(Greedy_decoder)\_Torch.ipynb](https://github.com/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer(Greedy_decoder)\_Torch.ipynb)
* [BERT_Torch.ipynb](https://github.com/graykode/nlp-tutorial/blob/master/5-2.BERT/BERT_Torch.ipynb)
* [Week_11_Transformers.ipynb](https://github.com/DanAnastasyev/DeepNLP-Course/blob/master/Week%2011/Week_11_Transformers.ipynb)
