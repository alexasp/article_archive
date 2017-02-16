#Project presentations

##Topic Classification of Journalistic Content.
https://drive.google.com/file/d/0BxQkKe29iv9SNTBRcEtlQTRobEE

###Takeaways: 
Framework keras uses tensorflow or theano, and makes setup of convolutional neural networks and possibly other models simpler.
Batch normalization can be a very good idea (for faster convergence).
Pre-trained word embeddings isn't always useful. In this case it decreased performance.
Had success speeding up training with a narrower CNN, using only one filter size.

#Optimization
##Downpour SGD
http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf

#Model selection

##A Study on Threshold Selection for Multi-label Classification
https://pdfs.semanticscholar.org/f3eb/f945aba8d70b8d7daf14021fe1220752f0f7.pdf


##On Over-fitting in Model Selection and Subsequent Selection Bias in
Performance Evaluation
http://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf

Over-fitting at the second level of inference (i.e., model selection) can have a very substantial damaging effect on the generalisation performance of
state-of-the-art machine learning algorithms.

##ROC Curve shape
http://stats.stackexchange.com/questions/111577/roc-graph-shape

..

#Neural networks
##General

Backpropegation updates is computed simply with recursive application of differentiation chain rule.

Deep networks can be made deeper than what can practically be trained by backpropagation by fully traning parts of the network as a classifier, then using the high level features learned by the network as input to train a new neural network - stacking neural networks.

Take advantage of dataset with multiple classes. If the data is from the same distribution, there are general concepts to be learned that will help support classification, and neural networks need a lot of data to be able to do this -> train multi-class networks.

### Neural Networks, Manifolds, and Topology
http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/


##Recurrent Neural Networks readup
http://karpathy.github.io/2015/05/21/rnn-effectiveness/ (some pretty cool examples here)
https://deeplearning4j.org/lstm.html
https://arxiv.org/pdf/1602.07776v3.pdf

Sequential processing is often possible even when it is not obvious such as fixed matrix input. E.g. a number can be read from left to right, and a house number can be formed by applying colors in sequence.

RNNs can easily be stacked deeply, e.g. y1 = rnn1.step(x), y = rnn2.step(y1) and so on. The second layer takes output of first RNN layer as input instead of data as input.

####Bidirectional RNN
Sometimes we require too long memory for normal RNNs. Use a bi-directional RNN instead to remember longer.

####LSTMs
RNNs suffer from vanishing gradients, and Long Short-Term Memory (LSTM) units were proposed to handle this. LSTMs are like storage cells, where input is allowed, current state is forgotten and output of state is allowed depending on the current input. It works like a controlled memory that can operate at various timescales.



##Understanding 
al neural networks for NLP
http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/

###Takeaways:
Convolutional NNs are not intuitively well fit for language, but they are very fast and work quite well on NLP problems. N-grams can be very expensive on large vocabularies, but wide filters in CNNs is no problem.

Word embeddings may be more beneficial when applied to shorter documents than longer documents.
####Parameters 
Filter size, filter stride length, number of filters, padding of input or not (narrow vs wide convolution), pooling strategy. A paper is referenced in the introduction that experiements with parameter variations, which could be useful if using CNNs.
####Model properties

A CNN can take multiple input channels (e.g. RGB in images, different type of word embeddings in NLP)

Pooling layers are used to ensure fixed size output matrix as well as reducing dimensionality. (And in images some translation and rotation invariance)

Suitable for classification tasks (sentiment, spam, topic), not so suitable for sentence tagging problems since convolutions and pooling looses word order.

This document uses concatenated word2vec embeddings as input layer, but mentiones that this is not well enough investigated. Maybe some embeddings are better, or they can be learned as part of the model.

####Character level CNN

Character information could also be used as input to the network, either iwht embedding or learning directly from character level input.

##Neural Turing Machines
https://arxiv.org/abs/1410.5401

...

#Generative models

##Generative Adverserial Nets.
https://drive.google.com/open?id=0BxQkKe29iv9SajZHY25sdmZyR1U

###Takeaways:
Deep neural nets have had sucess in discriminative learning, but generative models are expensive to train. Some approaches use approximations or limit the model, but adverserial neural nets uses two neural nets "in competition", where generative model G learns to map samples from random noise Z to data X, while discriminative model D tries to tell if data points is output from G or a sample of training data. The models are optimized in turn.

A tricky thing with this kind of model is balancing how long each model can train before switching to the other one, as the models can get stuck in bad states.

##Adverserial Autocoders 
https://drive.google.com/open?id=0BxQkKe29iv9SdTFSWmdnYTM5cVE
###Takeaways: 
Inspired by Adverserial Nets.

Uses normal autoencoders, but imposes a predefined distribution on samples in encoded space. This distribution is specified such that we can easily sample from encoded space and feed results through the decoder, yielding output that looks like training data.

By varying how we interact with the encoded space, we can do unsupervised learning, semi-supervised learning, cool tricks like separating style and class, clustering - and probably several other minor fun tricks. 


#Misc. models
##Softmax classifier
http://cs231n.github.io/linear-classify/#softmax
Generaliation of logistic regression to multiple classes. Uses a cross-entropy loss, where we minimize the cross-entropy between the predicted class probability distribution and the true dist (all prob. mass on the correct class).
