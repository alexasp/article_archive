#Project presentations

##Topic Classification of Journalistic Content.
https://drive.google.com/file/d/0BxQkKe29iv9SNTBRcEtlQTRobEE

###Takeaways: 
Framework keras uses tensorflow or theano, and makes setup of convolutional neural networks and possibly other models simpler.
Batch normalization can be a very good idea (for faster convergence).
Pre-trained word embeddings isn't always useful. In this case it decreased performance.

#Neural networks

##Understanding convolutional neural networks for NLP
http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/

###Takeaways:


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

