## Paper

* Title: Dropout: A Simple Way to Prevent Neural Networks from Overfitting
* Authors: Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov 
* Link: http://jmlr.org/papers/v15/srivastava14a.html

Abstract:

> Deep neural nets with a large number of parameters are very powerful machine
> learning systems. However, overfitting is a serious problem in such networks.
> Large networks are also slow to use, making it difficult to deal with
> overfitting by combining the predictions of many different large neural nets at
> test time. Dropout is a technique for addressing this problem. The key idea is
> to randomly drop units (along with their connections) from the neural network
> during training. This prevents units from co-adapting too much. During
> training, dropout samples from an exponential number of different “thinned”
> networks. At test time, it is easy to approximate the effect of averaging the
> predictions of all these thinned networks by simply using a single unthinned
> network that has smaller weights. This significantly reduces overfitting and
> gives major improvements over other regularization methods. We show that
> dropout improves the performance of neural networks on supervised learning
> tasks in vision, speech recognition, document classification and computational
> biology, obtaining state-of-the-art results on many benchmark data sets.

This part of section 2 "Motivation" is just such a lucid and interesting
analogy that I'll quote it here verbatim:

> A motivation for dropout comes from a theory of the role of sex in evolution
> (Livnat et al., 2010). Sexual reproduction involves taking half the genes of
> one parent and half of the other, adding a very small amount of random
> mutation, and combining them to produce an offspring. The asexual alternative
> is to create an offspring with a slightly mutated copy of the parent’s genes.
> It seems plausible that asexual reproduction should be a better way to
> optimize individual fitness because a good set of genes that have come to
> work well together can be passed on directly to the offspring. On the other
> hand, sexual reproduction is likely to break up these co-adapted sets of
> genes, especially if these sets are large and, intuitively, this should
> decrease the fitness of organisms that have already evolved complicated
> coadaptations.  However, sexual reproduction is the way most advanced
> organisms evolved.

> One possible explanation for the superiority of sexual reproduction is that,
> over the long term, the criterion for natural selection may not be individual
> fitness but rather mix-ability of genes. The ability of a set of genes to be
> able to work well with another random set of genes makes them more robust.
> Since a gene cannot rely on a large set of partners to be present at all
> times, it must learn to do something useful on its own or in collaboration
> with a small number of other genes. According to this theory, the role of
> sexual reproduction is not just to allow useful new genes to spread
> throughout the population, but also to facilitate this process by reducing
> complex co-adaptations that would reduce the chance of a new gene improving
> the fitness of an individual. Similarly, each hidden unit in a neural network
> trained with dropout must learn to work with a randomly chosen sample of
> other units. This should make each hidden unit more robust and drive it
> towards creating useful features on its own without relying on other hidden
> units to correct its mistakes. 


## ⁉️ Big Question / TLDR

Primarily, the paper is looking to evaluate Dropout as a method of reducing
overfitting and improving model generalization.

Dropout is a regularization technique - a family of techniques for reducing
overfitting (thereby improving generalization) by making the decision boundary
or fitted model smoother.

The most widely used implementation of Dropout is as a _stochastic_
regularization technique, and that is the implementation primarily tested in
this paper.

In addition, the authors find that there are additional improvements whereby
neuron co-adaptation is reduced and feature (representational) sparsity is
improved.


## Overall impression

A very solid paper. It gives easy-to-use practical recommendations for
using dropout.  It shows strong empirical results in favor of dropout.
It shares an interesting motivating idea from a different field.
It does lack a theoretical underpinning for dropout. It's possible
that such an underpinning can be found from
["Uncertainty In Deep Learning", Gal 2016](http://mlg.eng.cam.ac.uk/yarin/blog_2248.html)
which ties "approximate [Bayesian Neural Network] inference together with (...)
regularization techniques such as dropout"

The paper introduced me to a wide range of background material.


## Practical takeaways

### Takeaways: Dropout and its relation to architecture and training

Appendix A is titled "A Practical Guide for Training Dropout Networks" and
makes a few recommendations:

* Dropout rate: On real-valued input layers (images, speech), drop out 20%. For internal
  hidden layers, drop out 50%.
* Layer size: If you think a layer size of N is appropriate for your problem
  (e.g. an internal hidden layer of N=128 nodes) and want to us dropout rate P
  (say p=0.5 for 50% dropout), resize that layer to N/P (e.g. double it to
  128/0.5=256 nodes) to keep the same representational power.
* Optimizer: When using dropout, use 10-100 times the learning rate. Also bump
  up your momentum from 0.9 (typical rate without dropout) to 0.95-0.99. (Sec A.2)
* Regularization: Adding max-norm regularization works well, since you are
  increasing momentum and learning rate.

### Takeaways: Using Dropout with CNNs and RNNs

Why do we not see dropout in modern CNN networks -- e.g. fully convolutional
residual networks with batch normalization?

> I think dropout is still very useful when fitting complex supervised models
> on tasks with few labeled samples. However training deep supervised vision
> models is nowadays done with residual convolutional architectures that make
> intensive use of batch normalization. The slight stochastic rescaling and
> shifting of BN also serve as a stochastic regularizer and can make dropout
> regularization redundant. Furthermore dropout never really helped when
> inserted between convolution layers and was most useful between fully
> connected layers. Today's most performant vision models don't use fully
> connected layers anymore (they use convolutional blocks till the end and then
> some parameterless global averaging layer).

-- `/u/ogrisel` in [`/r/MachineLearning` "What happened to DropOut?"](https://www.reddit.com/r/MachineLearning/comments/5l3f1c)

Perhaps the structural correlation in CNNs (across spatial dimensions) and RNNs
(across time) means that dropout actually hurts more than it helps, unless
you take care to tie dropout weights to avoid this:

> Dropout does actually work quite well between recurrent units if you tie the
> dropout masks across time.

_(Blog post author note: See [Gal and Ghahramani "A Theoretically Grounded
Application of Dropout in Recurrent Neural
Networks"](https://arxiv.org/abs/1512.05287).) Continuing the same quote from
above:_

> The same actually holds for dropout between convolutional layers: if you tie
> the masks across spatial dimensions it regularises pretty well (too strongly
> in fact, in my experience). But since conv. layers have relatively few
> parameters anyway there's not so much of a need, and of course the
> regularising side-effects of batchnorm also help.
>
> The reason dropout doesn't work between conv. layers if you don't do this, is
> because the weight gradients of the conv. layers are averaged across the
> spatial dimensions, which tend to contain many very highly correlated
> activations. This ends up canceling out the effect of dropout if the mask is
> different at every spatial location. You're basically monte-carlo averaging
> over all possible masks, which is precisely what you don't want to do with
> dropout during training :)

-- `/u/benanne` in [`/r/MachineLearning` "What happened to DropOut?"](https://www.reddit.com/r/MachineLearning/comments/5l3f1c)

### Takeaways: Using Dropout to measure uncertainty

There is very interesting research by [Yarin
Gal](http://mlg.eng.cam.ac.uk/yarin/) that builds on dropout to produce
uncertainty measures (error bars) from neural networks. Very loosely, at
prediction time you use Monte Carlo dropout (predict from many thinned
networks), take the average of all the predictions as your prediction and the
_variance_ of all the predictions as the uncertainty.

Some excerpts I really enjoyed from Gal's
["What My Deep Model Doesn't Know..."](http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html) --
absolutely worth a read. Gal's writing is approachable and lucid.
Interactive figures in the webpages help reinforce the concepts.

> Understanding if your model is under-confident or falsely over-confident can
> help you get better performance out of it. We'll see below some simplified
> scenarios where model uncertainty increases far from the data. Recognising
> that your test data is far from your training data you could easily augment
> the training data accordingly.

> Using softmax to get probabilities is actually not enough to obtain model uncertainty.

> As you'd expect, it is as easy to implement these two equations. We can use
> the following few lines of Python code to get the predictive mean and
> uncertainty.

> At the moment it is thought in the field that dropout works because of the
> noise it introduces. I would say that the opposite is true: dropout works
> despite the noise it introduces!

## 🏙 Background Summary

> What work has been done before in this field to answer the big question? What
> are the limitations of that work? What, according to the authors, needs to be
> done next?

Prior methods of reducing overfitting:

* Early stopping
* Regularization: L2 weight penalization, lasso, KL-sparsity, max-norm
* Adding noise during the training process to make the predictor more robust -
  notably denoising autoencoders (DAEs).

Ensembling:

It is appealing to make an analogy between dropout and ensembling.  It seems
that Monte Carlo dropout is truly ensembling, and the weight scaling
approximation described is an approximation to this.

Further reading in
["Analysis of dropout learning regarded as ensemble learning", Hara et al 2017](https://arxiv.org/pdf/1706.06859.pdf)

Bayesian Neural Networks:

> In dropout, each model is weighted equally, whereas in a Bayesian neural
> network each model is weighted taking into account the prior and how well the
> model fits the data, which is the more correct approach. Bayesian neural nets
> are extremely useful for solving problems in domains where data is scarce
> such as medical diagnosis, genetics, drug discovery and other computational
> biology applications. However, Bayesian neural nets are slow to train and
> difficult to scale to very large network sizes.

I am keenly interested in reading more about Bayesian Deep Learning,
perhaps starting with [Yarin Gal's thesis "Uncertainty in Deep Learning"](http://mlg.eng.cam.ac.uk/yarin/blog_2248.html),
http://bayesiandeeplearning.org/ and/or http://www.gaussianprocess.org/gpml/chapters/

## ❓ Specific question(s)

* Q: Does dropout reduce overfitting and improve generalization?
* Q: Is weight scaling a sufficient approximation for full monte carlo dropout?
* Q: How does the accuracy of scaling dropout compare to that of Bayesian Neural Networks?
* Q: How does dropout perform compared to other regularization techniques (L2, L1, KL-sparsity, max-norm)?
* Q: What is the effect on feature sparsity?


## 💭 Approach

What are the authors going to do to answer the specific question(s)?

* Try a bunch of models on benchmark datasets with and without dropout.
* Measure weight activations and produce histograms to measure the effect on
  sparsity.
* Compare the weight scaling approximation technique to the Monte Carlo
  sampling technique for prediction to see how much accuracy is traded at what
  performance cost.

### Methods: Sparisty


## 📓 Results

### Results: Generalization

For generalization, the authors compared networks with and without dropout on
the following datasets, observing that "dropout improved generalization
performance on all data sets compared to neural networks that did not use
dropout":

> * MNIST : A standard toy data set of handwritten digits.
> * TIMIT : A standard speech benchmark for clean speech recognition.
> * CIFAR-10 and CIFAR-100 : Tiny natural images (Krizhevsky, 2009).
> * Street View House Numbers data set (SVHN) : Images of house numbers collected by Google Street View (Netzer et al., 2011).
> * ImageNet : A large collection of natural images.
> * Reuters-RCV1 : A collection of Reuters newswire articles.
> * Alternative Splicing data set: RNA features for predicting alternative gene splicing (Xiong et al., 2011).

### Results: Sparsity

A sparse feature is a feature that that has mostly zero values like a one-hot
encoding of categorization or a TFIDF encoding.  Think of them in contrast to
dense features - some examples of which are image data, audio data, and word
embedding vectors.  Sparse features _should_ be more interpretable, as
individual neurons will be activated (or individual dimensions given a high
value) which correspond to concepts. Think of this like NLP networks with a
"sentiment neuron", or the output of VGG where each softmax dimension is a
distinct class.

The authors assert that:

> In a good sparse model, there should only be a few highly activated units for
> any data case. Moreover, the average activation of any unit across data cases
> should be low.

The authors produce a histogram of activation weights as well as a histogram of
the mean activation weights across data samples.

For sparsity, indeed, we can see that dropout pushes the distribution of
activations to skew heavily toward zeroes with very few units showing high
activation. The mean activations are lowered from 2.0 to 0.7:

![](https://i.imgur.com/1dh0fYR.png)

From the paper:

> [Examining] features learned by an autoencoder on MNIST (...) each hidden
> unit [without dropout] on its own does not seem to be detecting a meaningful
> feature. On the other hand, [with dropout], the hidden units seem to detect
> edges, strokes and spots in different parts of the image. This shows that
> dropout does break up co-adaptations, which is probably the main reason why
> it leads to lower generalization errors.

### Results: Comparison to other regularization methods

Here is Table 9 from the paper with results:

![](https://i.imgur.com/Od7wg8Z.png)

### Results: Prediction: Weight scaling and Monte Carlo sampling

The authors find that the weight scaling approximation method is a
good approximation of the true model average:

![](https://i.imgur.com/Fcali6D.png)

> We (...) do classification by averaging the predictions of k randomly sampled
> neural networks. (...) It can be seen that around k = 50, the Monte-Carlo
> method becomes as good as the approximate method. Thereafter, the Monte-Carlo
> method is slightly better than the approximate method but well within one
> standard deviation of it.  This suggests that the weight scaling method is a
> fairly good approximation of the true model average.


### Results: Comparison to Bayesian Neural Networks

It is interesting to me that the authors find that Bayesian Neural Networks
yield beter performance than dropout networks at reducing overfitting in small
dataset regimes, at the cost of additional computation. I was not previously
aware of Bayesian Neural Networks and this does pique my interest in learning
more:

> Xiong et al. (2011) used Bayesian neural nets for this task. As expected, we
> found that Bayesian neural nets perform better than dropout. However, we see
> that dropout improves significantly upon the performance of standard neural
> nets and outperforms all other methods. The challenge in this data set is to
> prevent overfitting since the size of the training set is small.

The scaling approximation not as good as BNN, but for the one problem they
examine (predicting the occurrence of alternative splicing based on RNA
features), it's better than other approaches (early stopping, regression with
PCA, SVM with PCA).


## Words I don't know

I encountered some new terminology in this paper.

### Normal distribution notation

![](https://i.imgur.com/cshrQzW.png)

I learned that the notation `N(mu,sigma)` signifies the [Normal
distribution](https://en.wikipedia.org/wiki/Normal_distribution) (aka the Gaussian distribution)
with mean `mu` and standard deviation `sigma`.

### Marginalization 

I still don't quite grasp how this applies to dropout. Marginalization appears
to be defined as removing a subset of variables (or dimensions) by summing over
the distributions of the other variables.

See
the [Wikipedia page for marginal distribution](https://en.wikipedia.org/wiki/Marginal_distribution)
and [this Quora answer](https://www.quora.com/What-is-marginalization-in-probability):

> In short, marginalization is how to safely ignore variables.
> 
> Let's assume we have 2 variables, A and B. If we know P(A=a,B=b)
> for all possible values of a and b, we can calculate
> P(B=b) as the sum over all a of P(A=a,B=b). Here we "marginalized out" the
> variable A.

From section 9 of the paper:

> Dropout can be seen as a way of adding noise to the states of hidden units in
> a neural network. In this section, we explore the class of models that arise
> as a result of marginalizing this noise. These models can be seen as
> deterministic versions of dropout.

The authors go on to make an analogy to ridge regression - but ridge regression
will regularize weights and will rarely select them out entirely. (Unlike lasso,
which will more frequently choose coefficients of zero and thus select variables
out entirely. See ISLR 6.2 "The Variable Selection Property of the Lasso")
I don't then understand an intuitive relation between ridge regression and dropout.
I don't quite grasp this section 9 at all. The authors go on to say:

> However, the assumptions involved in this technique become successively weaker as more
> layers are added. Therefore, the results are not directly applicable to deep networks.

### Restricted Boltzmann Machines (RBMs)

The paper devotes entire sections to examining dropout on RBMs, which I had
heard about but never encountered before.

I briefly read the
[DeepLearning4J article on RBMs](https://deeplearning4j.org/restrictedboltzmannmachine)
and don't yet completely understand them. What I understand so far:

RBMs can be trained in an unsupervised fashion to perform something called
reconstruction, which is different from classification or regression, and which
finds the joint probability distribution of the inputs and the outputs. In this
case, a forward pass is performed from inputs to activations, and then a
backward pass is performed from those activations, through the network weights,
back to a "reconstruction" output. The distribution of this reconstruction is
compared to the distribution of the input with KL divergence.  The weights are
tuned to minimize this divergence, so the RBM learns to "reflect the structure
of the input, which is encoded in the activations of the first hidden layer."

> Reconstruction does something different from regression, which estimates a
> continous value based on many inputs, and different from classification,
> which makes guesses about which discrete label to apply to a given input
> example.


### Broadcasting

Consider that matrix multiplication is defined for compatible dimensionality;
e.g. we know how to multiply an AxB matrix by a second BxA matrix. Broadcasting defines
a similar operation with looser constraints on the shape of the operands.


```python
m = np.array([[1, 2, 3], [4,5,6], [7,8,9]])
#
# array([[1, 2, 3],
#        [4, 5, 6],
#        [7, 8, 9]])

c = np.array([10,20,30])
#
# array([10, 20, 30])

m + c
#
# array([[11, 22, 33],
#        [14, 25, 36],
#        [17, 28, 39]])
```

> When operating on two arrays, Numpy/PyTorch compares their shapes
> element-wise. It starts with the trailing dimensions, and works its way
> forward. Two dimensions are compatible when they are equal, or one of them is 1

Read more in Rachel Thomas' [All the Linear Algebra You Need for AI](https://github.com/fastai/fastai/blob/master/tutorials/linalg_pytorch.ipynb)


### KL sparsity

> KL-sparsity regularization, which minimizes the KL-divergence between the
> distribution of hidden unit activations and a target Bernoulli distribution.
["Improving Neural Networks with Dropout", Srivastava, 2013](http://www.cs.toronto.edu/~nitish/msc_thesis.pdf)


## Followup questions

Q: How does the breakup of coadaptation and increase in feature sparsity relate
to Hinton's capsule networks?  See forthcoming ["Dynamic Routing between
Capsules" at NIPS 2017](https://research.google.com/pubs/pub46351.html).
