# Template for evaluating and grading projects

**Evaluation of project number: 1**

**
Name: Thomas Larsen Greiner and Hao Zhao**


## 
Abstract
*Abstract: accurate and informative? 
Total number of possible points: 5*

Mark and comments: 5
Good.


## Introduction
*
Introduction: status of problem and the major objectives. 
Total number of
possible points: 10*

Mark and comments: 9

Good. 
The introduction seems a bit to similar to the abstract in that parts of
it reads like a table of contents. It is not necessary to go into details on the
methods at this point, but rather focus on motivating the reader to actually
read on. It could be longer if need be. 

The section describing the structure of the report is excellent. 


## Formalism
*Formalism/methods: Discussion of the methods used and their basis/suitability.
Total number of possible points 20*

Mark and comments: 18
Excellent.
Only thing missing is an approach to get confidence intervals for Ridge and
Lasso. The direct formula stated is true for OLS, but is not the same for the
other two. A numerical estimation via resampling is needed.

## Code
*Code/Implementations/test: Readability of code, implementation, testing and
discussion of benchmarks. Total number of possible points 20*

Mark and comments: 18
Code is readable, and through the results seems to be sufficiently tested.

In general though, there is a lot of linear code in the files. By that I mean
there are large amounts of global scope code that gets run in sequence. To your
credit you have made a valiant effort to make this readable. However, the reader
of your code is forced to jump up and down large portions of code in order to
see where and how a certain variable was defined 100 lines before. Try to
separate things into functions more, and use several script files to split
things up more.

Also, see above, code for confidence intervals does not cover the case of Ridge and
Lasso.

## Analysis
*Analysis: of results and the effectiveness of their selection and presentation.
Are the results well understood and discussed? Total number of possible points:
20*

Mark and comments: 20
Excellent. The presentation is very well done.


## Conclusions
*Conclusions, discussions and critical comments: on what was learned about the
method used and on the results obtained. Possible directions and future
improvements? Total number of possible points: 10*

Mark and comments: 9

Good, albeit a bit short. Recap some of the insights you gained, as well as any
potential ideas for how to improve results (this is mentioned briefly in the
results section).


## Overall presentation:
*Clarity of figures and overall presentation. Too much or too little? Total number of possible points: 10*

Mark and comments: 10
Excellent once again. Figures are well documented and can be looked at without
reading everything in the text. Good!


## Referencing
*Referencing: relevant works cited accurately? Total number of possible points 5*

Mark and comments: 5


## Overall
*Overall mark (%) and final possible final comments*

94/100

A very good job to you both. A pleasure to read, please keep this up.
