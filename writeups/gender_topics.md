# A look at our gender topics

It's about time to start writing up and sharing some first visualizations and results. 
Starting with a dive into the gender topics and associated terms seems like a good starting
point.

Note: the following analysis uses exclusively the journals dataset.

## The "Women's and Gender History" General Approach
As a way of grouping topics together, we have given each topic one or multiple general approaches
where appropriate. For example, the "Markets and Trade" topic falls under the general approach
"Economic History." The goal of this approach is to show both very broad trends 
and developments within a field, say from social history inspired agricultural history to
the current history of capitalism. 

"Women's and Gender History" is one of these general approaches. We have assigned this general
approach to four topics: Gender and Feminism, Family, Sexuality, and Consumption and Consumerism.
Sexuality is also labeled as Cultural History, for the other topics "Women's and Gender History" 
is the only general approach.

Did we get this labeling right? 

One way of checking is to compare all articles that mention "women" to all those that don't use
the word "women." It's an almost even split--5391 mention women, 4976 don't.

Now, we can look at what topics are overrepresented in the "mentions women" subcorpus, which gives
us as the top 6:

- Gender and Feminism
- Family
- Doctors and Patients 
- Consumption and Consumerism
- Sexuality
- Sports and Class in Great Britain

``` 
topic                                                         dunning    frequency_score    weight both  w women     w not women
-----------------------------------------------------------  ---------  -----------------  -----------  ----------  -------------
(61) Gender and Feminism                                     549241              0.980275   0.0117364   0.0221578     0.000445864
(46) Family                                                  435844              0.892799   0.0155799   0.0269713     0.00323854
(32) Doctors and Patients                                    157285              0.854878   0.00707363  0.0117601     0.00199636
(76) Consumption and Consumerism                             157618              0.843185   0.00763182  0.0125259     0.00232956
(71) Sexuality                                               140564              0.815808   0.00815038  0.0129704     0.00292843
(89) Sports and Class in Great Britain                       102086              0.795186   0.00683594  0.0106207     0.00273554
(84) Noise                                                    15475.2            0.752392   0.00143872  0.00212208    0.000698365
(54) Witchcraft and Magic                                     52775.8            0.748007   0.00508792  0.0074634     0.00251432
(68) 19th Century African American History                    79678.8            0.741623   0.00810721  0.0117968     0.00410993
(50) 20th Century Labor History                              118989              0.740707   0.0122023   0.0177349     0.0062083
```

I'll write a lot more about these distinctiveness comparisons in another writeup. For now, 
you can have a look at what Dunning and Frequency Score mean 
[here](https://github.com/srisi/gender_history/blob/699b9fa8de9490e84e294ce9c88b51d98c0da4d0/divergence_analysis.md).
Weight both means the average weight of the topic in the whole corpus. "w women" is the average 
weight of the topic in articles that mention women, "w not women" is the average weight of the topic
in articles that don't mention women.

At any rate, our classification looks about right. Only "Doctors and Patients" comes between our four
selected topics. It might be potentially be worth to include that in the gender topics. And the
same holds potentially for "Sports and Class in Great Britain."

## The history of Women's and Gender History
What kind of history does the general "Women's and Gender History" approach have in our journal 
dataset? Let's have a first look.

![](https://github.com/srisi/gender_history/raw/refactor/visualizations/topic_frequency_plots/Women's%20and%20Gender%20History.png)


The overall approach weight is the sum of the weights of the individual topics. At its peak around
2000, women's and gender history made up about 6% of total weight. 

The color of the graph indicates how male or female dominated a topic is at a given time. Red means
only women write about a topic while blue mens only men write about a topic. Obviously, most values
are somewhere in between. 

The math behind this frequency score is pretty simple:
`frequency score = avg_topic_weight(women) / (avg_topic_weight(women) + avg_topic_weight(men)`
For example, among women the "Gender and Feminism" topic has an average weight of 3.9%. Among men,
it has an average weight of 0.49%. That gives it a frequency score of 0.89. Frequency scores are
always between 0 and 1.

Before we dive into individual topics, it's worth observing that women's and gender history (as
seen through these four topics) already starts to rise in the 1970s. It's just that the "Family" 
topic contributes most of that early rise while gender and feminism follows about one decade later.

## Family

Let's have a look at the family topic now.

![](https://github.com/srisi/gender_history/raw/refactor/visualizations/topic_frequency_plots/46_Family.png)

For these visualizations of individual topics, I have selected six representative terms per topic,
in this case Family, Children, Women, Marriage, Household, and Parents. So the smaller graphs
no longer represent the history of topics; they now show the histories of terms.

This is a mix between the terms_prob from the topic and the terms that are most distinctive for the
topic by Dunning score.<br>
Terms prob: famili, children, marriag, household, marri, parent, child, mother, women<br>
Dunning terms: family, children, women, marriage, child, families, married, household

I did skip very similar terms (family and families, child and children). Apart from that seemed
like a reasonable representation of the two lists in six terms. For a more rigorous approach, I 
could also use just one or the other method. Thoughts?

At any rate, the terms seem to show two slightly different trends: family, children, and marriage 
all peak around 1980 while children and parents only peak around 2000. This might point to a later
cultural history of childhood.

## Gender and Feminism

The gender and feminism topic follows family by about one decade:

![](https://github.com/srisi/gender_history/raw/refactor/visualizations/topic_frequency_plots/61_Gender%20and%20Feminism.png)

Noteworthy here is the term gender, which was barely used in 1980 but became prominent by the 1990s.

The initial work in this area was almost exclusively done by women and even in the 2000s, the line
is only slightly less than dark red (I can dig into the numbers for this).

## Sexuality

The sexuality topic is strikingly different.

![](https://github.com/srisi/gender_history/raw/refactor/visualizations/topic_frequency_plots/71_Sexuality.png)

Most importantly, it is bimodal with one peak in the 1970s and another one in the 2000s. The 1970s
peak seems quite clearly a Freud peak and work on Freud, at least in the 1970s, was predominantly
done by men. The second peak in the 2000s, characterized by terms like sexuality, love, or emotional, 
represents work done mostly by women. Though if we dig into this further, I'm sure we'll find a strong
gay history topic here as well.

## Consumption and Consumerism

Finally, consumption and consumerism, which is still on the rise.

![](https://github.com/srisi/gender_history/raw/refactor/visualizations/topic_frequency_plots/76_Consumption%20and%20consumerism.png)

It's noteworthy here that more terms start to turn gray, i.e. mostly gender neutral. For the final
visualization, I'll need to find a way to make these terms a darker shade of gray to prevent the graph
from looking washed out.

I also don't know what to make of the peak with food around 1980. Maybe an earlier social history?
Maybe agriculture related?



