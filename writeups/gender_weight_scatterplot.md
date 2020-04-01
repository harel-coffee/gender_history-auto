# Gender-Weight Scatterplots

How can we present a summary of our topics in a single chart? (And what would we want to show?)

What we'd want to show are: <br>
a) how gendered a topic is<br>
b) how much overall weight a topic has<br>
c) when its importance rises or falls.

Here's one attempt of showing these three dimensions in a single chart:
![](https://github.com/srisi/gender_history/raw/refactor/visualizations/gender_frequency_scatterplots/gender_time_weight.jpg)

## So many colors... What am I looking at?

There's a lot going on in this graph. Let's start with the easiest part:

### Topic Weight (Y-axis):
The y-axis shows the mean weight of a topic across all 10000 articles. 
The largest topic, "General Historiography" has an overall weight of about 6.5%. "European 
Colonization of Asia," one of the smaller topics in the bottom right, has a weight of about 0.4%.
Note that this axis is on a log scale.

### Frequency Score / Gender Distribution (X-axis)
The x-axis tells us how gendered a topic is from
0 (only men write about a topic) to 1 (only women write about a topic).<br>
The math behind this score is simple:<br>
`frequency score(topic) = weight of topic among women / (weight of topic among women + weight of 
topic among men)`<br>
Here's an explanation that's more intuitive: Assume that we have an equal number of articles written
by men and women (we don't but let's presume that's the case for the moment). In that case, a
frequency score of 0.5 means that women contributed 50% of the weight to the topic and men the
remaining 50%. A score of 0.89 (the score of "Gender and Feminism") means that women contributed
90% of the weight and men 10%. <br>
The reality is more complicated simply because we have about three times as many articles authored
by men as articles authored by women (7506 to 2016). Hence, men did contribute about one third of 
the "Gender and Feminism" topic weight (37 to 79), simply because even when they are not gender
historians, they will occasionally use "Gender and Feminism" terms like "women," "male," or 
"female." <br>
If we look at the average weights, however, we see that among women, "Gender and 
Feminism" has an average weight of 3.93% whereas among men, it has an average weight of 0.49%. This
is the relationship that the frequency score expresses: 3.93% / (3.93% + 0.49%) = 0.89.

### Median Publication Year (dot color). 
Notice that the dots have different colors from dark blue to 
dark red. These colors show the median publication year for a given topic.<br>
What does that mean? Across all articles from 1951 to 2014, every topic has a certain total weight.
For "Gender and Feminism," that total weight is 116. The median publication year indicates the year
that separates this weight into two, i.e. half of the weight happened before or during that year
and the other half happened during or after that year. <br>
In the case of "Gender and Feminism," we can see, for example, that the median publication year was
sometime in the 1990s. <br>
From the colors, we can see some topics that peaked in the late 1960s or
 early 1970s like "German and Austro-Hungarian 
Diplomatic History" (I know, shocker!) or "American Political Thought" as well as some important 
recent topics like "Nations and Boundaries," a kind of cultural history of how nation states manage
their real and imagined borders, which peaked in 2007 in our dataset. <br>
Note: I don't like the color scheme. For a next iteration of this chart, I think I will just select
a color for each decade from the 1960s to the 2000s. I think that will make the graph look a lot
better.

## Ok, so what do we see?

I think the chart is useful because it captures some of the general trends that we discuss: 
work on "Gender and Feminism," "Family," "Doctors and Patients", "Consumption and Consumerism" is 
predominantly done by women. It is also worth noting that there is no male equivalent to 
"Gender and Feminism," i.e. a topic for which 80% or more of the work is done by men. So one very
simple answer to the question: What would we not know about if it weren't for women? is histories
that touch on gender and feminism.

## Date and gender correlation

There is something problematic about our general approach, though. Note that the most male-
dominated topics are also the most outdated ones: "German and Austro-Hungarian Diplomatic 
History" or "American Political Thought." An uncharitable reading of this might be that men are 
just doing boring, outdated work. 

However, that's not correct. Men seem to be working on "outdated" topics simply because men 
outnumbered women far more in the 1950s, 60s, and 70s than they do today. In the 1950s, men 
outnumber women in our dataset almost 10 to 1 (580 to 62). For the 2010s, this ratio has dropped
to about 1.7 to 1 (409 to 242). 

Another way of looking at this skew is that the average article published by a man came out in 
the early 1980s whereas the average article by a women was published in the mid 1990s. Hence,
articles by women seem to be, on average, more current while men seem to work more on outdated
topics.

## Dataset sampling

How we account for this skew? One of the Dans (McFarland? Jurafsky?) suggested drawing a sample
from the articles, so I generated a derived dataset with the following specifications:

- For each 5 year slice from 1950 to 2014, select 500 articles by men and 500 articles by women 
(with replacement), then create the same visualization.

@people with more background on sampling: is this a valid approach? It means that I have oversampled
some 5 year chunks and undersampled others. My original approach was to select the same number of
articles per five year period, thereby eliminating problems arising from different number of articles
per five year period. However, that doesn't solve the issue that men far outnumber women in the
early years, hence I selected 500 articles each by men and women for each 5 year period. But of 
course, that massively oversamples articles by women in the 1950s and 60s.

## Visualization with sampled dataset

At any rate, here's the visualization that we get when using a dataset with the same number of
articles by men and women for each 5 year period:
![](https://github.com/srisi/gender_history/raw/refactor/visualizations/gender_frequency_scatterplots/gender_weight_time_sampled.jpg)

Compared to the first chart, male topics have moved to the right and female topics to the left:
"Cultural Turn" moved from about 0.62 to 0.52. The blue cluster on the male side moved closer to the
center because articles written on these topics by women during the 1950s and 60s now get far more
weight.

So what do we see now?

Somewhat pointedly, there is no such thing as male history but there is female history. What I mean
is that no topic gets below a frequency score of 0.3 (i.e. 70% of the weight contributed by men)
but three topics ("Gender and Feminism," "Family," "Doctors and Patients") go above 0.7 (i.e. 70%
of the weight contributed by men). 

There's now also a curious cluster of Native American history topics on the right side? @Londa: 
did a large number of women work on these topics in the 1950s and 60s? Do you have an interpretation
on this?

I'm really fascinated by the contrast between "Narratives and Discourses" and "Cultural Turn." In
the earlier chart, "Cultural Turn" was one of the more female topics, in the new chart, 
"Narratives and Discourses" is the most male one but in either graph, the two topics are quite
far removed from each other. I'm intrigued by this distance because when labeling, the topics gave
Londa and me headaches. Both are historiography topics that deal language, culture, narratives, 
and interpretation, so seeing them so far removed is quite remarkable. But as I'm looking at it
now again, I notice much more men citing male philosophers and historians in "Narratives and 
Discourses." (Somewhat uncharitable: It's a topic about how historians can write about the past
that presumes both actors and writers to be male.) 
The topic shows quite a strong correlation with "German Intellectual History." 
"Cultural Turn," by contrast is much more about how to incorporate different perspectives into
history from women to subaltern. When we compare articles scoring above the 80th percentile for 
"Narratives and Discourses" to articles scoring above the 80th percentile for "Cultural Turn," the
top x distinctive terms for "Narratives and Discourses" are "history, philosophy, past, historian, 
events, theory." For "Cultural Turn," it's "women, american, workers, class, labor, colonial." I
find that quite intriguing.

Ok, those were more than just a few notes. Let me know what you think!

