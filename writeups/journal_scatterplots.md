# Journal Scatterplots

How different are our journals? Would it make sense to exclude some of them?
Or should we reduce the number of journals? 

Let's first see what's in the different journals.

This analysis is an extension of the [Gender-Weight Scatterplots](https://github.com/srisi/gender_history/blob/master/writeups/gender_weight_scatterplot.md).

For context, let's go back to the main scatterplot:
![](https://raw.githubusercontent.com/srisi/gender_history/refactor/visualizations/gender_frequency_scatterplots/gender_time_weight.jpg)

That's the polished version of this larger graph with all topics:
![](https://raw.githubusercontent.com/srisi/gender_history/master/visualizations/gender_frequency_scatterplots/gfs_labeling_copy.png)

All of the axis in this graph mean the same as in the first one, it's just larger and you 
probably need to click on the image to zoom in. In return, you'll be able to see the topic 
distribution in the middle in more detail.

You can find scatterplots for all individual journals [here.](https://github.com/srisi/gender_history/tree/master/visualizations/gender_frequency_scatterplots)

## AHR
With that in mind, let's have a look at the AHR chart:
![](https://raw.githubusercontent.com/srisi/gender_history/master/visualizations/gender_frequency_scatterplots/single_journalThe_American_Historical_Review.png)

Some observations: 
- We now have topics all the way to about 0.05%. Curiously, three of the lowest scoring topics deal
with Native American history
- Apart from that I think we can label AHR a pretty general history journal. I would have expected
(middle left) Afro-Eurasian Trade Routes, Spanish Empire, and Italian Fascism to drop much further
- gender and feminism is even more dominated by women than in the general dataset.
- However, there are now also some topics that are very male dominated (0.2 and less)

## AHR and JAH
For good measure, let's combine AHR with the JAH (and its predecessor, the Mississippi Valley 
 Historical Review). Generally quite similar:
 Note: you can find the JAH on its own [here.](https://raw.githubusercontent.com/srisi/gender_history/master/visualizations/gender_frequency_scatterplots/single_journalThe_Journal_of_American_History.png)
 
 Worth noting: AHR and JAH cover about one third of all articles (3801 out of 10367), so focusing on
 just them would give us a substantial chunk of data.
 
 ![](https://raw.githubusercontent.com/srisi/gender_history/master/visualizations/gender_frequency_scatterplots/ahr_and_jah.png)

## History and Theory
This was a journal that Londa was worried about because, well, it says theory right in the title.
I think it's actually not so bad. Sure, the historiography articles dominate but I think that's fine
for a particular kind of general history journal.
You can also find a [chart with all journals except History and Theory] (https://github.com/srisi/gender_history/raw/master/visualizations/gender_frequency_scatterplots/all_except_history_and_theory.png) though it looks quite similar
to the chart with all journals included

![](https://raw.githubusercontent.com/srisi/gender_history/master/visualizations/gender_frequency_scatterplots/single_journalHistory_and_Theory.png)

## Reviews in American History
We should probably delete Reviews in American History because we only include articles but not 
reviews. Hence, there are only 132 articles for this journal. And because of the low number, 
most topics are either very male or female.
![](https://raw.githubusercontent.com/srisi/gender_history/master/visualizations/gender_frequency_scatterplots/single_journalReviews_in_American_History.png)

## Ethnohistory
This is where our Native American history topics get their weight. I think this is a very 
anthropology/ethnology heavy topic that doesn't exist anymore today, hence why our "Indigenous History"
general approach peaks in the 1950s and 60s.
![](https://raw.githubusercontent.com/srisi/gender_history/master/visualizations/gender_frequency_scatterplots/single_journalEthnohistory.png)


## Other observations
- The [Journal of Interdisciplinary History](https://raw.githubusercontent.com/srisi/gender_history/master/visualizations/gender_frequency_scatterplots/single_journalThe_Journal_of_Interdisciplinary_History.png) has almost 20% weight for demography.
- The [Journal of World History](https://raw.githubusercontent.com/srisi/gender_history/master/visualizations/gender_frequency_scatterplots/single_journalJournal_of_World_History.png)
does world history and the [Journal of Social History](https://raw.githubusercontent.com/srisi/gender_history/master/visualizations/gender_frequency_scatterplots/single_journalJournal_of_Social_History.png)
does social history. I know, shocker!

