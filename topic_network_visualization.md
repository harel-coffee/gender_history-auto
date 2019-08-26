# Topic Network Visualizations
Here are some initial experiments with topic network visualizations and notes
on the results. 

All of this is vaguely influenced by Ted Underwood's notes on visualizing
topic models (https://tedunderwood.com/2012/11/11/visualizing-topic-models/).
The gist of it is: There are ways of using network visualizations that look good
and can provide some new insights. However, they are not really scientific
because they require ad-hoc parameter tuning.

The most frequent method, which I'm using here as well, is to calculate the
correlation coefficients between all topics and use those coefficients
as the edges of the network. This means that a connection between, say, "gender" and
 "sexuality" indicates that authors whose dissertations score highly for "gender"
 oftentimes also score highly for "sexuality."

Given this implementation, every node is connected to every other node (though
some of the weights can be negative because the correlations are negative). For simplicity's
sake, let's say that we only keep the positive weights.

Here's the problem: this still leaves us with 20-40 edges per topic. If we visualized
this graph, it would be a mess of uninterpretable intersecting lines.

Which is a way of saying, we need to prune the edges. Here, again following Underwood, 
is a way that looks nice: 
- For every node keep the link to the most correlated topic even if weak (otherwise 
we'll get a lot of unattached floating topics)
- Keep the second strongest link if it has a correlation > 0.1
- Keep all links with correlations > 0.2

Again, these criteria are ad-hoc and chosen to make the graph look nice--they don't capture all
relevant correlations. 

Here's a complete heatmap of [topic-topic correlations](https://plot.ly/~stephan.risi/254/)

## Graph
The color of the node indicates how gendered it is. Green -> female, brown -> male. Light gray (e.g. 
development) indicates neutral topics.

The size of the node indicates its overall frequency, i.e. the large the more frequent the topic is.

For a final version, I can reorganize this plot such that the female topics are towards the left side
of the graph and the male topics towards the right side. However, since our topic model will still change,
I didn't want to spend time on this for the moment.

![alt text](https://github.com/srisi/gender_history/raw/master/data/networks/gephi_thin.png)

Notes and observations:
- gephi didn't like it when topics had the same names, e.g. we had 3 "labor" topics and 2 "history of 
the west" topics. I have tried to split them up, e.g. into "History of the West (Frontier)" and 
"History of the West (Industry)"
- The frontier topic is quite interesting, I think. For example, it links via colonialism to identity
construction and draws together Latin America and Native American history.
- Incidentally, the split in the Latin American topics is fascinating: We have the Latin America topic
that connects to the History of the West (Frontier), which I think is really a 19th century and earlier
Spanish colonial topic. By comparison, IR/Latin America is mostly a 20th century topic. "Hispanic" is a
connecting topic between them (the connection between "Latin America" and "Hispanic" got pruned out. 
Correlations: "IR/Latin America"-"Hispanic": 0.10. "Hispanic"-"Latin America": 0.09. "IR/Latin America"-
"Latin America": 0.02)
- At this point back to gender: I think the graph indicates what we already suspected: without female 
historians, we would know a lot less about womens' history, gender, sexuality, medicine, cultural history, 
African American history, film, and art. 
- Maybe more intriguing: It doesn't seem like gender history is ghettoized at all--it's deeply connected
to many other topics. The premier ghettoized topics that we have captured are Jewish history, sports 
history, and Islamic/Ottoman history. They are all barely connected to any other topics. 
- There's an obvious problem to this interpretation of course: maybe it just indicates that the topic model
can neatly capture some topics/themes while others come to reset in-between a number of them.
- An alternative interpretation is that topics like Jewish or medical history may have evolved substantially during the period we 
cover. However, some key terms that all historians in this area use ("jewish", "disease") have
remained the same and allow us to identify them quickly.


Here's a denser but I think less useful version:

![alt text](https://github.com/srisi/gender_history/raw/master/data/networks/gephi_dense.png)




