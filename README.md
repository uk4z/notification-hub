# Notification hub
The project is to create an app to transform voice message into a written notification. 


## Text summarization

Text summarization is one of the most known NLP problem and it will be tackled in this section. The first simple approach is to do an extractive summarization. 

We will be working with an example to try understanding the mechanisms of this type of text summarization. Other examples are available in the corresponding folder `example`. Let's say a person called Sarah sends an audio to her friend Michael. Here is the transcription of the audio to analyze:  

```
Sarah: (sighs) Hey, Michael. I hope you're doing well. I know it's been a while since we caught up, but I really need to talk to someone, and you've always been such a great friend.

(softly) So, Francis and I...we broke up. It happened a few days ago, and I'm struggling to cope with it. It's been really hard on me. You know how close we were, and this just feels like a huge void in my life right now.

I remember how you were there for me when we first got together, and now, I don't know who else to turn to. I miss your comforting words, your wisdom, and your laughter. I just need my friend right now, Michael.

(voice cracking) I was wondering if we could meet up soon, maybe this weekend? I'd really appreciate your support and advice. You've always had a way of helping me see things more clearly, and I really need that right now.

I hope you're not too busy, but if you are, I completely understand. Just hearing your voice in person would mean the world to me. I could use a good friend by my side, especially right now. Let me know if you're available, and we can plan something. Thanks, Michael. I miss you and can't wait to see you.
```

Basically, Sarah wants to catch up with Michael telling him about her trip she took with some of her friends. 

Using **Spacy** tools, we can implement an algorithm counting the number of meaningful word apparitions. The text contains 89 meaningful words with 1 or 2 apparitions. From this data we can normalize the frequence of apparition to get a more accurate weight for the words. Once this is done, we are able to weight each sentences with the most used meaningful words. Let's say we take the three most important sentences according to the algorithm. 

We can choose the number of meaningful words used for the algorithm. Let's see the different results for some number of most used meaningful words taken into account. 

`n_words = 1:` 

```
I know it's been a while since we caught up, but I really need to talk to someone, and you've always been such a great friend.

You know how close we were, and this just feels like a huge void in my life right now.

I remember how you were there for me when we first got together, and now, I don't know who else to turn to.
```

`n_words >= 5:`

```
I know it's been a while since we caught up, but I really need to talk to someone, and you've always been such a great friend.

I just need my friend right now, Michael.

You know how close we were, and this just feels like a huge void in my life right now.
```

We can see a slight difference if we only take one word into account. Here the downside is limited but we can imagine that we would lose too much information by taking one word into account. On the other hand, taking every words might not be the most efficient in terms of space complexity. Therefore, we can add a criteria to select words with a total weight higher than *0.5*. 
