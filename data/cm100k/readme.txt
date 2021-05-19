SUMMARY
================================================================================

These files contain 103,584 anonymous ratings on 1,084 music tracks entered
by 1,054 CrowdFlower users between June 2015 and February 2016.

You may cite the following paper to acknowledge the use of the dataset in 
publications:

  Rocio Cañamares, Pablo Castells. Should I Follow the Crowd? A Probabilistic
  Analysis of the Effectiveness of Popularity in Recommender Systems. 41st
  Annual International ACM SIGIR Conference on Research and Development in
  Information Retrieval (SIGIR 2018). Ann Arbor, Michigan, USA, July 2018.

Complementary details and usage of the dataset by the authors are reported in
the following additional papers:

  Rocío Cañamares, Pablo Castells. From the PRP to the Low Prior Discovery
  Recall Principle for Recommender Systems. 41st Annual International ACM
  SIGIR Conference on Research and Development in Information Retrieval
  (SIGIR 2018). Ann Arbor, Michigan, USA. July 2018.

  R. Cañamares and P. Castells. A Probabilistic Reformulation of Memory-Based 
  Collaborative Filtering – Implications on Popularity Biases. International 
  ACM SIGIR Conference on Research and Development in Information Retrieval
  (SIGIR 2017). Tokyo, Japan, August 2017, pp. 215-224.


DATA COLLECTION
================================================================================

The 1,084 tracks were selected uniformly at random from the Deezer database
using its public API. The Deezer database had more than 30 million  tracks at 
the time of the selection (2015). 

Each user was assigned 100 of these tracks uniformly at random and was asked
to rate them. For each track, users had to listen to a short clip and select 
one of these five options:
- I really like it
- It’s nice, I enjoy listening to it
- So-so
- I don’t like it
- Flawed audio, or not really music

They also had to answer a Yes/No question about whether they had ever heard 
the song before.


RATINGS FILE DESCRIPTION
================================================================================

All ratings are contained in the file "ratings.txt" and are in the following 
format (tab separated, "\t"):

UserID \t TrackID \t Rating \t Known

- UserIDs range between 0 and 1054 
- TrackIDs correspond to the track ids in Deezer
- Ratings range from 1 to 4, representing the five options users were given: 
		4 - "I really like it"
		3 - "It’s nice, I enjoy listening to it"
		2 - "So-so"
		1 - "I don’t like it" or "Flawed audio, or not really music"
- Known indicates whether the user had already heard the track before (Known 
  = 1) or not (Known = 0).
- Each user has between 90 and 100 ratings.


TRACKS FILE DESCRIPTION
================================================================================

Track information is in the file "items.txt" and is in the following format 
(tab separated, "\t"):

TrackID \t Url \t Title \t Artist

- The field values are provided as they appeared on the Deezer database.
