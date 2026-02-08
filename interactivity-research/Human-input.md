I'm reading learnings. I would organize the headlines like so:

First, let's assume this is a machine learning project (conceptually, our artifacts may not map cleanly to it). The training data is our conversation and any datasets or algorithms we download from online. The validation set is split off from our training dataset. I'll split test data into two categories, automated and manual. Automated is my user entered tap data. manual is looking at the LEDs. Looking at the LEDs is ground truth but is the most manual. With that, below is my information breakdown.

Details:
### Discovery: Bass Detection is Fundamentally Broken in Naive Implementations
	- this just means our code isnt working yet, we havent foudn the right breakdown
### Discovery: librosa beat_track Doubles Tempo on Syncopated Rock
	- we think librosa may  not be the right tool
### Insight: User Tap Annotations Are Richer Than Beat Positions
	- User may be bad at tapping so not guarunteed. Worth taking with a grain of salt. That said, human input is important and the particular user's taste should be preserved as at least a validation set to check against.
### Insight: Two Independent Quality Axes
	- This is a key. I see the music decomposition quality as more of a validation thing, whereas the actual manual LEDs as more of a test thing. or maybe they are boeth test but in two steps.
### Insight: The "Feeling Layer" Requires Human-in-the-Loop Development
	- Theory, not directly actionable.
### Discovery: Feelings Map to Derivatives, Not Absolute Values
	- This another key point from our research i'd like to keep around. I.E. -- what are the _features_ we need to pull from the data to properly represent important moments? This is one of them. 
### Discovery: "Airiness" Is Deviation From Local Context, Not Absolute Acoustic Properties	
	- Another feature.
### Discovery: User Taps Track Bass Peaks, Not Onsets (and Switch Modes)
	- Theory: onsets may be a low value feature
### Discovery: All Algorithmic Beat Trackers Fail on Dense Rock
	- To finalize this claim we should do some testing where i listen to the rock and verify your findings.
### Research Conclusions
	- I'm not sure i agree with most of these, except the constraints and the steal. which i don't understand.
### Tools Built
	- OK documentation

OK now i'll read STATUS.md.
I've already categorized pillar 1 as the first test step. if it proves good, we could move it to validation step. I've also already said that the user could be WRONG about what makes a good light show, or the user could have miscommunicated what they wanted to highlight about the tap segement.

for all of pillar1 1-6 i think these are good principles about musical analysis from our pov. Let's keep these seperate from external sources like wled's audio reactive and then we MIGHT compose external and internal research later. TBD

The others look ok, although it's a lot of information. probably easy to get confused. 
