from collections import OrderedDict
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load('en_core_web_sm')

class TextRank4Keyword():
    """Extract keywords from text"""
    
    def __init__(self):
        self.d = 0.85 # damping coefficient, usually is .85
        self.min_diff = 1e-5 # convergence threshold
        self.steps = 10 # iteration steps
        self.node_weight = None # save keywords and its weight

    
    def set_stopwords(self, stopwords):  
        """Set stop words"""
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True
    
    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences
        
    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab
    
    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs
        
    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())
    
    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1
            
        # Get Symmeric matrix
        g = self.symmetrize(g)
        
        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm!=0) # this is ignore the 0 element in norm
        
        return g_norm

    
    def get_keywords(self, number=10):
        """Print top number keywords"""
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        for i, (key, value) in enumerate(node_weight.items()):
            print(key + ' - ' + str(value))
            if i > number:
                break
        
        
    def analyze(self, text, 
                candidate_pos=['NOUN', 'PROPN'], 
                window_size=4, lower=False, stopwords=list()):
        """Main function to analyze text"""
        
        # Set stop words
        self.set_stopwords(stopwords)
        
        # Pare text by spaCy
        doc = nlp(text)
        
        # Filter sentences
        sentences = self.sentence_segment(doc, candidate_pos, lower) # list of list of words
        
        # Build vocabulary
        vocab = self.get_vocab(sentences)
        
        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)
        
        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)
        
        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))
        
        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1-self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr))  < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]
        
        self.node_weight = node_weight

text = '''
Let me begin by saying thanks to all you who've traveled, from far and wide, to brave the cold today.
We all made this journey for a reason. It's humbling, but in my heart I know you didn't come here just for me, you came here because you believe in what this country can be. In the face of war, you believe there can be peace. In the face of despair, you believe there can be hope. In the face of a politics that's shut you out, that's told you to settle, that's divided us for too long, you believe we can be one people, reaching for what's possible, building that more perfect union.
That's the journey we're on today. But let me tell you how I came to be here. As most of you know, I am not a native of this great state. I moved to Illinois over two decades ago. I was a young man then, just a year out of college; I knew no one in Chicago, was without money or family connections. But a group of churches had offered me a job as a community organizer for $13,000 a year. And I accepted the job, sight unseen, motivated then by a single, simple, powerful idea - that I might play a small part in building a better America.
My work took me to some of Chicago's poorest neighborhoods. I joined with pastors and lay-people to deal with communities that had been ravaged by plant closings. I saw that the problems people faced weren't simply local in nature - that the decision to close a steel mill was made by distant executives; that the lack of textbooks and computers in schools could be traced to the skewed priorities of politicians a thousand miles away; and that when a child turns to violence, there's a hole in his heart no government could ever fill.
It was in these neighborhoods that I received the best education I ever had, and where I learned the true meaning of my Christian faith.
After three years of this work, I went to law school, because I wanted to understand how the law should work for those in need. I became a civil rights lawyer, and taught constitutional law, and after a time, I came to understand that our cherished rights of liberty and equality depend on the active participation of an awakened electorate. It was with these ideas in mind that I arrived in this capital city as a state Senator.
It was here, in Springfield, where I saw all that is America converge - farmers and teachers, businessmen and laborers, all of them with a story to tell, all of them seeking a seat at the table, all of them clamoring to be heard. I made lasting friendships here - friends that I see in the audience today.
It was here we learned to disagree without being disagreeable - that it's possible to compromise so long as you know those principles that can never be compromised; and that so long as we're willing to listen to each other, we can assume the best in people instead of the worst.
That's why we were able to reform a death penalty system that was broken. That's why we were able to give health insurance to children in need. That's why we made the tax system more fair and just for working families, and that's why we passed ethics reforms that the cynics said could never, ever be passed.
It was here, in Springfield, where North, South, East and West come together that I was reminded of the essential decency of the American people - where I came to believe that through this decency, we can build a more hopeful America.
And that is why, in the shadow of the Old State Capitol, where Lincoln once called on a divided house to stand together, where common hopes and common dreams still, I stand before you today to announce my candidacy for President of the United States.
I recognize there is a certain presumptuousness - a certain audacity - to this announcement. I know I haven't spent a lot of time learning the ways of Washington. But I've been there long enough to know that the ways of Washington must change.
The genius of our founders is that they designed a system of government that can be changed. And we should take heart, because we've changed this country before. In the face of tyranny, a band of patriots brought an Empire to its knees. In the face of secession, we unified a nation and set the captives free. In the face of Depression, we put people back to work and lifted millions out of poverty. We welcomed immigrants to our shores, we opened railroads to the west, we landed a man on the moon, and we heard a King's call to let justice roll down like water, and righteousness like a mighty stream.
Each and every time, a new generation has risen up and done what's needed to be done. Today we are called once more - and it is time for our generation to answer that call.
For that is our unyielding faith - that in the face of impossible odds, people who love their country can change it.
That's what Abraham Lincoln understood. He had his doubts. He had his defeats. He had his setbacks. But through his will and his words, he moved a nation and helped free a people. It is because of the millions who rallied to his cause that we are no longer divided, North and South, slave and free. It is because men and women of every race, from every walk of life, continued to march for freedom long after Lincoln was laid to rest, that today we have the chance to face the challenges of this millennium together, as one people - as Americans.
All of us know what those challenges are today - a war with no end, a dependence on oil that threatens our future, schools where too many children aren't learning, and families struggling paycheck to paycheck despite working as hard as they can. We know the challenges. We've heard them. We've talked about them for years.
What's stopped us from meeting these challenges is not the absence of sound policies and sensible plans. What's stopped us is the failure of leadership, the smallness of our politics - the ease with which we're distracted by the petty and trivial, our chronic avoidance of tough decisions, our preference for scoring cheap political points instead of rolling up our sleeves and building a working consensus to tackle big problems.
For the last six years we've been told that our mounting debts don't matter, we've been told that the anxiety Americans feel about rising health care costs and stagnant wages are an illusion, we've been told that climate change is a hoax, and that tough talk and an ill-conceived war can replace diplomacy, and strategy, and foresight. And when all else fails, when Katrina happens, or the death toll in Iraq mounts, we've been told that our crises are somebody else's fault. We're distracted from our real failures, and told to blame the other party, or gay people, or immigrants.
And as people have looked away in disillusionment and frustration, we know what's filled the void. The cynics, and the lobbyists, and the special interests who've turned our government into a game only they can afford to play. They write the checks and you get stuck with the bills, they get the access while you get to write a letter, they think they own this government, but we're here today to take it back. The time for that politics is over. It's time to turn the page.
We've made some progress already. I was proud to help lead the fight in Congress that led to the most sweeping ethics reform since Watergate.
But Washington has a long way to go. And it won't be easy. That's why we'll have to set priorities. We'll have to make hard choices. And although government will play a crucial role in bringing about the changes we need, more money and programs alone will not get us where we need to go. Each of us, in our own lives, will have to accept responsibility - for instilling an ethic of achievement in our children, for adapting to a more competitive economy, for strengthening our communities, and sharing some measure of sacrifice. So let us begin. Let us begin this hard work together. Let us transform this nation.
Let us be the generation that reshapes our economy to compete in the digital age. Let's set high standards for our schools and give them the resources they need to succeed. Let's recruit a new army of teachers, and give them better pay and more support in exchange for more accountability. Let's make college more affordable, and let's invest in scientific research, and let's lay down broadband lines through the heart of inner cities and rural towns all across America.
And as our economy changes, let's be the generation that ensures our nation's workers are sharing in our prosperity. Let's protect the hard-earned benefits their companies have promised. Let's make it possible for hardworking Americans to save for retirement. And let's allow our unions and their organizers to lift up this country's middle-class again.
Let's be the generation that ends poverty in America. Every single person willing to work should be able to get job training that leads to a job, and earn a living wage that can pay the bills, and afford child care so their kids have a safe place to go when they work. Let's do this.
Let's be the generation that finally tackles our health care crisis. We can control costs by focusing on prevention, by providing better treatment to the chronically ill, and using technology to cut the bureaucracy. Let's be the generation that says right here, right now, that we will have universal health care in America by the end of the next president's first term.
Let's be the generation that finally frees America from the tyranny of oil. We can harness homegrown, alternative fuels like ethanol and spur the production of more fuel-efficient cars. We can set up a system for capping greenhouse gases. We can turn this crisis of global warming into a moment of opportunity for innovation, and job creation, and an incentive for businesses that will serve as a model for the world. Let's be the generation that makes future generations proud of what we did here.
Most of all, let's be the generation that never forgets what happened on that September day and confront the terrorists with everything we've got. Politics doesn't have to divide us on this anymore - we can work together to keep our country safe. I've worked with Republican Senator Dick Lugar to pass a law that will secure and destroy some of the world's deadliest, unguarded weapons. We can work together to track terrorists down with a stronger military, we can tighten the net around their finances, and we can improve our intelligence capabilities. But let us also understand that ultimate victory against our enemies will come only by rebuilding our alliances and exporting those ideals that bring hope and opportunity to millions around the globe.
But all of this cannot come to pass until we bring an end to this war in Iraq. Most of you know I opposed this war from the start. I thought it was a tragic mistake. Today we grieve for the families who have lost loved ones, the hearts that have been broken, and the young lives that could have been. America, it's time to start bringing our troops home. It's time to admit that no amount of American lives can resolve the political disagreement that lies at the heart of someone else's civil war. That's why I have a plan that will bring our combat troops home by March of 2008. Letting the Iraqis know that we will not be there forever is our last, best hope to pressure the Sunni and Shia to come to the table and find peace.
Finally, there is one other thing that is not too late to get right about this war - and that is the homecoming of the men and women - our veterans - who have sacrificed the most. Let us honor their valor by providing the care they need and rebuilding the military they love. Let us be the generation that begins this work.
I know there are those who don't believe we can do all these things. I understand the skepticism. After all, every four years, candidates from both parties make similar promises, and I expect this year will be no different. All of us running for president will travel around the country offering ten-point plans and making grand speeches; all of us will trumpet those qualities we believe make us uniquely qualified to lead the country. But too many times, after the election is over, and the confetti is swept away, all those promises fade from memory, and the lobbyists and the special interests move in, and people turn away, disappointed as before, left to struggle on their own.
That is why this campaign can't only be about me. It must be about us - it must be about what we can do together. This campaign must be the occasion, the vehicle, of your hopes, and your dreams. It will take your time, your energy, and your advice - to push us forward when we're doing right, and to let us know when we're not. This campaign has to be about reclaiming the meaning of citizenship, restoring our sense of common purpose, and realizing that few obstacles can withstand the power of millions of voices calling for change.
By ourselves, this change will not happen. Divided, we are bound to fail.
But the life of a tall, gangly, self-made Springfield lawyer tells us that a different future is possible.
He tells us that there is power in words.
He tells us that there is power in conviction.
That beneath all the differences of race and region, faith and station, we are one people.
He tells us that there is power in hope.
As Lincoln organized the forces arrayed against slavery, he was heard to say: "Of strange, discordant, and even hostile elements, we gathered from the four winds, and formed and fought to battle through."
That is our purpose here today.
That's why I'm in this race.
Not just to hold an office, but to gather with you to transform a nation.
I want to win that next battle - for justice and opportunity.
I want to win that next battle - for better schools, and better jobs, and health care for all.
I want us to take up the unfinished business of perfecting our union, and building a better America.
And if you will join me in this improbable quest, if you feel destiny calling, and see as I see, a future of endless possibility stretching before us; if you sense, as I sense, that the time is now to shake off our slumber, and slough off our fear, and make good on the debt we owe past and future generations, then I'm ready to take up the cause, and march with you, and work with you. Together, starting today, let us finish the work that needs to be done, and usher in a new birth of freedom on this Earth.
'''

tr4w = TextRank4Keyword()
tr4w.analyze(text, candidate_pos = ['NOUN', 'PROPN', 'VERB'], window_size=4, lower=False)
tr4w.get_keywords(50)

text2 = '''
Good morning. As some of you know, Senator Lugar and I recently traveled to Russia, Ukraine, and Azerbaijan to witness firsthand both the progress we're making in securing the world's most dangerous weapons, as well as the serious challenges that lie ahead.
Now, few people understand these challenges better than the co-founder of the Cooperative Threat Reduction Program, Dick Lugar, and this is something that became particularly clear to me during one incident on the trip.
We were in Ukraine, visiting a pathogen laboratory in Kiev. This is a city of two and a half million, and in a non-descript building right in the middle of town stood this facility that once operated on the fringes of the Soviet biological weapons program.
We entered through no fences or discernible security, and once we did, we found ourselves in a building with open first-floor windows and padlocks that many of us would not use to secure our own luggage.
Our guide then brought us right up to what looked like a mini-refrigerator. Inside, staring right at us, were rows upon rows of test tubes. She picked them up, clanked them around, and we listened to the translator explain what she was saying. Some of the tubes, he said, were filled with anthrax. Others, the plague.
At this point I turned around and said "Hey, where's Lugar? Doesn't he want to see this?" I found him standing about fifteen feet away, all the way in the back of the room. He looked at me and said, "Been there, done that."
Of course, Dick has been there and he has done that, and thanks to the Cooperative Threat Reduction Programs he co-founded with Senator Sam Nunn, we've made amazing progress in finding, securing, and guarding some of the deadliest weapons that were left scattered throughout the former Soviet Union after the Cold War.
But this is one story that shows our job is far from finished at a time when demand for these weapons has never been greater.
Right now, rogue states and despotic regimes are looking to begin or accelerate their own nuclear programs. And as we speak, members of Al Qaeda and other terrorists organizations are aggressively pursuing weapons of mass destruction, which they would use without hesitation.
We've heard the horror stories - attempts by rogue states to recruit former Soviet weapons scientists; terrorists shopping for weapons grade materials on the black market. Some weapons experts believe that terrorists are likely to find enough fissile material to build a bomb in the next ten years - and we can imagine with horror what the world will be like if they succeed.
Today, experts tell us that we're in a race against time to prevent this scenario from unfolding. And that is why the nuclear, chemical, and biological weapons within the borders of the former Soviet Union represent the greatest threat to the security of the United States - a threat we need to think seriously and intelligently about in the months to come.
Fortunately, the success of Cooperative Threat Reduction - especially in securing nuclear weapons - serves as a model of how we can do this. And so the question we need to be asking ourselves today is, what is the future of this program? With the situation in Russia and the rest of the former Soviet Union so drastically different than it was in 1991, or even in 1996 or 2001, what must we do to effectively confront this threat in the days and years to come?
The answers to these questions will require sustained involvement by the Executive Branch, Congress, non-governmental organizations, and the international community. Everyone has a role to play, and everyone must accelerate this involvement.
For my part, I would suggest three important elements that should be included in such a discussion.
First, the Nunn-Lugar program should be more engaged in containing proliferation threats from Soviet-supplied, civilian research reactors throughout Russia and the Independent States.
The Department of Energy and others have certainly made progress in converting civilian reactors to low-enriched uranium, taking back spent fuel, and closing unnecessary facilities.
Yet, a serious threat still remains. Many of these aging research facilities have the largest, least secure quantities of highly enriched uranium in the world - the quickest way to a nuclear weapon. For a scientist or other employee to simply walk out of the lab with enough material to construct a weapon of mass destruction is far too easy, and the consequences would be far too devastating. Not to mention the environmental and public health and safety catastrophe that could come from a failure to store and transport these materials safely and securely.
In a way that balances the needs of science and security, more needs to be done to bring these materials - as well as other sources that can be used to construct improvised nuclear weapons and radiological devices -- under control and dramatically reduce the proliferation threat they pose.
In the years ahead, this should become an increasing priority for the Nunn-Lugar program, the Congress, and the Russians, who are already taking important steps to help implement these programs.
I want to turn to a second critical area: biological weapons threat reduction programs.
Throughout the Cold War, the Soviet Union was engaged in a massive undertaking in the field of germ warfare.
At its height in the late 1980's, this program stockpiled of some of the most dangerous agents known to man - plague, smallpox, and anthrax - to name just a few. As one book says, "disease by the ton was its industry."
Besides the devastation they can cause to a civilian population, biological agents can also be effective in asymmetrical warfare against U.S. troops. While they are often difficult to use, they are easy to transport, hard to detect, and, as we saw in Kiev, not always well secured.
Here in Washington, we saw what happened when just two letters filled with just a few grams of Anthrax were sent to the U.S. Senate. Five postal employees were killed and the Senate office buildings were closed for months.
This was two letters.
Fortunately, however, we've made some good progress on this front. For years, Nunn-Lugar programs have been effectively upgrading security at sites in six countries across the former Soviet Union. And the Kiev story is heading in the right direction - while we were in Ukraine, Dick, through his tireless and personal intervention, was able to achieve a breakthrough with that government, bringing that facility and others under the Cooperative Threat Reduction program.
But because of the size, secrecy, and scope of the Soviet biological weapons program, we are still dangerously behind in dealing with this proliferation threat. We need to be sure that Nunn-Lugar is increasingly focused on these very real non-proliferation and bioterrorism threats.
One of the most important steps is for Russia to permit the access and transparency necessary to deal with the threat.
Additional steps should also be taken to consolidate and secure dangerous pathogen collections, strengthen bio-reconnaissance networks to provide early warning of bio-attack and natural disease outbreaks, and have our experts work together to develop improved medical countermeasures. As the Avian Influenza outbreak demonstrates, even the zealous Russian border guard is helpless against the global sweep of biological threats.
My third recommendation - which I'll just touch briefly on and let Senator Lugar talk about in more detail - is that we need to start thinking creatively about some of the next-generation efforts on nuclear, biological, and chemical weapons.
On our trip, we saw two areas where this is possible: elimination of heavy conventional weapons, and interdiction efforts to help stop the flow of dangerous materials across borders.
In Donetsk, I stood among piles of conventional weapons that were slowly being dismantled. While the government of Ukraine is making progress here, the limited funding they have means that at the current pace, it will take sixty years to dismantle these weapons. But we've all seen how it could take far less time for these weapons to leak out and travel around the world, fueling insurgencies and violent conflicts from Africa to Afghanistan. By destroying these inventories, this is one place we could be making more of a difference.
One final point. For any of these efforts that I've mentioned to work as we move forward, we must also think critically and strategically about Washington's relationship with Moscow.
Right now, there are forces within the former Soviet Union and elsewhere that want these non-proliferation programs to stop. Our detention for three hours in Perm is a testament to these forces. Additionally, in the last few years, we've seen some disturbing trends from Russia itself - the deterioration of democracy and the rule of law, the abuses that have taken place in Chechnya, Russian meddling in the former Soviet Union - that raise serious questions about our relationship.
But when we think about the threat that these weapons pose to our global security, we cannot allow the U.S.-Russian relationship to deteriorate to the point where Russia does not think it's in their best interest to help us finish the job we started. We must safeguard these dangerous weapons, material, and expertise. .
One way we could strengthen this relationship is by thinking about the Russians as more of a partner and less of a subordinate in the Cooperative Threat Reduction effort.
This does not mean that we should ease up one bit on issues affecting our national security. Outstanding career officials who run the Nunn-Lugar program -- people like Col. Jim Reid and Andy Weber who are here this morning -- will be there every step of the way to ensure that U.S. interests are protected.
Time and time again on the trip, I saw their skill and experience when negotiating with the Russians. I also saw their ability to ensure that shortcomings were addressed and programs were implemented correctly.
But thinking of the Russians more as partners does mean being more thoughtful, respectful, and consistent about what we say and what we do. It means that the Russians can and should do more to support these programs. And it means more sustained engagement, including more senior-level visits to Nunn-Lugar program sites.
It's important for senior officials to go and visit these sites, to check their progress and shortcomings; to see what's working and what's not. But lately we haven't seen many of these visits. We need to see more.
We also need to ensure that the Cooperative Threat Reduction umbrella agreement, due to expire in 2006, is renewed in a timely manner.
And we need to work together to obtain a bilateral agreement on biological threat reduction.
There is no doubt that there is a tough road ahead. It will be difficult. And it will be dangerous.
But, when I think about what is at stake I am reminded by a quote from the late President Kennedy given in a speech at American University in 1963 about threats posed by the Soviet Union. "Let us not be blind to our differences--but let us also direct attention to our common interests and to the means by which those differences can be resolved...For in the final analysis, our most basic common link is that we all inhabit this small planet. We all breathe the same air. We all cherish our children's future. And we are all mortal.''
Much of what President Kennedy described in 1963 remains true to this day - and we owe it to ourselves and our children to get it right.
Thank you.
'''

print("------------------------------------------")
tr4w = TextRank4Keyword()
tr4w.analyze(text2, candidate_pos = ['NOUN', 'PROPN', 'VERB'], window_size=4, lower=False)
tr4w.get_keywords(50)
