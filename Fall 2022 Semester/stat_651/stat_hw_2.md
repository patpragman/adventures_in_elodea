Patrick Pragman
UAID 30812903

Stat F651

Homework 2,



**Problem 1**

(Casella/Berger Problem 1.1) For each of the following experiments, describe  
the sample space. Also, state whether the sample space is finite, countably  
infinite, or uncountably infinite.

 

Problem (1a) Toss a coin four times.

 

A sample space is "the set $S$ of all possible outcomes of a particular experiement."  In this case, our experiment consists of tossing a coin four times.  That means for each toss we have an outcome $H$ or $T$ which stack together for the entire experiment we see something like  $\{HTHT\}$ 



In this example, the first flip yielded heads, the second tails, the third heads, and the fourth tails in $\{TTTT\}$ all four flips resulted in tails..

 The total number of singletons in the sample space is $2^{4} = 16$.  Since there are a finite number of possibilities in the sample space, the sample space is finite.

---

Problem (1b)  Count the number of insect-damaged leaves on a plant.



For this one to make sense, we have to think about the experiment a 
little bit. We take in a "plant" as an experiment, and we spit out a 
value $a$ where $a$ is the number of damaged leaves.

Then  $\mathcal{S} = \{a \text{ such that }a\in \mathbb{Z}^+ \cup 0 \} =\{0, 1, 2, 3, ...\}$

That is, "the set of all positive integeres $a$ or 0" provided there is not a "single leafiest" plant out there in the universe.  In it's current form, the sample space is countably infinite.

---

Problem (1c)  Measure the lifetime (in hours) of a particular brand of light bulb.



If we can measure arbitrarily precise hours (not practical in the real world, but maybe we want to think that we can) then 

   $\mathcal{S} = \{x: x \geq 0\} = \mathbb{R}^+$ which is uncountably infinite.

If we don't allow for arbitrary precision, and we partitioned the reals into a "smaller" sample space, we could get a countably infinite sample space equal to the positive integers and zero.  I'm not sure what is a better strategy here.

---

Problem (1d)  Record the weight of 10-day-old rats

This again is related to the precision of our measuring tools, but provided that a 10-day-old rat's length can be a real number greater than zero, then I'd say this is "uncountably inifinite" and $\mathcal{S}$ is the set of all positive real numbers.  That is: $\mathcal{S} = \{x: x \geq 0\} = \mathbb{R}^+$

If we are forced to round to some nearest fraction because of our tools, then this becomes countably infinite (because the rational numbers and the integers have the same cardinality) but for all practical purposes, the length of a baby rat is a real number, so I'd say this is uncountably infinite.

I would also say that the sample space also depends on whether we mark the "dad baby rats" as $0$ or something else?  If a rat lives for 9 days, then dies, do we record a zero, or do we record the 9 day old length?  If the sample space includes "all possible values of the experiment" perhaps we could have a value "DEAD" meaning that the rat died?  This seems somewhat more complex than at first glance.  

Also, we could have a "biggest possible rat after 10 days" that could further "bound" the sample space, but that doesn't change that it's uncountably infinite.

---

Problem (1e) Observe the proportion of defectives in a shipment of electronic components.

This seems similar to the leaf problem.  Each experiment is a shipment of electronic components, and the value returned from that experiment is rational number of $\frac{\text{defective components}}{\text{total components}}$.  If our experiment is for any hypothetical amount of components in any shipment in the universe and the total number of components is some arbitrary integer greater than zero, then the sample space is countably infinite because you could get any rational number greater than zero.

This seems wrong to me though, because - really if you're looking at a specific shipment of electronic components, then $\mathcal{S}$ is finite - it's bounded by the size of the shipment.  In a simple case, suppose that there are only 3 possible components in the shipment.  Then: 

$\mathcal{S} = \{\frac{0}{3}, \frac{1}{3}, \frac{2}{3}, \frac{3}{3} \}$

Which is decidedly countable - namely, $|\mathcal{S}| = 4$.  

So, I think a more fitting way to look at it would be to say:

For an shipment of size $n$ where $n \in \mathbb{N}$ the sample space of defective equipment in that shipment is $\mathcal{S} = \{\frac{a}{n} \text{ such that } a\in\mathbb{N}\cup0  \text{ and } a \leq n \} = \{ \frac{0}{n}, \frac{1}{n}, \frac{2}{n}, ... 1 \} $

---

**Problem 2**

Casella/Berger Problem 1.4.) For events A and B, find formulas for the prob-  
abilities of the following events in terms of the quantities P (A), P (B) and  
P (A ∩ B). (These take a bit of thinking; I find that it helps me sort them out  
if I sketch a Venn diagram for them.)



Problem (2a) either $A$ or $B$ or both.

*add the probabilities of the two regions togehter and subtract the middle so you don't double count*

$P(A) + P(B) - P(A \cap B)$

---

Problem (2b) either A or B but not both.

Logically this is all the probability of stuff that's in $A$ minus the probability of the stuff in "both" then the probability of all the stuff in $B$ minus the probability of the stuff in both.
That's a mouthful to say  "subtract both twice" yielding:

$P(A) + P(B) - 2P(A \cap B)$

---

Problem (2c) at least one of $A$ or $B$



At least one means "the probability that it lands in A or B or both" that is the probability that it lands in $A \cup B$.  That's the same as number 1, so...

$P(A) + P(B) - P(A \cap B)$

---

Problem (2d) at most one of $A$ or $B$



At most $A$ or $B$ implies that a result could land in $A$ or it could land in $B$ or it could land in neither of them - but it cannot land in the region where they both are - that is $A \cap B$.  So we care about the probability of $(A \cap B)^c$

That works out to:
$P((A\cap B)^c) = 1 - P(A \cap B)$

---

**Problem 3**

(Casella/Berger Problem 1.5.) Approximately 1/3 of all human twins are iden-  
tical (one-egg) and 2/3 are fraternal (two-egg) twins. Identical twins are neces-  
sarily the same sex, with male and female being equally likely. Among fraternal  
twins, approximately 1/4 are both female, 1/4 are both male, and half are one  
male and one female. Finally, among all U.S. births, approximately 1 in 90 is a  
twin birth. Define the following events:

A = {a U.S. birth results in twin females}  
B = {a U.S. birth results in identical twins}  
C = {a U.S. birth results in twins}  

Problem (3a) State, in words, the event A ∩ B ∩ C.  



The set of identical twin female births in the US.

---

Problem (3b) Find P (A ∩ B ∩ C)

$P(A\cap B \cap C) = \frac{1}{90} \frac{1}{3} \frac{1}{2} = \frac{1}{540}$

---

**Problem 4**

Casella/Berger Problem 1.2d) Verify the following identity (by showing that  
each side is a subset of the other side):  
A ∪ B = A ∪ (B ∩ $A^c$)  



Proof:

Suppose $A$ and $B$ are sets.

Let $x \in A\cup B$.



This gives us two possibilities:

 $x \in A$ or $x \in B$

if $x\in A$ then $x \in A \cup (B \cap A^c)$ because $x \in A$.

if $x \in B$ then $x \in B \cap A^c$ so $x \in A \cup (B \cap A^c)$



either way, $x \in A \cup (B \cap A^c)$

hence, $A \cup B \subseteq A \cup (B \cap A^c)$.



What if, on the other hand, we let $x \in A \cup (B \cap A^c)$.

Then again we have two possible situations

$x \in A$ or $x \in (B \cap A^c)$.



if $x\in A$, then $x \in A \cup B$ by the definition of set union.

if $x \in (B \cap A^c)$, then $x \in B$ so $x \in A \cup B$.

Either way, $x \in A \cup B$, so

$A \cup (B \cap A^c) \subseteq A \cup B$.



Therefore $A \cup (B \cap A^c) = A \cup B$.

---

Sketch a Venn diagram to illustrate:

the red area of $A \cup B$:

![](/home/patrickpragman/.var/app/com.github.marktext.marktext/config/marktext/images/2022-09-08-20-35-18-IMG_0050.JPG)

if they are disjoint it looks like this

![](/home/patrickpragman/.var/app/com.github.marktext.marktext/config/marktext/images/2022-09-08-20-33-36-IMG_0049.JPG)

Ok, now let's look at the area $A^c \cap B$ - that is "everything outside A but inside B"

![](/home/patrickpragman/.var/app/com.github.marktext.marktext/config/marktext/images/2022-09-08-20-41-48-IMG_0051.JPG)

When you union that with A you get the same thing as $A \cup B$.  Of course, it's not rigorous without the written proof, but it makes sense "the union of two things is the same as the union of the first thing with intersection of one thing with every that is not in the other."

---

**Problem 5**

If P (A) = 1/3 and P ($B^c$) = 1/4, can events A and B be disjoint? (Casella/Berger  
problem 1.13.)

$P(A) = \frac{1}{3}$ and $P(B^c) = \frac{1}{4}$

so $P(B) = \frac{3}{4}
$

Suppose $A \cap B = \emptyset$ ($A$ and $B$ are disjoint.)

Then $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ by theorem 1.2.2b.

further, by Theorem 1.2.1a $P(A\cap B) = 0$

so,

$P(A \cup B) = P(A) + P(B) = \frac{1}{3} + \frac{3}{4} \geq 1$.  This is a contradiction!  Thus $A$ and $B$ cannot be disjoint.



**Problem 6**

Problem (6a) Sketch a Venn diagram to illustrate a sample space containing events A  
and B such that P (A∩B) = 0.12, P (A\B) = 0.28, and P ((A∪B)c) = 0.42.  

I tried my hand at one of the venn diagram libraries in R with this one - the numbers inside the circles are the "percent" probabilities expressed as whole numbers:

![](/home/patrickpragman/.var/app/com.github.marktext.marktext/config/marktext/images/2022-09-05-13-27-44-image.png)

Problem (6b) Find (calculate) P (A), P (B), P (A ∪ B), and P (Ac ∩ B).



$P(A) = P(A \setminus B) + P(A \cap B) = 0.28 + 0.12 = 0.40$

$P(B) = 1 - P((A \cup B)^c) - P(A\setminus B) = 1 - 0.42 - 0.28= 0.30$

$P(A \cup B) = 1 - P((A \cup B)^c) = 1 - 0.42 = 0.58$

$P(A^c \cap B) = P(B) - P(A \cap B)$ (by theorem 1.2.2) $= 0.3 - 0.12 = 0.18$



**Problem 7**

Find (calculate) the number of elements there are in the sigma algebra B for  
the following sample spaces S. (You need to justify your answers by describing  
and/or listing some of the outcomes in the sample space.)  



Problem (7a) S = { outcomes obtained by flipping 4 distinct coins once } (e.g. penny,  
nickel, dime, quarter)  

You have 2 choices per coin, and you flip 4 different coins.  This is "ordered" - that is to say each experiment has an associated coin, and each one has 2 possible states.  That means that the sample space has $2^4 = 16$ entries, and the Sigma Algebra $|\mathcal{B}| = 2^{16}
$

---

Problem (7b) S = { outcomes obtained by flipping 4 identical coins once } (e.g. four  
1919 pennies)  

The coins are identical, so they're unordered with replacement.

so, $|\mathcal{S}| = \frac{5!}{2! 3!} = 10$ but we don't count $\{HHTT\}$ any different from $\{TTHH\}$, so divide that by two again so really $|\mathcal{S}| = 5$.  That means the size of $\mathcal{B}$ is $|\mathcal{B}| = 2^5=32$.

---

Problem (7c) S = { outcomes obtained by flipping 4 coins once, where two of the coins are (say) indistinguishable pennies, and the other two are (say) indistin-  
guishable quarters }  



So, this one is a little easier to draw out, so I did:

![](/home/patrickpragman/.var/app/com.github.marktext.marktext/config/marktext/images/2022-09-08-21-15-51-IMG_0052.JPG)

So we have "coin type 1" and "coin type 2" in this experiment.  Coin type one can have one of the values that can be found in column one, and each one of those has the possibility of being followed by all three of the results in column two.

Ultimately, that means that $|\mathcal{S}| = 3\cdot3 = 9$ so $|\mathcal{B}| = 2^{9}=512$

---

Problem (7d)  S = { outcomes obtained by flipping a single coin 4 times }

It's a single coin, so we're looking at "ordered" with replacement, so $n^r = 4^2 = 16$ so $|\mathcal{B}| = 2^{16} = 65,536$ 

---

Problem (8)

Finally, shade the regions:

![](/home/patrickpragman/.var/app/com.github.marktext.marktext/config/marktext/images/2022-09-05-14-05-39-image.png)
